# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Step: Translate English text to a target language via LLM.

Sends batches of JSONL records to an LLM for translation.  Supports
concurrent batch processing with checkpoint/resume.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .common import (
    build_user_payload,
    chunked,
    create_llm_translator,
    ensure_stable_ids,
    llm_json_call,
    load_jsonl,
    parse_model_json,
    read_processed_ids,
    save_jsonl,
)

logger = logging.getLogger("byol-data-prep")


# ──────────────────────────────────────────────────────────────────────────────
# Single-batch processing
# ──────────────────────────────────────────────────────────────────────────────


def _process_translate_batch(
    translator,
    batch: List[Dict[str, Any]],
    max_completion_tokens: int,
    bad_outputs_log: str,
    max_retries: int = 2,
) -> List[Dict[str, Any]]:
    """Translate a single batch.  Returns list of translated records."""
    raw_output = ""
    user_payload = build_user_payload(batch)

    for attempt in range(max_retries + 1):
        try:
            raw_output = llm_json_call(translator, user_payload, max_completion_tokens)
            parsed = parse_model_json(raw_output)
            if not parsed or "items" not in parsed or not isinstance(parsed["items"], list):
                raise ValueError("Model returned invalid JSON structure")

            by_id = {str(rec["id"]): rec for rec in batch}
            translated_map: Dict[str, str] = {}
            for rec in parsed["items"]:
                rec_id = str(rec.get("id"))
                if rec_id in by_id and rec_id not in translated_map:
                    translated_map[rec_id] = (rec.get("translation") or "").strip()

            output_records: List[Dict[str, Any]] = []
            for rec in batch:
                rid = str(rec["id"])
                updated = dict(rec)
                if rid in translated_map and translated_map[rid]:
                    updated["text"] = translated_map[rid]
                output_records.append(updated)
            return output_records

        except Exception as e:
            os.makedirs(os.path.dirname(bad_outputs_log) or ".", exist_ok=True)
            with open(bad_outputs_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "error": str(e),
                    "attempt": attempt,
                    "batch_ids": [str(x["id"]) for x in batch],
                    "raw_output_sample": raw_output[:2000] if raw_output else None,
                }, ensure_ascii=False) + "\n")

            if attempt == max_retries:
                logger.error(f"Translate batch failed after retries: {e}")
                return []

    return []


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ──────────────────────────────────────────────────────────────────────────────


def run_translate(
    *,
    input_file: str,
    output_file: str,
    system_prompt: str,
    model_name: str,
    reasoning_effort: str = "low",
    api_version: str = "2024-08-01-preview",
    batch_size: int = 16,
    concurrency: int = 96,
    checkpoint_every: int = 5000,
    token_budget_per_item: int = 1536,
    max_completion_tokens: int | None = None,
    max_samples: int | None = None,
) -> str:
    """Run the translation pipeline.

    Args:
        input_file: Path to input JSONL (English text).
        output_file: Path to output JSONL (translated text).
        system_prompt: LLM system prompt for translation.
        model_name: Azure deployment name (e.g. ``"gpt-5"``).
        reasoning_effort: Reasoning effort level.
        api_version: Azure OpenAI API version.
        batch_size: Items per LLM call.
        concurrency: Max parallel requests.
        checkpoint_every: Flush to disk every N batches.
        token_budget_per_item: Heuristic per-item token budget.
        max_completion_tokens: Override auto-computed token budget.
        max_samples: If set, only process the first N records.

    Returns:
        Path to the output file.
    """
    if max_completion_tokens is None:
        max_completion_tokens = int(token_budget_per_item * batch_size * 2)

    bad_log = output_file + ".bad_outputs.log"

    # Create LLM client
    translator = create_llm_translator(
        model_name=model_name,
        system_prompt=system_prompt,
        reasoning_effort=reasoning_effort,
        max_tokens=max_completion_tokens,
        api_version=api_version,
    )

    # Load data
    data = load_jsonl(input_file)
    ensure_stable_ids(data)

    # Resume support
    processed_ids = read_processed_ids(output_file)
    if processed_ids:
        logger.info(f"Resuming: {len(processed_ids)} already-processed IDs found.")

    to_process = [r for r in data if str(r["id"]) not in processed_ids]
    if max_samples is not None and len(to_process) > max_samples:
        logger.info(f"--max-samples: capping from {len(to_process):,} to {max_samples:,} records")
        to_process = to_process[:max_samples]
    if not to_process:
        logger.info("Nothing to do — all records already translated.")
        return output_file

    batches = chunked(to_process, batch_size)
    total_items = len(to_process)
    logger.info(f"Translating {total_items:,} items in {len(batches):,} batches")

    pending: List[Dict[str, Any]] = []
    batches_done = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_idx = {
            pool.submit(
                _process_translate_batch,
                translator, batch, max_completion_tokens, bad_log,
            ): idx
            for idx, batch in enumerate(batches)
        }

        # Collect results indexed by batch position for ordered output
        results_by_idx: list = [None] * len(batches)
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Translating"):
            idx = future_to_idx[future]
            results_by_idx[idx] = future.result()
            batches_done += 1

    # Flatten results in original input order
    for result in results_by_idx:
        if result:
            pending.extend(result)

    # Write all results
    if pending:
        save_jsonl(pending, output_file)

    logger.info(f"Translation complete → {output_file}")
    return output_file
