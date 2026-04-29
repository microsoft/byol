# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Step: Refine / clean text via LLM (target-language or English).

Sends batches of JSONL records to an LLM for rewriting — improving
clarity, fixing grammar, and optionally dropping non-target-language
items.  Supports concurrent batch processing with checkpoint/resume.

Two modes controlled by *mode*:

* ``"tgt_lang"`` — refine target-language text (FineWeb-2).  Non-target-
  language items are dropped and logged to a ``*.removed.jsonl`` sidecar.
* ``"eng"`` — clean English text (FineWeb-Edu).  All items are kept.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .common import (
    append_removed_records,
    build_user_payload,
    chunked,
    create_llm_translator,
    ensure_stable_ids,
    llm_json_call,
    load_jsonl,
    parse_model_json,
    read_processed_ids,
    read_removed_ids,
    save_jsonl,
)

logger = logging.getLogger("byol-data-prep")


# ──────────────────────────────────────────────────────────────────────────────
# Single-batch processing
# ──────────────────────────────────────────────────────────────────────────────


def _process_refine_tgt_lang_batch(
    translator,
    batch: List[Dict[str, Any]],
    max_completion_tokens: int,
    bad_outputs_log: str,
    max_retries: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Refine target-language batch.  Returns (refined, removed)."""
    raw_output = ""
    user_payload = build_user_payload(batch)

    for attempt in range(max_retries + 1):
        try:
            raw_output = llm_json_call(translator, user_payload, max_completion_tokens)
            parsed = parse_model_json(raw_output)
            if not parsed or "items" not in parsed or not isinstance(parsed["items"], list):
                raise ValueError("Model returned invalid JSON structure")

            by_id = {str(rec["id"]): rec for rec in batch}
            refined_map: Dict[str, str] = {}
            for rec in parsed["items"]:
                rec_id = str(rec.get("id"))
                if rec_id in by_id and rec_id not in refined_map:
                    refined_map[rec_id] = (rec.get("refined") or "").strip()

            output_records: List[Dict[str, Any]] = []
            removed_records: List[Dict[str, Any]] = []
            for rec in batch:
                rid = str(rec["id"])
                original_text = rec.get("text", "")
                refined_text = refined_map.get(rid, "")
                if refined_text:
                    updated = dict(rec)
                    updated["text"] = refined_text
                    output_records.append(updated)
                else:
                    removed_records.append({
                        "id": rid,
                        "original_text": original_text,
                        "refined_text": refined_text,
                        "reason": "empty_refined_or_non_target_lang",
                    })
            return output_records, removed_records

        except Exception as e:
            _log_bad_output(bad_outputs_log, e, attempt, batch, raw_output)
            if attempt == max_retries:
                logger.error(f"Refine tgt_lang batch failed after retries: {e}")
                return [], []


def _process_refine_eng_batch(
    translator,
    batch: List[Dict[str, Any]],
    max_completion_tokens: int,
    bad_outputs_log: str,
    max_retries: int = 2,
) -> List[Dict[str, Any]]:
    """Clean English batch.  Returns list of cleaned records."""
    raw_output = ""
    user_payload = build_user_payload(batch)

    for attempt in range(max_retries + 1):
        try:
            raw_output = llm_json_call(translator, user_payload, max_completion_tokens)
            parsed = parse_model_json(raw_output)
            if not parsed or "items" not in parsed or not isinstance(parsed["items"], list):
                raise ValueError("Model returned invalid JSON structure")

            by_id = {str(rec["id"]): rec for rec in batch}
            cleaned_map: Dict[str, str] = {}
            for rec in parsed["items"]:
                rec_id = str(rec.get("id"))
                if rec_id in by_id and rec_id not in cleaned_map:
                    cleaned_map[rec_id] = (rec.get("cleaned") or "").strip()

            output_records: List[Dict[str, Any]] = []
            for rec in batch:
                rid = str(rec["id"])
                updated = dict(rec)
                if rid in cleaned_map:
                    updated["text"] = cleaned_map[rid]
                output_records.append(updated)
            return output_records

        except Exception as e:
            _log_bad_output(bad_outputs_log, e, attempt, batch, raw_output)
            if attempt == max_retries:
                logger.error(f"Refine eng batch failed after retries: {e}")
                return []

    return []  # unreachable but keeps type-checker happy


def _log_bad_output(
    log_path: str, error: Exception, attempt: int,
    batch: List[Dict[str, Any]], raw_output: str,
) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "error": str(error),
            "attempt": attempt,
            "batch_ids": [str(x["id"]) for x in batch],
            "raw_output_sample": raw_output[:2000] if raw_output else None,
        }, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ──────────────────────────────────────────────────────────────────────────────


def run_refine(
    *,
    mode: str,
    input_file: str,
    output_file: str,
    system_prompt: str,
    model_name: str,
    reasoning_effort: str = "low",
    api_version: str = "2024-08-01-preview",
    batch_size: int = 4,
    concurrency: int = 32,
    checkpoint_every: int = 50,
    token_budget_per_item: int = 2048,
    max_completion_tokens: int | None = None,
    max_samples: int | None = None,
) -> str:
    """Run the refinement/cleaning pipeline.

    Args:
        mode: ``"tgt_lang"`` or ``"eng"``.
        input_file: Path to input JSONL.
        output_file: Path to output JSONL.
        system_prompt: LLM system prompt.
        model_name: Azure deployment name (e.g. ``"gpt-5"``).
        reasoning_effort: Reasoning effort level.
        api_version: Azure OpenAI API version.
        batch_size: Items per LLM call.
        concurrency: Max parallel requests.
        checkpoint_every: Flush to disk every N batches.
        token_budget_per_item: Heuristic per-item token budget.
        max_completion_tokens: Override auto-computed budget.
        max_samples: If set, only process the first N records.

    Returns:
        Path to the output file.
    """
    if mode not in ("tgt_lang", "eng"):
        raise ValueError(f"Invalid refine mode: {mode!r}. Must be 'tgt_lang' or 'eng'.")

    if max_completion_tokens is None:
        max_completion_tokens = int(token_budget_per_item * batch_size * 2)

    removed_path = output_file + ".removed.jsonl"
    bad_log = output_file + ".bad_outputs.log"

    # Create LLM client via AzureOpenAIGPT5Translator
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
    removed_ids = read_removed_ids(removed_path) if mode == "tgt_lang" else set()
    already_done = processed_ids | removed_ids
    if already_done:
        logger.info(
            f"Resuming: {len(processed_ids)} written, {len(removed_ids)} removed "
            f"({len(already_done)} total)"
        )

    to_process = [r for r in data if str(r["id"]) not in already_done]
    if max_samples is not None and len(to_process) > max_samples:
        logger.info(f"--max-samples: capping from {len(to_process):,} to {max_samples:,} records")
        to_process = to_process[:max_samples]
    if not to_process:
        logger.info("Nothing to do — all records already processed.")
        return output_file

    batches = chunked(to_process, batch_size)
    total_items = len(to_process)
    logger.info(f"Processing {total_items:,} items in {len(batches):,} batches (mode={mode})")

    pending: List[Dict[str, Any]] = []
    pending_removed: List[Dict[str, Any]] = []
    batches_done = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        if mode == "tgt_lang":
            future_to_idx = {
                pool.submit(
                    _process_refine_tgt_lang_batch,
                    translator, batch, max_completion_tokens, bad_log,
                ): idx
                for idx, batch in enumerate(batches)
            }
        else:
            future_to_idx = {
                pool.submit(
                    _process_refine_eng_batch,
                    translator, batch, max_completion_tokens, bad_log,
                ): idx
                for idx, batch in enumerate(batches)
            }

        # Collect results indexed by batch position for ordered output
        results_by_idx: list = [None] * len(batches)
        desc = "Refining (tgt_lang)" if mode == "tgt_lang" else "Cleaning (eng)"
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc=desc):
            idx = future_to_idx[future]
            results_by_idx[idx] = future.result()
            batches_done += 1

    # Flatten results in original input order
    for result in results_by_idx:
        if result is None:
            continue
        if mode == "tgt_lang":
            refined_batch, removed_batch = result
            if refined_batch:
                pending.extend(refined_batch)
            if removed_batch:
                pending_removed.extend(removed_batch)
        else:
            if result:
                pending.extend(result)

    # Write all results
    if pending:
        save_jsonl(pending, output_file)
    if pending_removed:
        append_removed_records(removed_path, pending_removed)

    logger.info(f"Refinement complete → {output_file}")
    return output_file
