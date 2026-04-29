# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Step: Translate AYA dataset entries to a target language via LLM.

Handles the AYA-specific format (``inputs`` / ``targets`` fields) and
supports separate processing of train/test splits.  Native target-language
entries are copied directly without translation.  Concurrent batch
processing with checkpoint/resume.

The translation prompt and payload structure are preserved from the
original ``translate_aya_dataset_with_gpt5.py`` script.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from ..constants import AYA_SOURCE_LANGUAGE_CODES
from .common import (
    chunked,
    create_llm_translator,
    llm_json_call,
    load_jsonl,
    parse_model_json,
    save_jsonl,
)

logger = logging.getLogger("byol-data-prep")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _build_aya_payload(batch: List[Dict[str, Any]]) -> str:
    """Build user payload for AYA translation (inputs + targets)."""
    payload = {
        "items": [
            {
                "id": str(
                    item.get("translation_id", item.get("id", idx))
                ),
                "inputs": item.get("inputs", ""),
                "targets": item.get("targets", ""),
            }
            for idx, item in enumerate(batch)
        ]
    }
    return json.dumps(payload, ensure_ascii=False)


def _read_processed_ids_aya(output_path: str) -> Set[str]:
    """Read already-processed IDs from an AYA output file."""
    processed: Set[str] = set()
    if not os.path.exists(output_path):
        return processed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = (
                    obj.get("translation_id")
                    or obj.get("id")
                    or obj.get("_doc_id")
                    or obj.get("source_index")
                )
                if rid is not None:
                    processed.add(str(rid))
            except Exception:
                continue
    return processed


# ──────────────────────────────────────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────────────────────────────────────


def _process_aya_batch(
    translator: Any,
    batch: List[Dict[str, Any]],
    max_completion_tokens: int,
    bad_outputs_log: str,
    max_retries: int = 2,
) -> List[Dict[str, Any]]:
    """Translate a single AYA batch.  Returns translated records."""
    raw_output = ""
    user_payload = _build_aya_payload(batch)

    for attempt in range(max_retries + 1):
        try:
            raw_output = llm_json_call(translator, user_payload, max_completion_tokens)
            parsed = parse_model_json(raw_output)
            if not parsed or "items" not in parsed or not isinstance(parsed["items"], list):
                raise ValueError("Model returned invalid JSON structure")

            by_id = {
                str(rec.get("translation_id", rec.get("id", idx))): rec
                for idx, rec in enumerate(batch)
            }

            # Build translation mapping
            translated_map: Dict[str, Tuple[str, str]] = {}
            for translated in parsed["items"]:
                rec_id = str(translated.get("id"))
                if rec_id in by_id and rec_id not in translated_map:
                    inputs_trans = (translated.get("inputs_translation") or "").strip()
                    targets_trans = (translated.get("targets_translation") or "").strip()
                    translated_map[rec_id] = (inputs_trans, targets_trans)

            # Emit output records
            output_records: List[Dict[str, Any]] = []
            for rec in batch:
                rid = str(rec.get("translation_id", rec.get("id", "unknown")))
                updated = dict(rec)
                if rid in translated_map:
                    inputs_trans, targets_trans = translated_map[rid]
                    if inputs_trans and targets_trans:
                        updated["inputs"] = inputs_trans
                        updated["targets"] = targets_trans
                output_records.append(updated)

            return output_records

        except Exception as e:
            os.makedirs(os.path.dirname(bad_outputs_log) or ".", exist_ok=True)
            with open(bad_outputs_log, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "error": str(e),
                            "attempt": attempt,
                            "batch_ids": [
                                str(x.get("translation_id", x.get("id", idx)))
                                for idx, x in enumerate(batch)
                            ],
                            "raw_output_sample": raw_output[:2000] if raw_output else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if attempt == max_retries:
                logger.error(f"AYA translate batch failed after retries: {e}")
                return []

    return []


# ──────────────────────────────────────────────────────────────────────────────
# Single-split processing
# ──────────────────────────────────────────────────────────────────────────────


def _process_aya_split(
    *,
    input_file: str,
    output_file: str,
    tgt_lang_code: str,
    source_language_codes: List[str],
    system_prompt: str,
    model_name: str,
    reasoning_effort: str,
    api_version: str,
    batch_size: int,
    concurrency: int,
    checkpoint_every: int,
    max_completion_tokens: int,
    max_samples: int | None = None,
) -> str:
    """Process a single AYA split (train or test)."""
    bad_log = output_file + ".bad_outputs.log"

    if not os.path.exists(input_file):
        logger.warning(f"AYA input file not found: {input_file} — skipping.")
        return output_file

    data = load_jsonl(input_file)
    if not data:
        logger.info(f"No data in {input_file}.")
        return output_file

    # Ensure stable IDs
    for idx, item in enumerate(data):
        if "translation_id" not in item:
            item["translation_id"] = str(item.get("id", idx))

    # Resume support
    processed_ids = _read_processed_ids_aya(output_file)
    if processed_ids:
        logger.info(f"Resuming: {len(processed_ids)} already-processed IDs in {output_file}")

    # Split into native (direct copy) vs needs-translation
    to_translate: List[Dict[str, Any]] = []
    direct_copy: List[Dict[str, Any]] = []

    for record in data:
        rid = str(record.get("translation_id", record.get("id", "")))
        if rid in processed_ids:
            continue
        lang_code = record.get("language_code", "")
        if lang_code == tgt_lang_code:
            direct_copy.append(record)
        elif lang_code in source_language_codes:
            to_translate.append(record)

    logger.info(f"AYA split: {len(to_translate)} to translate, {len(direct_copy)} native (direct copy)")

    # Write native entries first
    if direct_copy:
        save_jsonl(direct_copy, output_file)
        logger.info(f"  Added {len(direct_copy)} native entries")

    if not to_translate:
        logger.info("  No entries to translate in this split.")
        return output_file

    # Apply max_samples cap
    if max_samples is not None and len(to_translate) > max_samples:
        logger.info(f"  --max-samples: capping from {len(to_translate):,} to {max_samples:,}")
        to_translate = to_translate[:max_samples]

    # Create LLM client
    translator = create_llm_translator(
        model_name=model_name,
        system_prompt=system_prompt,
        reasoning_effort=reasoning_effort,
        max_tokens=max_completion_tokens,
        api_version=api_version,
    )

    batches = chunked(to_translate, batch_size)
    logger.info(f"  Translating {len(to_translate):,} AYA items in {len(batches):,} batches")

    pending: List[Dict[str, Any]] = []
    batches_done = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_idx = {
            pool.submit(
                _process_aya_batch, translator, batch, max_completion_tokens, bad_log
            ): idx
            for idx, batch in enumerate(batches)
        }

        results_by_idx: list = [None] * len(batches)
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Translating AYA",
        ):
            idx = future_to_idx[future]
            results_by_idx[idx] = future.result()
            batches_done += 1

    # Flatten in original order
    for result in results_by_idx:
        if result:
            pending.extend(result)

    if pending:
        save_jsonl(pending, output_file)

    logger.info(f"  AYA split translation complete → {output_file}")
    return output_file


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ──────────────────────────────────────────────────────────────────────────────


def run_translate_aya(
    *,
    train_input_file: str,
    test_input_file: str,
    train_output_file: str,
    test_output_file: str,
    tgt_lang_code: str,
    system_prompt: str,
    model_name: str,
    reasoning_effort: str = "low",
    api_version: str = "2024-08-01-preview",
    batch_size: int = 8,
    concurrency: int = 96,
    checkpoint_every: int = 50,
    token_budget_per_item: int = 2048,
    max_completion_tokens: int | None = None,
    source_language_codes: Optional[List[str]] = None,
    max_samples: int | None = None,
) -> Tuple[str, str]:
    """Run the AYA translation pipeline for both train and test splits.

    Args:
        train_input_file: Path to filtered AYA train JSONL.
        test_input_file: Path to filtered AYA test JSONL.
        train_output_file: Output path for translated train JSONL.
        test_output_file: Output path for translated test JSONL.
        tgt_lang_code: Target language ISO 639-3 code.
        system_prompt: LLM system prompt for AYA translation.
        model_name: Azure deployment name.
        reasoning_effort: Reasoning effort level.
        api_version: Azure OpenAI API version.
        batch_size: Items per LLM call.
        concurrency: Max parallel requests.
        checkpoint_every: Flush to disk every N batches.
        token_budget_per_item: Heuristic per-item token budget.
        max_completion_tokens: Override auto-computed token budget.
        source_language_codes: Source languages for translation.
        max_samples: If set, cap per-split samples.

    Returns:
        Tuple of (train_output_file, test_output_file).
    """
    if max_completion_tokens is None:
        max_completion_tokens = int(token_budget_per_item * batch_size * 2)

    if source_language_codes is None:
        source_language_codes = list(AYA_SOURCE_LANGUAGE_CODES)

    for label, input_file, output_file in [
        ("train", train_input_file, train_output_file),
        ("test", test_input_file, test_output_file),
    ]:
        logger.info(f"Processing AYA {label} split...")
        _process_aya_split(
            input_file=input_file,
            output_file=output_file,
            tgt_lang_code=tgt_lang_code,
            source_language_codes=source_language_codes,
            system_prompt=system_prompt,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            api_version=api_version,
            batch_size=batch_size,
            concurrency=concurrency,
            checkpoint_every=checkpoint_every,
            max_completion_tokens=max_completion_tokens,
            max_samples=max_samples,
        )

    return train_output_file, test_output_file
