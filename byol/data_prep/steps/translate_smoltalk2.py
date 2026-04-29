# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Step: Translate SmolTalk2 conversation dataset to a target language via LLM.

Sends batches of SmolTalk2 records (with ``messages`` and
``chat_template_kwargs``) to an LLM for translation.  Supports
concurrent batch processing with checkpoint/resume.

The translation prompt and payload structure are preserved from the
original ``c_translate_smoltalk2_combined_instruction_dataset_with_gpt5.py``.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .common import (
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
# Helpers (from the original smoltalk2 translation script)
# ──────────────────────────────────────────────────────────────────────────────


def _extract_custom_instructions(chat_template_kwargs: Any) -> str:
    """Extract custom_instructions from chat_template_kwargs (dict or JSON string)."""
    if not chat_template_kwargs:
        return ""
    if isinstance(chat_template_kwargs, dict):
        return chat_template_kwargs.get("custom_instructions", "") or ""
    if isinstance(chat_template_kwargs, str) and not chat_template_kwargs.strip():
        return ""
    try:
        kwargs_dict = json.loads(chat_template_kwargs)
        return kwargs_dict.get("custom_instructions", "") or ""
    except (json.JSONDecodeError, TypeError):
        return ""


def _reconstruct_chat_template_kwargs(
    original_kwargs: Any, new_custom_instructions: str
) -> str:
    """Reconstruct chat_template_kwargs with translated custom_instructions."""
    if not original_kwargs:
        return original_kwargs if isinstance(original_kwargs, str) else ""
    if isinstance(original_kwargs, dict):
        kwargs_dict = dict(original_kwargs)
        if "custom_instructions" in kwargs_dict:
            kwargs_dict["custom_instructions"] = new_custom_instructions
        return json.dumps(kwargs_dict, ensure_ascii=False)
    if isinstance(original_kwargs, str) and not original_kwargs.strip():
        return original_kwargs
    try:
        kwargs_dict = json.loads(original_kwargs)
        if "custom_instructions" in kwargs_dict:
            kwargs_dict["custom_instructions"] = new_custom_instructions
        return json.dumps(kwargs_dict, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return original_kwargs


def _build_smoltalk2_payload(batch: List[Dict[str, Any]]) -> str:
    """Build user payload for SmolTalk2 translation (messages + custom_instructions)."""
    payload_items = []
    for item in batch:
        item_data: Dict[str, Any] = {
            "id": str(item["id"]),
            "messages": [],
            "custom_instructions": "",
        }
        # Extract messages
        for msg in item.get("messages", []):
            if isinstance(msg, dict) and "content" in msg and "role" in msg:
                item_data["messages"].append(
                    {"content": msg["content"], "role": msg["role"]}
                )
        # Extract custom instructions
        item_data["custom_instructions"] = _extract_custom_instructions(
            item.get("chat_template_kwargs", "")
        )
        payload_items.append(item_data)

    return json.dumps({"items": payload_items}, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────────────────────────────────────


def _process_smoltalk2_batch(
    translator: Any,
    batch: List[Dict[str, Any]],
    max_completion_tokens: int,
    bad_outputs_log: str,
    max_retries: int = 2,
) -> List[Dict[str, Any]]:
    """Translate a single SmolTalk2 batch.  Returns translated records."""
    raw_output = ""
    user_payload = _build_smoltalk2_payload(batch)

    for attempt in range(max_retries + 1):
        try:
            raw_output = llm_json_call(translator, user_payload, max_completion_tokens)
            parsed = parse_model_json(raw_output)
            if not parsed or "items" not in parsed or not isinstance(parsed["items"], list):
                raise ValueError("Model returned invalid JSON structure")

            by_id = {str(rec["id"]): rec for rec in batch}

            # Build translation mapping
            translated_map: Dict[str, Dict[str, Any]] = {}
            for translated in parsed["items"]:
                rec_id = str(translated.get("id"))
                if rec_id in by_id and rec_id not in translated_map:
                    translated_map[rec_id] = {
                        "messages": translated.get("messages", []),
                        "custom_instructions": translated.get("custom_instructions", ""),
                    }

            # Reconstruct output records with translations
            output_records: List[Dict[str, Any]] = []
            for rec in batch:
                rid = str(rec["id"])
                updated = dict(rec)

                if rid in translated_map:
                    translation_data = translated_map[rid]

                    # Update messages with translations
                    translated_messages = translation_data.get("messages", [])
                    orig_messages = rec.get("messages", [])
                    if translated_messages and len(translated_messages) == len(orig_messages):
                        new_messages = []
                        for i, orig_msg in enumerate(orig_messages):
                            new_msg = dict(orig_msg)
                            if i < len(translated_messages):
                                new_msg["content"] = translated_messages[i].get(
                                    "content", orig_msg.get("content", "")
                                )
                            new_messages.append(new_msg)
                        updated["messages"] = new_messages

                    # Update custom_instructions in chat_template_kwargs
                    original_kwargs = rec.get("chat_template_kwargs", "")
                    new_ci = translation_data.get("custom_instructions", "")
                    if original_kwargs and _extract_custom_instructions(original_kwargs):
                        updated["chat_template_kwargs"] = _reconstruct_chat_template_kwargs(
                            original_kwargs, new_ci
                        )

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
                            "batch_ids": [str(x["id"]) for x in batch],
                            "raw_output_sample": raw_output[:2000] if raw_output else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if attempt == max_retries:
                logger.error(f"SmolTalk2 translate batch failed after retries: {e}")
                return []

    return []


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ──────────────────────────────────────────────────────────────────────────────


def run_translate_smoltalk2(
    *,
    input_file: str,
    output_file: str,
    system_prompt: str,
    model_name: str,
    reasoning_effort: str = "low",
    api_version: str = "2024-08-01-preview",
    batch_size: int = 8,
    concurrency: int = 64,
    checkpoint_every: int = 50,
    token_budget_per_item: int = 2048,
    max_completion_tokens: int | None = None,
    max_samples: int | None = None,
) -> str:
    """Run the SmolTalk2 translation pipeline.

    Args:
        input_file: Path to combined SmolTalk2 JSONL (English).
        output_file: Path to output JSONL (translated).
        system_prompt: LLM system prompt for SmolTalk2 translation.
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
        logger.info("Nothing to do — all SmolTalk2 records already translated.")
        return output_file

    batches = chunked(to_process, batch_size)
    logger.info(
        f"Translating {len(to_process):,} SmolTalk2 items in {len(batches):,} batches"
    )

    pending: List[Dict[str, Any]] = []
    batches_done = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_idx = {
            pool.submit(
                _process_smoltalk2_batch,
                translator,
                batch,
                max_completion_tokens,
                bad_log,
            ): idx
            for idx, batch in enumerate(batches)
        }

        results_by_idx: list = [None] * len(batches)
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Translating SmolTalk2",
        ):
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

    logger.info(f"SmolTalk2 translation complete → {output_file}")
    return output_file
