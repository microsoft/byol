# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Translate evaluation benchmark datasets from English → target language.

This module handles the core translation logic for all supported eval benchmarks.
It uses the ``byol.translation_backends`` package to perform translations, supporting
both API-based translators (Microsoft, Google) and LLM-based translators (GPT-5).

Field types handled:
    - **Direct string** (e.g. ``"question"``): Translate the string value.
    - **Nested dict list** (e.g. ``"choices.text"``): Access ``row["choices"]["text"]``
      and translate each string in the resulting list.
    - **Top-level list** (e.g. ``"endings"``): Translate each string in the list.
    - **Top-level list of strings** (e.g. ``"correct_answers"``): Translate each.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from byol.common.translator_support import validate_translator_for_language
from byol.translation_backends.factory import get_translator

logger = logging.getLogger("byol-data-prep")

# Maximum fraction of translation failures before aborting
_MAX_FAILURE_RATE = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Field access helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_field_value(row: Dict[str, Any], field: str) -> Any:
    """Retrieve a (possibly nested) field value from a row dict.

    Supports dot-notation for one level of nesting:
        ``"choices.text"`` → ``row["choices"]["text"]``
    """
    if "." in field:
        parent_key, child_key = field.split(".", 1)
        parent = row.get(parent_key)
        if parent is None:
            return None
        if isinstance(parent, dict):
            return parent.get(child_key)
        return None
    return row.get(field)


def _set_field_value(row: Dict[str, Any], field: str, value: Any) -> None:
    """Set a (possibly nested) field value in a row dict."""
    if "." in field:
        parent_key, child_key = field.split(".", 1)
        if parent_key not in row:
            row[parent_key] = {}
        row[parent_key][child_key] = value
    else:
        row[field] = value


# ─────────────────────────────────────────────────────────────────────────────
# Translation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _translate_text(
    text: str,
    translator_model: str,
    tgt_lang: str,
    src_lang: str = "en",
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Translate a single text string using the translation backends factory.

    Creates a fresh translator instance per call (no caching), which ensures
    that custom ``system_prompt`` / ``max_tokens`` are honoured for LLM backends.
    """
    if not text or not text.strip():
        return text

    kwargs: Dict[str, Any] = {}
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    translator = get_translator(
        translator_model,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        **kwargs,
    )
    try:
        return translator.translate(text)
    except Exception:
        logger.warning("Translation failed for text: %s…", text[:80], exc_info=True)
        raise


def _translate_field_value(
    value: Any,
    translator_model: str,
    tgt_lang: str,
    src_lang: str = "en",
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Any:
    """Translate a field value which may be a string or a list of strings."""
    if isinstance(value, str):
        return _translate_text(
            value, translator_model, tgt_lang, src_lang, system_prompt, max_tokens
        )
    if isinstance(value, list):
        return [
            _translate_text(
                item, translator_model, tgt_lang, src_lang, system_prompt, max_tokens
            )
            if isinstance(item, str)
            else item
            for item in value
        ]
    # Non-translatable (int, None, etc.) — return as-is
    return value


# ─────────────────────────────────────────────────────────────────────────────
# Row-level translation
# ─────────────────────────────────────────────────────────────────────────────


def translate_row(
    row: Dict[str, Any],
    fields: Sequence[str],
    translator_model: str,
    tgt_lang: str,
    src_lang: str = "en",
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Translate specified fields of a single row in-place and return it."""
    for field in fields:
        original = _get_field_value(row, field)
        if original is None:
            continue
        translated = _translate_field_value(
            original,
            translator_model,
            tgt_lang,
            src_lang,
            system_prompt,
            max_tokens,
        )
        _set_field_value(row, field, translated)
    return row


def _translate_row_with_translator(
    row: Dict[str, Any],
    fields: Sequence[str],
    translator: Any,
    tgt_lang: str,
) -> Dict[str, Any]:
    """Translate fields using a pre-created translator instance (no re-init)."""
    for field in fields:
        original = _get_field_value(row, field)
        if original is None:
            continue
        if isinstance(original, str):
            if original.strip():
                translated = translator.translate(original)
            else:
                translated = original
        elif isinstance(original, list):
            translated = [
                translator.translate(item) if isinstance(item, str) and item.strip() else item
                for item in original
            ]
        else:
            translated = original
        _set_field_value(row, field, translated)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_local_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return a list of dicts."""
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_hf_dataset(
    dataset_path: str,
    dataset_config: Optional[str],
    split: str,
    hf_loader: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load a split from a HuggingFace dataset and convert to list of dicts.

    Lazily imports ``datasets`` to avoid heavy startup cost.
    For datasets with legacy scripts (e.g. juletxara/mgsm), use a custom
    ``hf_loader`` to download raw files directly.
    """
    if hf_loader:
        loader_fn = HF_CUSTOM_LOADERS.get(hf_loader)
        if loader_fn:
            return loader_fn(dataset_path, dataset_config, split)

    from datasets import load_dataset  # type: ignore[import-untyped]

    kwargs: Dict[str, Any] = {"path": dataset_path, "split": split}
    if dataset_config:
        kwargs["name"] = dataset_config

    logger.info(
        "Loading HF dataset: %s (config=%s, split=%s)",
        dataset_path,
        dataset_config,
        split,
    )
    ds = load_dataset(**kwargs)
    return [dict(row) for row in ds]  # type: ignore[union-attr]


def _load_mgsm(dataset_path: str, dataset_config: Optional[str], split: str) -> List[Dict[str, Any]]:
    """Custom loader for juletxara/mgsm (uses legacy script, not compatible with datasets>=4.0).

    Downloads the TSV (test) or exemplars.py (train) directly from the repo.
    """
    from huggingface_hub import hf_hub_download
    import csv

    lang = dataset_config or "en"

    if split == "test":
        tsv_path = hf_hub_download(dataset_path, f"mgsm_{lang}.tsv", repo_type="dataset")
        rows = []
        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    rows.append({
                        "question": row[0],
                        "answer": None,
                        "answer_number": int(row[1].replace(",", "")),
                        "equation_solution": None,
                    })
        logger.info("Loaded MGSM %s/%s: %d rows from TSV", lang, split, len(rows))
        return rows

    elif split == "train":
        exemplars_path = hf_hub_download(dataset_path, "exemplars.py", repo_type="dataset")
        # exemplars.py defines EXEMPLARS dict and EXEMPLAR_NUMBER_ANSWERS list
        namespace: Dict[str, Any] = {}
        with open(exemplars_path, encoding="utf-8") as f:
            exec(f.read(), namespace)
        exemplars = namespace.get("MGSM_EXEMPLARS", namespace.get("EXEMPLARS", {}))
        number_answers = namespace.get("EXEMPLAR_NUMBER_ANSWERS", [])
        lang_exemplars = exemplars.get(lang, {})
        rows = []
        for key in sorted(lang_exemplars.keys(), key=int):
            entry = lang_exemplars[key]
            idx = int(key) - 1
            rows.append({
                "question": entry["q"],
                "answer": entry["a"],
                "answer_number": number_answers[idx] if idx < len(number_answers) else 0,
                "equation_solution": None,
            })
        logger.info("Loaded MGSM %s/%s: %d rows from exemplars", lang, split, len(rows))
        return rows

    raise ValueError(f"Unknown MGSM split: {split}")


HF_CUSTOM_LOADERS: Dict[str, Any] = {
    "_mgsm": _load_mgsm,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main translation function for a single benchmark + split
# ─────────────────────────────────────────────────────────────────────────────


def translate_benchmark_split(
    *,
    benchmark_name: str,
    split: str,
    rows: List[Dict[str, Any]],
    fields: Sequence[str],
    translator_model: str,
    tgt_lang: str,
    output_path: str,
    src_lang: str = "en",
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_workers: int = 8,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
) -> str:
    """Translate all rows for a benchmark split and write to JSONL.

    Uses :class:`concurrent.futures.ThreadPoolExecutor` for parallel
    translation of rows (I/O-bound API calls).

    Args:
        benchmark_name: Benchmark identifier (e.g. ``"copa"``).
        split: Dataset split name (e.g. ``"test"``).
        rows: List of row dicts to translate.
        fields: Field names to translate.
        translator_model: Model name for ``get_translator()``.
        tgt_lang: Target language for the translator. For API translators
                  this is typically a language code (``"nya"``); for LLM
                  translators it may be a full name (``"Chichewa"``).
        output_path: Path for the output JSONL file.
        src_lang: Source language code (default ``"en"``).
        system_prompt: Optional system prompt for LLM translators.
        max_tokens: Optional max_tokens for LLM translators.
        max_workers: Number of parallel threads.
        batch_size: Not used directly — kept for API parity.
        max_samples: If set, only translate this many rows.

    Returns:
        The ``output_path`` written to.
    """
    # Validate translator supports the target language before starting
    validate_translator_for_language(translator_model, tgt_lang)

    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]

    total = len(rows)
    logger.info(
        "Translating %s/%s (%d rows, translator=%s, workers=%d)",
        benchmark_name,
        split,
        total,
        translator_model,
        max_workers,
    )

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Create translator ONCE and share across threads/batches.
    from byol.translation_backends.registry import MODEL_REGISTRY
    model_cfg = MODEL_REGISTRY.get(translator_model)
    is_local = model_cfg and model_cfg.model_type == "local"

    shared_kwargs: Dict[str, Any] = {}
    if system_prompt is not None:
        shared_kwargs["system_prompt"] = system_prompt
    if max_tokens is not None:
        shared_kwargs["max_tokens"] = max_tokens

    shared_translator = get_translator(
        translator_model,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        **shared_kwargs,
    )

    translated_rows: List[Optional[Dict[str, Any]]] = [None] * total
    fail_count = 0

    if is_local:
        # ── Local models: batched translation for GPU efficiency ──────────
        # Collect all translatable texts, batch-translate, redistribute.
        logger.info("Using batched translation for local model %s (batch_size=%d)", translator_model, batch_size)

        # Step 1: extract all texts to translate with their (row_idx, field, position) coords
        text_entries: list[tuple[int, str, int | None, str]] = []  # (row_idx, field, list_pos, text)
        for i, row in enumerate(rows):
            for fld in fields:
                val = _get_field_value(row, fld)
                if val is None:
                    continue
                if isinstance(val, str):
                    if val.strip():
                        text_entries.append((i, fld, None, val))
                elif isinstance(val, list):
                    for j, item in enumerate(val):
                        if isinstance(item, str) and item.strip():
                            text_entries.append((i, fld, j, item))

        all_texts = [entry[3] for entry in text_entries]
        logger.info("  %d texts to translate across %d rows × %d fields", len(all_texts), total, len(fields))

        # Step 2: batch translate
        all_translated: list[str] = []
        for batch_start in range(0, len(all_texts), batch_size):
            batch = all_texts[batch_start:batch_start + batch_size]
            try:
                batch_result = shared_translator.translate_batch(texts=batch, batch_size=batch_size)
                all_translated.extend(batch_result)
            except Exception:
                logger.warning("Batch %d failed, falling back to per-item", batch_start // batch_size)
                for text in batch:
                    try:
                        all_translated.append(shared_translator.translate(text))
                    except Exception:
                        all_translated.append(text)
                        fail_count += 1

        # Step 3: redistribute translations back into rows
        # Deep-copy rows first so originals aren't modified
        import copy
        row_copies = [copy.deepcopy(row) for row in rows]
        for (row_idx, fld, list_pos, _orig), translated_text in zip(text_entries, all_translated):
            if list_pos is None:
                _set_field_value(row_copies[row_idx], fld, translated_text)
            else:
                current = _get_field_value(row_copies[row_idx], fld)
                if isinstance(current, list) and list_pos < len(current):
                    current[list_pos] = translated_text

        for i, row in enumerate(row_copies):
            translated_rows[i] = row

        done_count = total
        logger.info(
            "  [%s/%s] %d / %d rows translated (%d failures)",
            benchmark_name, split, done_count, total, fail_count,
        )
    else:
        # ── API models: parallel threads (I/O-bound) ─────────────────────
        done_count = 0

        def _translate_single(idx: int, row: Dict[str, Any]) -> tuple[int, Dict[str, Any], bool]:
            try:
                translated = _translate_row_with_translator(
                    row,
                    fields=fields,
                    translator=shared_translator,
                    tgt_lang=tgt_lang,
                )
                return idx, translated, True
            except Exception:
                logger.warning(
                    "Row %d translation failed for %s/%s", idx, benchmark_name, split,
                    exc_info=True,
                )
                return idx, row, False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_translate_single, i, row): i
                for i, row in enumerate(rows)
            }
            for future in as_completed(futures):
                idx, translated_row, success = future.result()
                translated_rows[idx] = translated_row
                done_count += 1
                if not success:
                    fail_count += 1
                if done_count % 100 == 0 or done_count == total:
                    logger.info(
                        "  [%s/%s] %d / %d rows translated (%d failures)",
                        benchmark_name,
                        split,
                        done_count,
                        total,
                        fail_count,
                    )

    # Abort if too many failures
    if total > 0 and fail_count / total > _MAX_FAILURE_RATE:
        raise RuntimeError(
            f"Translation aborted for {benchmark_name}/{split}: "
            f"{fail_count}/{total} rows failed "
            f"({fail_count / total:.0%} > {_MAX_FAILURE_RATE:.0%} threshold). "
            f"Check translator '{translator_model}' configuration and language support."
        )

    # Write output
    with open(output_path, "w", encoding="utf-8") as fh:
        for row in translated_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("  ✓ Wrote %d rows to %s", total, output_path)
    return output_path
