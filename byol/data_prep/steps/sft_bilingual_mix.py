# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Step: Create the final SFT bilingual mix for training.

Combines all SFT data sources into a single shuffled JSONL:
  - English SmolTalk2 (original)
  - Translated SmolTalk2 (target language)
  - Translated AYA dataset (train split, converted to messages format)
  - Native AYA entries (target language, converted to messages format)

Also writes:
  - A test JSONL from AYA test split
  - ``dataset_info.json`` for LlamaFactory (sharegpt format for train,
    prompt/response for test)

Users can inject additional JSONL files via the ``extra_sources``
configuration (see :class:`~byol.data_prep.config.ExtraSource`).
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..constants import (
    DATASET_TAG_SFT_AYA_NATIVE_TRAIN,
    DATASET_TAG_SFT_AYA_TRANSLATED_TRAIN,
    DATASET_TAG_SFT_SMOLTALK2_ENG,
    DATASET_TAG_SFT_SMOLTALK2_TRANSLATED,
    DEFAULT_SEED,
)

if TYPE_CHECKING:
    from ..config import ExtraSource

logger = logging.getLogger("byol-data-prep")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read all JSON objects from a JSONL file."""
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tag_rows(rows: List[Dict[str, Any]], dataset_tag: str) -> List[Dict[str, Any]]:
    """Add / overwrite the ``dataset`` field on every row."""
    for row in rows:
        row["dataset"] = dataset_tag
    return rows


def _aya_to_messages(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an AYA record (inputs/targets) to messages format.

    Returns a new dict with ``messages`` field (and ``dataset`` tag preserved).
    """
    messages = [
        {"role": "user", "content": record.get("inputs", "")},
        {"role": "assistant", "content": record.get("targets", "")},
    ]
    return {
        "dataset": record.get("dataset", ""),
        "messages": messages,
    }


def _strip_metadata_for_training(record: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only ``dataset`` and ``messages`` fields for the training JSONL.

    Removes internal-only fields like ``id``, ``source``, ``chat_template_kwargs``,
    ``translation_id``, ``language_code``, etc.
    """
    return {
        "dataset": record.get("dataset", ""),
        "messages": record.get("messages", []),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────


def create_sft_bilingual_mix(
    *,
    smoltalk2_eng_jsonl: str,
    smoltalk2_translated_jsonl: str,
    aya_translated_train_jsonl: str,
    aya_translated_test_jsonl: str,
    output_jsonl: str,
    test_output_jsonl: str,
    dataset_info_path: str,
    dataset_name: str,
    test_dataset_name: str,
    lang_code: str,
    tgt_lang_code: str,
    seed: int = DEFAULT_SEED,
    max_samples: Optional[int] = None,
    overwrite: bool = False,
    extra_sources: Optional[List["ExtraSource"]] = None,
) -> str:
    """Build the SFT bilingual mix JSONL and write ``dataset_info.json``.

    Parameters
    ----------
    smoltalk2_eng_jsonl:
        Path to the combined English SmolTalk2 JSONL.
    smoltalk2_translated_jsonl:
        Path to the translated SmolTalk2 JSONL.
    aya_translated_train_jsonl:
        Path to the translated AYA train JSONL.
    aya_translated_test_jsonl:
        Path to the translated AYA test JSONL.
    output_jsonl:
        Destination path for the combined SFT training JSONL.
    test_output_jsonl:
        Destination path for the SFT test JSONL.
    dataset_info_path:
        Destination path for ``dataset_info.json``.
    dataset_name:
        LlamaFactory dataset name for the SFT training set
        (e.g. ``nya_english_sft``).
    test_dataset_name:
        LlamaFactory dataset name for the SFT test set
        (e.g. ``nya_sft_test``).
    lang_code:
        ISO 639-3 language code used in dataset tags (e.g. ``nya``).
    tgt_lang_code:
        Target language ISO 639-3 code (e.g. ``nya``).
    seed:
        Random seed for shuffle reproducibility.
    max_samples:
        If set, cap each source to at most this many rows before mixing.
    overwrite:
        If False and *output_jsonl* exists, skip generation.
    extra_sources:
        Optional list of :class:`~byol.data_prep.config.ExtraSource`
        objects specifying additional JSONL files to fold into the mix.
        Supported formats: ``"sharegpt"`` (rows with ``messages``) and
        ``"aya"`` (rows with ``inputs`` / ``targets``).

    Returns
    -------
    str
        Path to the written output JSONL.
    """
    if not overwrite and os.path.exists(output_jsonl):
        logger.info(
            f"SKIPPED: SFT bilingual mix already exists at {output_jsonl}. "
            "Use --overwrite to regenerate."
        )
        return output_jsonl

    all_rows: List[Dict[str, Any]] = []

    # ── 1. English SmolTalk2 ─────────────────────────────────────────────
    smoltalk2_eng_tag = DATASET_TAG_SFT_SMOLTALK2_ENG
    if os.path.exists(smoltalk2_eng_jsonl):
        rows = _read_jsonl(smoltalk2_eng_jsonl)
        if max_samples is not None:
            rows = rows[:max_samples]
        _tag_rows(rows, smoltalk2_eng_tag)
        # Strip to messages-only format
        rows = [_strip_metadata_for_training(r) for r in rows]
        logger.info(f"  SmolTalk2 English: {len(rows):,} rows")
        all_rows.extend(rows)
    else:
        logger.warning(f"  SmolTalk2 English not found at {smoltalk2_eng_jsonl}")

    # ── 2. Translated SmolTalk2 ──────────────────────────────────────────
    smoltalk2_trans_tag = DATASET_TAG_SFT_SMOLTALK2_TRANSLATED.format(lang=lang_code)
    if os.path.exists(smoltalk2_translated_jsonl):
        rows = _read_jsonl(smoltalk2_translated_jsonl)
        if max_samples is not None:
            rows = rows[:max_samples]
        _tag_rows(rows, smoltalk2_trans_tag)
        rows = [_strip_metadata_for_training(r) for r in rows]
        logger.info(f"  SmolTalk2 translated: {len(rows):,} rows")
        all_rows.extend(rows)
    else:
        logger.warning(f"  SmolTalk2 translated not found at {smoltalk2_translated_jsonl}")

    # ── 3. Translated AYA train ──────────────────────────────────────────
    aya_trans_tag = DATASET_TAG_SFT_AYA_TRANSLATED_TRAIN.format(lang=lang_code)
    aya_native_tag = DATASET_TAG_SFT_AYA_NATIVE_TRAIN.format(lang=lang_code)
    if os.path.exists(aya_translated_train_jsonl):
        rows = _read_jsonl(aya_translated_train_jsonl)
        if max_samples is not None:
            rows = rows[:max_samples]

        # Separate native vs translated entries
        native_rows = [r for r in rows if r.get("language_code") == tgt_lang_code]
        translated_rows = [r for r in rows if r.get("language_code") != tgt_lang_code]

        # Tag appropriately
        _tag_rows(native_rows, aya_native_tag)
        _tag_rows(translated_rows, aya_trans_tag)

        # Convert to messages format
        native_msg = [_aya_to_messages(r) for r in native_rows]
        translated_msg = [_aya_to_messages(r) for r in translated_rows]

        logger.info(
            f"  AYA train: {len(native_msg):,} native + "
            f"{len(translated_msg):,} translated = {len(native_msg) + len(translated_msg):,} rows"
        )
        all_rows.extend(native_msg)
        all_rows.extend(translated_msg)
    else:
        logger.warning(f"  AYA translated train not found at {aya_translated_train_jsonl}")

    # ── Extra user-supplied sources ──────────────────────────────────────
    for src in extra_sources or []:
        if not os.path.exists(src.path):
            logger.warning(
                f"SFT bilingual mix: extra source '{src.dataset_tag}' "
                f"not found at {src.path} — skipping."
            )
            continue
        rows = _read_jsonl(src.path)
        if max_samples is not None:
            rows = rows[:max_samples]
        _tag_rows(rows, src.dataset_tag)

        if src.format == "aya":
            # Convert inputs/targets → messages format
            rows = [_aya_to_messages(r) for r in rows]
        elif src.format == "sharegpt":
            # Already has messages — strip to messages-only
            rows = [_strip_metadata_for_training(r) for r in rows]
        else:
            # Treat "text" or unknown as raw text — create a single-turn message
            rows = [
                {
                    "dataset": r.get("dataset", src.dataset_tag),
                    "messages": [{"role": "user", "content": r.get("text", "")}],
                }
                for r in rows
            ]

        logger.info(
            f"  extra ({src.dataset_tag}, fmt={src.format}): "
            f"{len(rows):,} rows from {src.path}"
        )
        all_rows.extend(rows)

    if not all_rows:
        raise FileNotFoundError(
            "Cannot create SFT bilingual mix: no source files were found. "
            "Run earlier pipeline steps first."
        )

    # ── Shuffle ──────────────────────────────────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(all_rows)
    logger.info(f"  Total SFT training mix: {len(all_rows):,} rows (shuffled, seed={seed})")

    # ── Write training JSONL ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"  Wrote {output_jsonl}")

    # ── 4. AYA test set ──────────────────────────────────────────────────
    if os.path.exists(aya_translated_test_jsonl):
        test_rows = _read_jsonl(aya_translated_test_jsonl)
        if max_samples is not None:
            test_rows = test_rows[:max_samples]
        # Tag test rows
        test_tag = f"aya_dataset_translated_to_{lang_code}_test"
        _tag_rows(test_rows, test_tag)
        # Keep AYA format (inputs/targets) for test — matches existing dataset_info
        os.makedirs(os.path.dirname(test_output_jsonl), exist_ok=True)
        with open(test_output_jsonl, "w", encoding="utf-8") as f:
            for row in test_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"  Wrote SFT test: {len(test_rows):,} rows → {test_output_jsonl}")
    else:
        logger.warning(f"  AYA translated test not found at {aya_translated_test_jsonl}")

    # ── Write dataset_info.json ──────────────────────────────────────────
    _write_dataset_info(
        dataset_info_path=dataset_info_path,
        output_jsonl=output_jsonl,
        test_output_jsonl=test_output_jsonl,
        dataset_name=dataset_name,
        test_dataset_name=test_dataset_name,
    )

    return output_jsonl


def _write_dataset_info(
    *,
    dataset_info_path: str,
    output_jsonl: str,
    test_output_jsonl: str,
    dataset_name: str,
    test_dataset_name: str,
) -> None:
    """Write or update ``dataset_info.json`` with SFT dataset entries."""
    info_dir = os.path.dirname(dataset_info_path)

    # Relative paths from the dataset_info.json directory
    rel_train = os.path.relpath(output_jsonl, info_dir)
    rel_test = os.path.relpath(test_output_jsonl, info_dir)

    # Load existing dataset_info if present (may contain CPT entries)
    existing: Dict[str, Any] = {}
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}

    # Training dataset — sharegpt format (messages with role/content)
    existing[dataset_name] = {
        "file_name": rel_train,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }

    # Test dataset — AYA format (prompt/response)
    if os.path.exists(test_output_jsonl):
        existing[test_dataset_name] = {
            "file_name": rel_test,
            "columns": {
                "prompt": "inputs",
                "response": "targets",
            },
        }

    os.makedirs(info_dir, exist_ok=True)
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info(f"  Wrote {dataset_info_path}")
