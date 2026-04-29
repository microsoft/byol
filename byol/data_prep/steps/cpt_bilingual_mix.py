# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Step 6: Create a shuffled bilingual mix for CPT training.

Concatenates all three CPT data sources (refined target-language,
refined English, translated English→target-language), tags each row with
a ``dataset`` field, shuffles, and writes a single JSONL file plus a
``dataset_info.json`` ready for LlamaFactory.

Users can inject additional JSONL files via the ``extra_sources``
configuration (see :class:`~byol.data_prep.config.ExtraSource`).
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from ..constants import (
    DATASET_TAG_REFINED_ENG,
    DATASET_TAG_REFINED_TGT_LANG,
    DATASET_TAG_TRANSLATED,
    DEFAULT_SEED,
)

if TYPE_CHECKING:
    from ..config import ExtraSource

logger = logging.getLogger("byol-data-prep")


def _read_jsonl(path: str) -> list[dict]:
    """Read all JSON objects from a JSONL file."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tag_rows(rows: list[dict], dataset_tag: str) -> list[dict]:
    """Add / overwrite the ``dataset`` field on every row."""
    for row in rows:
        row["dataset"] = dataset_tag
    return rows


def create_bilingual_mix(
    *,
    tgt_lang_jsonl: str,
    eng_jsonl: str,
    translated_jsonl: str,
    output_jsonl: str,
    dataset_info_path: str,
    dataset_name: str,
    lang_code: str,
    seed: int = DEFAULT_SEED,
    max_samples: Optional[int] = None,
    overwrite: bool = False,
    extra_sources: Optional[List["ExtraSource"]] = None,
) -> str:
    """Build the bilingual mix JSONL and write ``dataset_info.json``.

    Parameters
    ----------
    tgt_lang_jsonl:
        Path to refined (or raw) target-language JSONL.
    eng_jsonl:
        Path to refined (or raw) English JSONL.
    translated_jsonl:
        Path to translated English→target-language JSONL.
    output_jsonl:
        Destination path for the combined JSONL.
    dataset_info_path:
        Destination path for ``dataset_info.json``.
    dataset_name:
        LlamaFactory dataset name (e.g. ``nya_english_cpt``).
    lang_code:
        ISO 639-3 language code used in dataset tags (e.g. ``nya``).
    seed:
        Random seed for shuffle reproducibility.
    max_samples:
        If set, cap each source to at most this many rows before mixing.
    overwrite:
        If False and *output_jsonl* exists, skip generation.
    extra_sources:
        Optional list of :class:`~byol.data_prep.config.ExtraSource`
        objects specifying additional JSONL files to fold into the mix.

    Returns
    -------
    str
        Path to the written output JSONL.
    """
    if not overwrite and os.path.exists(output_jsonl):
        logger.info(
            f"SKIPPED: Bilingual mix already exists at {output_jsonl}. "
            "Use --overwrite to regenerate."
        )
        return output_jsonl

    # ── Collect source files ─────────────────────────────────────────────
    all_rows: list[dict] = []

    tgt_tag = DATASET_TAG_REFINED_TGT_LANG.format(lang=lang_code)

    for label, path, tag in [
        ("target-language", tgt_lang_jsonl, tgt_tag),
        ("English", eng_jsonl, DATASET_TAG_REFINED_ENG),
        ("translated", translated_jsonl, DATASET_TAG_TRANSLATED),
    ]:
        if not os.path.exists(path):
            logger.warning(f"Bilingual mix: {label} file not found at {path} — skipping.")
            continue
        rows = _read_jsonl(path)
        if max_samples is not None:
            rows = rows[:max_samples]
        _tag_rows(rows, tag)
        logger.info(f"  {label}: {len(rows):,} rows from {path}")
        all_rows.extend(rows)

    # ── Extra user-supplied sources ──────────────────────────────────────
    for src in extra_sources or []:
        if not os.path.exists(src.path):
            logger.warning(
                f"Bilingual mix: extra source '{src.dataset_tag}' "
                f"not found at {src.path} — skipping."
            )
            continue
        rows = _read_jsonl(src.path)
        if max_samples is not None:
            rows = rows[:max_samples]
        _tag_rows(rows, src.dataset_tag)
        logger.info(
            f"  extra ({src.dataset_tag}): {len(rows):,} rows from {src.path}"
        )
        all_rows.extend(rows)

    if not all_rows:
        raise FileNotFoundError(
            "Cannot create bilingual mix: no source files were found. "
            "Run earlier pipeline steps first."
        )

    # ── Shuffle ──────────────────────────────────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(all_rows)
    logger.info(f"  Total bilingual mix: {len(all_rows):,} rows (shuffled, seed={seed})")

    # ── Write JSONL ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"  Wrote {output_jsonl}")

    # ── Write dataset_info.json ──────────────────────────────────────────
    # File paths inside dataset_info.json are relative to the dir
    # containing dataset_info.json (i.e. bilingual_mix/).
    relative_jsonl = os.path.relpath(output_jsonl, os.path.dirname(dataset_info_path))

    dataset_info = {
        dataset_name: {
            "file_name": relative_jsonl,
            "columns": {"prompt": "text"},
        }
    }
    os.makedirs(os.path.dirname(dataset_info_path), exist_ok=True)
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info(f"  Wrote {dataset_info_path}")

    return output_jsonl
