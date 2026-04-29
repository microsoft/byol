# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Step: Download and filter the AYA dataset for SFT.

Downloads the CohereForAI/aya_dataset from HuggingFace, filters for
specified source languages plus the target language, and writes
separate train/test JSONL files.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ..constants import AYA_DATASET_REPO_ID, AYA_SOURCE_LANGUAGE_CODES

logger = logging.getLogger("byol-data-prep")


def download_aya(
    *,
    tgt_lang_code: str,
    output_dir: str,
    train_jsonl: str,
    test_jsonl: str,
    source_language_codes: Optional[List[str]] = None,
    cache_dir: str = "",
    max_samples: Optional[int] = None,
) -> Tuple[str, str]:
    """Download and filter the AYA dataset for the target language.

    Parameters
    ----------
    tgt_lang_code:
        Target language ISO 639-3 code (e.g. ``"nya"``).
    output_dir:
        Directory for output files.
    train_jsonl:
        Path to write the filtered train JSONL.
    test_jsonl:
        Path to write the filtered test JSONL.
    source_language_codes:
        List of source language codes to include for translation.
        Defaults to :data:`AYA_SOURCE_LANGUAGE_CODES`.
    cache_dir:
        HuggingFace cache directory for the dataset download.
    max_samples:
        If set, limit to first N samples per split (for testing).

    Returns
    -------
    Tuple[str, str]
        Paths to (train_jsonl, test_jsonl).
    """
    from datasets import load_dataset

    if source_language_codes is None:
        source_language_codes = list(AYA_SOURCE_LANGUAGE_CODES)

    all_language_codes = source_language_codes + [tgt_lang_code]

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Downloading AYA dataset from HuggingFace ({AYA_DATASET_REPO_ID})...")
    dataset = load_dataset(AYA_DATASET_REPO_ID, cache_dir=cache_dir or None)

    for split_name, output_file in [("train", train_jsonl), ("test", test_jsonl)]:
        if split_name not in dataset:
            logger.warning(f"Split '{split_name}' not found in AYA dataset, skipping...")
            continue

        split_data = dataset[split_name]
        logger.info(f"AYA {split_name}: {len(split_data)} total entries")

        if max_samples is not None:
            split_data = split_data.select(range(min(max_samples, len(split_data))))
            logger.info(f"  Limited to first {len(split_data)} samples")

        all_entries: List[Dict[str, Any]] = []
        source_count = 0
        target_count = 0

        for entry in tqdm(split_data, desc=f"Filtering AYA {split_name}"):
            lang_code = entry.get("language_code", "")
            if lang_code not in all_language_codes:
                continue

            entry_dict = dict(entry)
            entry_dict["split"] = split_name

            if lang_code in source_language_codes:
                source_count += 1
            elif lang_code == tgt_lang_code:
                target_count += 1

            all_entries.append(entry_dict)

        logger.info(f"  Source language entries (need translation): {source_count}")
        logger.info(f"  Target language entries (native, direct copy): {target_count}")
        logger.info(f"  Total filtered entries: {len(all_entries)}")

        # Assign unique IDs
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for idx, entry in enumerate(all_entries):
                entry["translation_id"] = f"{split_name}_{entry.get('language_code', 'unk')}_{idx}"
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"  Wrote {len(all_entries):,} entries → {output_file}")

    return train_jsonl, test_jsonl
