# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Step: Download FineWeb-2 data for a target language.

Downloads the specified language subset from ``HuggingFaceFW/fineweb-2``
and saves each split as a JSONL file.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("byol-data-prep")


def download_tgt_lang_fineweb2(
    *,
    lang_code: str,
    output_dir: str,
    dataset_name: str | None = None,
    splits: tuple[str, ...] | list[str] = ("train",),
    max_samples: int | None = None,
) -> str:
    """Download FineWeb-2 data for a target language.

    Args:
        lang_code: ISO 639-3 language code (e.g. ``"nya"``).
        output_dir: Directory to write the JSONL files into.
        dataset_name: HuggingFace subset name. Defaults to ``"{lang_code}_Latn"``.
        splits: Which splits to download (default: ``("train",)``).
        max_samples: If set, write at most this many records per split.

    Returns:
        Path to the output directory containing the downloaded JSONL files.
    """
    from datasets import load_dataset
    from ..config import _guess_fineweb2_subset

    subset = dataset_name or _guess_fineweb2_subset(lang_code)
    logger.info(f"Downloading FineWeb-2 subset '{subset}' …")

    ds = load_dataset("HuggingFaceFW/fineweb-2", name=subset)

    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        if split not in ds:
            logger.warning(f"Split '{split}' not found in dataset — skipping.")
            continue
        output_path = os.path.join(output_dir, f"{lang_code}_{split}.jsonl")
        logger.info(f"Writing {split} split to {output_path} …")
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for item in ds[split]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                if max_samples is not None and count >= max_samples:
                    break
        logger.info(f"  → {count:,} records written.")
        if max_samples is not None and count >= max_samples:
            logger.info(f"  (capped at --max-samples {max_samples})")

    logger.info(f"FineWeb-2 download complete → {output_dir}")
    return output_dir
