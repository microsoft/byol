# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Step: Download and combine SmolTalk2 subsets for SFT.

Downloads configured SmolTalk2/SmolTalk subsets from HuggingFace,
samples as specified, and writes a single combined JSONL file with
``messages`` and ``chat_template_kwargs`` fields ready for translation.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

from ..constants import DEFAULT_SEED, SMOLTALK2_DATASETS_CONFIG

logger = logging.getLogger("byol-data-prep")


def _extract_custom_instructions(chat_template_kwargs: str) -> str:
    """Extract custom_instructions from a chat_template_kwargs JSON string."""
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


def _download_single_subset(
    dataset_name: str,
    subset: str,
    split: Optional[str],
    default_source: str,
    default_chat_template_kwargs: str,
    sample_size: Optional[int],
    seed: int,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Download a single SmolTalk2/SmolTalk subset and return records."""
    from datasets import load_dataset

    effective_split = split or "train"
    logger.info(
        f"  Loading {dataset_name} subset={subset} split={effective_split} "
        f"(sample_size={'all' if sample_size is None else sample_size})"
    )

    try:
        ds = load_dataset(dataset_name, name=subset, split=effective_split)
    except Exception as e:
        logger.warning(f"  Failed to load {dataset_name}/{subset}/{effective_split}: {e}")
        return []

    records: List[Dict[str, Any]] = []
    for item in ds:
        record: Dict[str, Any] = {}
        # Extract messages
        messages = item.get("messages", [])
        if not messages:
            continue
        record["messages"] = messages

        # Extract source and chat_template_kwargs
        record["source"] = item.get("source", default_source)
        raw_kwargs = item.get("chat_template_kwargs", "")

        # Normalize: HF returns a dict for smoltalk2; convert to JSON string
        if isinstance(raw_kwargs, dict):
            if raw_kwargs:
                record["chat_template_kwargs"] = json.dumps(raw_kwargs, ensure_ascii=False)
            else:
                record["chat_template_kwargs"] = ""
        elif not raw_kwargs or (isinstance(raw_kwargs, str) and not raw_kwargs.strip()):
            # Build from default
            if default_chat_template_kwargs:
                record["chat_template_kwargs"] = json.dumps(
                    {"custom_instructions": default_chat_template_kwargs},
                    ensure_ascii=False,
                )
            else:
                record["chat_template_kwargs"] = ""
        else:
            record["chat_template_kwargs"] = raw_kwargs

        records.append(record)

    # Sample if needed
    if sample_size is not None and len(records) > sample_size:
        rng = random.Random(seed)
        records = rng.sample(records, sample_size)
        logger.info(f"    Sampled {sample_size} from {len(ds)} records")

    # Apply max_samples cap (for testing)
    if max_samples is not None and len(records) > max_samples:
        records = records[:max_samples]

    logger.info(f"    → {len(records)} records")
    return records


def download_smoltalk2(
    *,
    output_dir: str,
    output_jsonl: str,
    datasets_config: Optional[List[Dict[str, Any]]] = None,
    seed: int = DEFAULT_SEED,
    max_samples: Optional[int] = None,
) -> str:
    """Download and combine SmolTalk2 subsets into a single JSONL.

    Parameters
    ----------
    output_dir:
        Directory to write output files.
    output_jsonl:
        Path to the combined output JSONL file.
    datasets_config:
        List of dataset configurations. If ``None``, uses
        :data:`SMOLTALK2_DATASETS_CONFIG`.
    seed:
        Random seed for sampling reproducibility.
    max_samples:
        If set, cap per-subset samples (for quick testing).

    Returns
    -------
    str
        Path to the written JSONL file.
    """
    if datasets_config is None:
        datasets_config = list(SMOLTALK2_DATASETS_CONFIG)

    os.makedirs(output_dir, exist_ok=True)

    all_records: List[Dict[str, Any]] = []

    logger.info(f"Downloading {len(datasets_config)} SmolTalk2 subsets...")
    for cfg in datasets_config:
        records = _download_single_subset(
            dataset_name=cfg["dataset_name"],
            subset=cfg["subset"],
            split=cfg.get("split"),
            default_source=cfg["default_source"],
            default_chat_template_kwargs=cfg["default_chat_template_kwargs"],
            sample_size=cfg.get("sample_size"),
            seed=seed,
            max_samples=max_samples,
        )
        all_records.extend(records)

    # Assign stable IDs
    for idx, record in enumerate(all_records):
        record["id"] = str(idx)

    # Write combined JSONL
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        f"SmolTalk2 combined: {len(all_records):,} records → {output_jsonl}"
    )
    return output_jsonl
