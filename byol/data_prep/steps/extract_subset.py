# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Step: Extract a token-count subset from FineWeb-Edu parquet files.

Loads all parquet shards, filters by max token count, shuffles
deterministically, and extracts a subset with a target token budget.
Saves as both ``.parquet`` and ``.jsonl``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import DEFAULT_MAX_TOKEN_COUNT, DEFAULT_SEED

logger = logging.getLogger("byol-data-prep")


def _load_all_parquets(directory: str) -> pd.DataFrame:
    """Load and concatenate all parquet files in *directory*."""
    parquet_files = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(directory)
        for f in files
        if f.endswith(".parquet")
    )
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {directory}")
    logger.info(f"Loading {len(parquet_files)} parquet files from {directory} …")
    dfs = [pd.read_parquet(p) for p in parquet_files]
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"  Loaded {len(combined):,} total rows.")
    return combined


def _count_non_whitespace(text: str) -> int:
    return sum(1 for c in text if not c.isspace())


def extract_subset(
    *,
    parquet_dir: str,
    output_dir: str,
    target_tokens: int,
    max_token_count: int = DEFAULT_MAX_TOKEN_COUNT,
    seed: int = DEFAULT_SEED,
    output_basename: str | None = None,
    max_samples: int | None = None,
) -> str:
    """Extract a subset of FineWeb-Edu with a target token budget.

    Args:
        parquet_dir: Directory containing the source parquet shards.
        output_dir: Directory to write the subset files into.
        target_tokens: Desired total token count for the subset.
        max_token_count: Rows with ``token_count > max_token_count`` are
            filtered out before sampling.
        seed: Random seed for reproducible shuffling.
        output_basename: Optional basename for output files. If ``None``,
            a name is derived from *target_tokens*.

    Returns:
        Path to the output JSONL file.
    """
    np.random.seed(seed)

    df = _load_all_parquets(parquet_dir)

    # Filter by max token count
    filtered = df[df["token_count"] <= max_token_count].copy()
    logger.info(
        f"After filtering (token_count <= {max_token_count}): "
        f"{len(filtered):,} rows (dropped {len(df) - len(filtered):,})"
    )

    # Deterministic shuffle
    shuffled = filtered.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Accumulate rows until we hit the token budget
    total_tokens = 0
    sampled_indices: list[int] = []
    for idx, row in shuffled.iterrows():
        if total_tokens + row["token_count"] > target_tokens:
            break
        total_tokens += int(row["token_count"])
        sampled_indices.append(idx)  # type: ignore[arg-type]
        if max_samples is not None and len(sampled_indices) >= max_samples:
            break

    result = shuffled.iloc[sampled_indices].copy()

    if "char_count_no_whitespace" not in result.columns:
        result["char_count_no_whitespace"] = result["text"].apply(_count_non_whitespace)

    # Derive output filenames
    if output_basename is None:
        if target_tokens >= 1_000_000:
            suffix = f"{target_tokens // 1_000_000}M"
        elif target_tokens >= 1_000:
            suffix = f"{target_tokens // 1_000}K"
        else:
            suffix = str(target_tokens)
        output_basename = f"finewebedu-{suffix}T"

    os.makedirs(output_dir, exist_ok=True)
    parquet_out = os.path.join(output_dir, f"{output_basename}.parquet")
    jsonl_out = os.path.join(output_dir, f"{output_basename}.jsonl")

    result.to_parquet(parquet_out, index=False)
    result.to_json(jsonl_out, orient="records", lines=True, force_ascii=False)

    logger.info(
        f"Subset extracted: {len(result):,} rows, {total_tokens:,} tokens"
    )
    logger.info(f"  → {parquet_out}")
    logger.info(f"  → {jsonl_out}")

    return jsonl_out
