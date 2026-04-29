# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Step: Download FineWeb-Edu English data and extract a token-matched subset.

Downloads parquet shards from the ``HuggingFaceFW/fineweb-edu`` 10BT
sample, then extracts a subset whose total token count matches the
target-language corpus (counted with tiktoken ``o200k_base``).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from ..constants import (
    DEFAULT_MAX_TOKEN_COUNT,
    DEFAULT_SEED,
    DEFAULT_TIKTOKEN_ENCODING,
    FINEWEBEDU_NUM_SHARDS,
    FINEWEBEDU_SHARD_TEMPLATE,
)
from .common import count_tokens_jsonl

logger = logging.getLogger("byol-data-prep")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _download_shards(
    output_dir: str,
    num_shards: int,
    max_samples: int | None,
) -> str:
    """Download FineWeb-Edu parquet shards from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    os.makedirs(output_dir, exist_ok=True)
    actual_shards = 1 if max_samples is not None else num_shards
    if max_samples is not None:
        logger.info(
            f"--max-samples active: downloading only 1 shard instead of {num_shards}"
        )
    logger.info(f"Downloading {actual_shards} FineWeb-Edu shards → {output_dir}")

    for i in range(actual_shards):
        filename = FINEWEBEDU_SHARD_TEMPLATE.format(index=i)
        path = hf_hub_download(
            repo_id="HuggingFaceFW/fineweb-edu",
            filename=filename,
            repo_type="dataset",
            local_dir=output_dir,
        )
        logger.info(f"  Downloaded shard {i + 1}/{actual_shards}: {path}")

    logger.info("FineWeb-Edu download complete.")
    return output_dir


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


def _extract_subset(
    parquet_dir: str,
    target_tokens: int,
    output_jsonl: str,
    max_token_count: int = DEFAULT_MAX_TOKEN_COUNT,
    seed: int = DEFAULT_SEED,
    max_samples: int | None = None,
) -> str:
    """Extract a subset from parquet files matching a token budget.

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

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    result.to_json(output_jsonl, orient="records", lines=True, force_ascii=False)

    logger.info(
        f"Subset extracted: {len(result):,} rows, {total_tokens:,} tokens → {output_jsonl}"
    )
    return output_jsonl


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────


def download_and_extract_finewebedu(
    *,
    parquet_dir: str,
    output_jsonl: str,
    tgt_lang_jsonl: str,
    num_shards: int = FINEWEBEDU_NUM_SHARDS,
    target_tokens_override: int | None = None,
    max_token_count: int = DEFAULT_MAX_TOKEN_COUNT,
    seed: int = DEFAULT_SEED,
    encoding_name: str = DEFAULT_TIKTOKEN_ENCODING,
    max_samples: int | None = None,
) -> str:
    """Download FineWeb-Edu shards and extract a token-matched English subset.

    The target token budget is computed automatically by counting tiktoken
    tokens in *tgt_lang_jsonl*.  Pass *target_tokens_override* to use a
    fixed budget instead.

    Args:
        parquet_dir: Directory to download shards into.
        output_jsonl: Path to write the extracted subset JSONL.
        tgt_lang_jsonl: Path to the target-language JSONL file (for
            automatic token counting).
        num_shards: Number of parquet shards to download.
        target_tokens_override: Fixed token budget (skips auto-count).
        max_token_count: Drop rows with more tokens than this.
        seed: Random seed for reproducible shuffling.
        encoding_name: tiktoken encoding for token counting.
        max_samples: If set, download 1 shard and cap subset rows.

    Returns:
        Path to the output JSONL file.
    """
    # 1. Download parquet shards
    _download_shards(parquet_dir, num_shards, max_samples)

    # 2. Determine target token budget
    if target_tokens_override is not None and target_tokens_override > 0:
        target_tokens = target_tokens_override
        logger.info(f"Using override target_tokens = {target_tokens:,}")
    else:
        logger.info(
            f"Counting tiktoken ({encoding_name}) tokens in {tgt_lang_jsonl} …"
        )
        target_tokens = count_tokens_jsonl(tgt_lang_jsonl, encoding_name)
        logger.info(f"Target-language token count: {target_tokens:,} → using as English budget")

    # 3. Extract subset
    return _extract_subset(
        parquet_dir=parquet_dir,
        target_tokens=target_tokens,
        output_jsonl=output_jsonl,
        max_token_count=max_token_count,
        seed=seed,
        max_samples=max_samples,
    )
