# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""I/O utilities for JSONL files with concurrency-safe operations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Platform-specific file locking
try:
    from fcntl import LOCK_EX, LOCK_UN, flock
    _HAS_FLOCK = True
except ImportError:
    _HAS_FLOCK = False
    # Windows fallback
    try:
        import msvcrt
        _HAS_MSVCRT = True
    except ImportError:
        _HAS_MSVCRT = False

from .config import CORE_KEYS


def load_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return []
    
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def save_jsonl(
    data: list[dict[str, Any]], 
    file_path: str | Path,
    ensure_ascii: bool = False,
) -> None:
    """
    Save a list of dictionaries to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Output file path
        ensure_ascii: If True, escape non-ASCII characters
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")


def load_existing_results(output_file: str | Path) -> list[dict[str, Any]] | None:
    """
    Load existing results if file exists.
    
    Args:
        output_file: Path to the results file
        
    Returns:
        List of result dictionaries, or None if file doesn't exist
    """
    output_file = Path(output_file)
    if not output_file.exists():
        return None
    return load_jsonl(output_file)


def merge_result_records(
    base_results: list[dict[str, Any]], 
    new_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge translator-specific keys from new_results into base_results by 'id'.
    
    Keeps base order, fills in missing base rows from new_results when needed.
    Only non-CORE_KEYS fields are merged (i.e., translator results).
    
    Args:
        base_results: Existing results (from disk)
        new_results: New results to merge in
        
    Returns:
        Merged results list preserving original order
    """
    by_id: dict[int | str, dict[str, Any]] = {r["id"]: r for r in base_results}
    
    # Ensure base rows exist with core fields
    for r in new_results:
        if r["id"] not in by_id:
            by_id[r["id"]] = {k: r[k] for k in CORE_KEYS if k in r}
    
    # Merge translator keys (non-core keys)
    for r in new_results:
        base_record = by_id[r["id"]]
        for key, value in r.items():
            if key not in CORE_KEYS:
                base_record[key] = value
    
    # Preserve order
    if base_results:
        order = [r["id"] for r in base_results]
    else:
        order = [r["id"] for r in new_results]
    
    return [by_id[i] for i in order]


def _acquire_lock(lock_file) -> None:
    """Acquire a file lock (platform-specific)."""
    if _HAS_FLOCK:
        flock(lock_file, LOCK_EX)
    elif _HAS_MSVCRT:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)


def _release_lock(lock_file) -> None:
    """Release a file lock (platform-specific)."""
    if _HAS_FLOCK:
        flock(lock_file, LOCK_UN)
    elif _HAS_MSVCRT:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def save_results(
    results: list[dict[str, Any]], 
    output_file: str | Path,
) -> None:
    """
    Concurrency-safe merge-and-save operation.
    
    Uses file locking to prevent concurrent writes from corrupting data.
    Reads current file, merges with new results, writes atomically.
    
    Args:
        results: New results to save/merge
        output_file: Path to the output file
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    lock_path = output_file.with_suffix(output_file.suffix + ".lock")
    lock_file = None
    
    try:
        # Acquire lock if available
        if _HAS_FLOCK or _HAS_MSVCRT:
            lock_file = open(lock_path, "w")
            _acquire_lock(lock_file)
        
        # Load existing, merge, and save atomically
        on_disk = load_existing_results(output_file) or []
        merged = merge_result_records(on_disk, results)
        
        # Write to temp file first
        tmp_path = output_file.with_suffix(output_file.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for record in merged:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # Atomic replace
        os.replace(tmp_path, output_file)
        
    finally:
        if lock_file:
            try:
                _release_lock(lock_file)
                lock_file.close()
            except Exception:
                pass


def load_csv(file_path: str | Path) -> "pd.DataFrame":
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas DataFrame
    """
    import pandas as pd
    return pd.read_csv(file_path)


__all__ = [
    "load_jsonl",
    "save_jsonl",
    "load_existing_results",
    "merge_result_records",
    "save_results",
    "load_csv",
]
