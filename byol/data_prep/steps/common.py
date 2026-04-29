# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shared utilities for data-preparation pipeline steps.

Provides JSONL I/O, robust JSON parsing, batch chunking, checkpoint/resume
support, and a helper to create an LLM client via
:class:`~byol.translation_backends.api.azure_openai.AzureOpenAIGPT5Translator`.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger("byol-data-prep")


# ──────────────────────────────────────────────────────────────────────────────
# JSONL I/O
# ──────────────────────────────────────────────────────────────────────────────


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def count_tokens_jsonl(path: str, encoding_name: str = "o200k_base") -> int:
    """Count total tiktoken tokens across all ``text`` fields in a JSONL file.

    Args:
        path: Path to the JSONL file.
        encoding_name: tiktoken encoding name (default ``"o200k_base"``).

    Returns:
        Total token count.
    """
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if text:
                    total += len(enc.encode(text))
            except Exception:
                continue
    return total


def save_jsonl(records: Iterable[Dict[str, Any]], path: str) -> None:
    """Append *records* to a JSONL file (creates dirs as needed)."""
    recs = list(records)
    if not recs:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_removed_records(removed_path: str, records: Iterable[Dict[str, Any]]) -> None:
    """Append removed/dropped records to a sidecar JSONL."""
    save_jsonl(records, removed_path)


# ──────────────────────────────────────────────────────────────────────────────
# ID helpers
# ──────────────────────────────────────────────────────────────────────────────


def get_record_id(item: Dict[str, Any], idx: int) -> str:
    """Return existing ``id`` or fall back to positional index."""
    existing = item.get("id")
    return str(existing if existing is not None else idx)


def ensure_stable_ids(data: List[Dict[str, Any]]) -> None:
    """Assign an ``id`` field to every record that lacks one (in-place)."""
    for idx, item in enumerate(data):
        item.setdefault("id", get_record_id(item, idx))


def read_processed_ids(output_path: str) -> Set[str]:
    """Scan an existing output file to determine which IDs are done."""
    processed: Set[str] = set()
    if not os.path.exists(output_path):
        return processed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = obj.get("id") or obj.get("_doc_id") or obj.get("source_index")
                if rid is not None:
                    processed.add(str(rid))
            except Exception:
                continue
    return processed


def read_removed_ids(path: str) -> Set[str]:
    """Read IDs that were previously removed/dropped."""
    ids: Set[str] = set()
    if not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = obj.get("id")
                if rid is not None:
                    ids.add(str(rid))
            except Exception:
                ids.add(line)
    return ids


# ──────────────────────────────────────────────────────────────────────────────
# Batching
# ──────────────────────────────────────────────────────────────────────────────


def chunked(seq: List[Any], size: int) -> List[List[Any]]:
    """Split *seq* into sub-lists of at most *size* elements."""
    return [seq[i : i + size] for i in range(0, len(seq), size)]


# ──────────────────────────────────────────────────────────────────────────────
# JSON payload helpers
# ──────────────────────────────────────────────────────────────────────────────


def build_user_payload(batch: List[Dict[str, Any]]) -> str:
    """Serialise a batch of records into the user-message JSON payload."""
    payload = {
        "items": [
            {"id": str(item["id"]), "text": item.get("text", "")} for item in batch
        ]
    }
    return json.dumps(payload, ensure_ascii=False)


def strip_code_fences(s: str) -> str:
    """Remove markdown code fences (```json ... ```) from a string."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def remove_invalid_control_chars(s: str) -> str:
    """Strip control characters that break JSON parsing."""
    return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", s)


def extract_outer_json_object(s: str) -> Optional[str]:
    """Extract the outermost ``{…}`` substring."""
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return None


def parse_model_json(raw: str) -> Optional[Dict[str, Any]]:
    """Robustly parse LLM JSON output, handling code fences & control chars."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    s = strip_code_fences(raw)
    s = remove_invalid_control_chars(s)
    candidate = extract_outer_json_object(s) or s
    try:
        return json.loads(candidate)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Output path helpers
# ──────────────────────────────────────────────────────────────────────────────


def derive_output_file(input_path: str, suffix: str) -> str:
    """Derive an output path from *input_path* by appending *suffix*."""
    d, base = os.path.split(input_path)
    name, ext = os.path.splitext(base)
    if ext == "":
        ext = ".jsonl"
    return os.path.join(d, f"{name}_{suffix}{ext}")


# ──────────────────────────────────────────────────────────────────────────────
# LLM client helper
# ──────────────────────────────────────────────────────────────────────────────


def create_llm_translator(
    *,
    model_name: str,
    system_prompt: str,
    reasoning_effort: str = "low",
    max_tokens: int = 1024,
    api_version: str = "2024-08-01-preview",
):
    """Create an :class:`AzureOpenAIGPT5Translator` configured for data-prep.

    The caller can then use ``translator.client`` for batched JSON-mode calls
    while reusing the translator's Azure Entra authentication setup.

    Returns:
        An ``AzureOpenAIGPT5Translator`` instance.
    """
    from byol.translation_backends.api.azure_openai import AzureOpenAIGPT5Translator

    translator = AzureOpenAIGPT5Translator(
        src_lang="auto",
        tgt_lang="auto",
        model_name=model_name,
        system_prompt=system_prompt,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        api_version=api_version,
    )
    return translator


def llm_json_call(
    translator,
    user_payload: str,
    max_completion_tokens: int,
) -> str:
    """Make a single JSON-mode LLM call using the translator's client.

    Args:
        translator: An ``AzureOpenAIGPT5Translator`` instance.
        user_payload: The serialised user-message string.
        max_completion_tokens: Token budget for the completion.

    Returns:
        Raw response content string.
    """
    response = translator.client.chat.completions.create(
        model=translator.model_name,
        messages=[
            {"role": "system", "content": translator.system_prompt},
            {"role": "user", "content": user_payload},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=max_completion_tokens,
        reasoning_effort=translator.reasoning_effort,
    )
    return (response.choices[0].message.content or "").strip()
