# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Translator language support validation.

Provides functions to check whether a translator supports a given language
and to suggest alternative translators when the chosen one doesn't.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from byol.common.exceptions import LanguageNotSupportedError

# CSV path relative to repo root
_CSV_PATH = Path(__file__).parent.parent.parent / "data" / "translator_language_support.csv"

# Cached support matrix: list of row dicts from the CSV
_support_matrix: list[dict[str, str]] | None = None

# Map translator model names → CSV column names
_TRANSLATOR_COLUMN_MAP: dict[str, str] = {
    "microsoft-translator": "microsoft_translator",
    "google-translator": "google_translator",
    "nllb-200-600m": "nllb",
    "nllb-200-1.3b": "nllb",
    "nllb-200-3.3b": "nllb",
    "seamless-m4t-medium": "seamless_m4t",
    "seamless-m4t-large": "seamless_m4t",
    "madlad-400-3b": "madlad_400",
    "madlad-400-7b": "madlad_400",
    "translategemma": "translategemma",
}

# Patterns for LLM-based translators that are always considered supported
_LLM_ALWAYS_SUPPORTED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^gpt-"),
    re.compile(r"^deepseek-"),
    re.compile(r"^gemma-3-"),
    re.compile(r"^qwen3-"),
    re.compile(r"^apertus-"),
    re.compile(r"^aya-"),
]

# Explicit LLM translator → CSV column (for those that have a column)
_LLM_PREFIX_COLUMN_MAP: list[tuple[str, str]] = [
    ("gpt-", "gpt"),
    ("deepseek-", "deepseek"),
]

# Human-readable group names for CSV columns
_COLUMN_GROUP_NAMES: dict[str, str] = {
    "microsoft_translator": "Microsoft Translator",
    "google_translator": "Google Translator",
    "nllb": "NLLB",
    "seamless_m4t": "SeamlessM4T",
    "madlad_400": "MADLAD-400",
    "translategemma": "TranslateGemma",
    "gpt": "GPT",
    "deepseek": "DeepSeek",
}

# Translator columns in the CSV (all except language/code columns)
_TRANSLATOR_COLUMNS = [
    "microsoft_translator",
    "google_translator",
    "nllb",
    "seamless_m4t",
    "madlad_400",
    "translategemma",
    "gpt",
    "deepseek",
]


def _load_matrix() -> list[dict[str, str]]:
    """Load and cache the translator language support CSV."""
    global _support_matrix
    if _support_matrix is not None:
        return _support_matrix

    if not _CSV_PATH.exists():
        _support_matrix = []
        return _support_matrix

    with open(_CSV_PATH, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        _support_matrix = list(reader)
    return _support_matrix


def _find_language_row(lang: str) -> dict[str, str] | None:
    """Find a row in the support matrix matching the given language identifier.

    Searches across language name, iso2, iso3, flores200, and bcp47 columns
    case-insensitively.
    """
    matrix = _load_matrix()
    lang_lower = lang.lower().strip()
    for row in matrix:
        if any(
            row.get(col, "").lower().strip() == lang_lower
            for col in ("language", "iso2", "iso3", "flores200", "bcp47")
        ):
            return row
    return None


def _resolve_column(translator: str) -> str | None:
    """Resolve a translator model name to its CSV column name.

    Returns None if the translator is an LLM that's always supported
    but has no dedicated CSV column.
    """
    # Exact match first
    col = _TRANSLATOR_COLUMN_MAP.get(translator)
    if col is not None:
        return col

    # LLM prefix → column
    translator_lower = translator.lower()
    for prefix, column in _LLM_PREFIX_COLUMN_MAP:
        if translator_lower.startswith(prefix):
            return column

    # LLM patterns that are always supported (no dedicated column)
    for pattern in _LLM_ALWAYS_SUPPORTED_PATTERNS:
        if pattern.match(translator_lower):
            return None  # always supported, no column to check

    return None


def _is_always_supported_llm(translator: str) -> bool:
    """Check if a translator is an LLM that's always considered supported."""
    translator_lower = translator.lower()
    return any(p.match(translator_lower) for p in _LLM_ALWAYS_SUPPORTED_PATTERNS)


def is_language_supported(translator: str, lang: str) -> bool:
    """Check if a translator supports a given language.

    Args:
        translator: Translator name (e.g., "microsoft-translator", "nllb-200-3.3b")
        lang: Language in any format (ISO-2, ISO-3, Flores-200, full name)

    Returns:
        True if supported, False if not, True if unknown (permissive for
        unlisted languages/translators).
    """
    # LLMs that are always supported
    if _is_always_supported_llm(translator):
        return True

    column = _resolve_column(translator)
    if column is None:
        # Unknown translator — be permissive
        return True

    row = _find_language_row(lang)
    if row is None:
        # Language not in matrix — be permissive
        return True

    return row.get(column, "0").strip() == "1"


def get_supported_translators(lang: str) -> list[str]:
    """Get all translators that support a given language.

    Args:
        lang: Language in any format

    Returns:
        List of translator group names that support this language.
    """
    row = _find_language_row(lang)
    if row is None:
        # Language not in matrix — return all as potentially supported
        return list(_COLUMN_GROUP_NAMES.values())

    supported = []
    for col in _TRANSLATOR_COLUMNS:
        if row.get(col, "0").strip() == "1":
            supported.append(_COLUMN_GROUP_NAMES.get(col, col))
    return supported


def validate_translator_for_language(translator: str, lang: str) -> None:
    """Validate that a translator supports a language, raising an error if not.

    Args:
        translator: Translator name
        lang: Language in any format

    Raises:
        LanguageNotSupportedError: If the translator doesn't support the language,
            with a message suggesting alternative translators.
    """
    if is_language_supported(translator, lang):
        return

    alternatives = get_supported_translators(lang)
    codes = resolve_language_codes(lang)
    display_lang = codes["language"] if codes else lang

    alt_text = ", ".join(alternatives) if alternatives else "none known"
    raise LanguageNotSupportedError(
        language=display_lang,
        translator=translator,
        supported=alternatives,
        suggestion=f"Try one of: {alt_text}",
    )


def resolve_language_codes(lang: str) -> dict[str, str] | None:
    """Look up all code formats for a language.

    Args:
        lang: Language in any format (iso2, iso3, flores200, full name)

    Returns:
        Dict with keys: language, iso2, iso3, flores200, bcp47.
        None if language not found in the support matrix.
    """
    row = _find_language_row(lang)
    if row is None:
        return None

    return {
        "language": row["language"],
        "iso2": row["iso2"],
        "iso3": row["iso3"],
        "flores200": row["flores200"],
        "bcp47": row["bcp47"],
    }
