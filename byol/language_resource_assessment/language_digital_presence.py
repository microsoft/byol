# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Language Digital Presence & Classification

This module provides:
1. Language classification based on digital resource availability
2. Language metadata lookup (speakers, corpus size, family)

Classification tiers (based on corpus size in words):
- Extreme-Low-Resource (≤ 5×10⁶ words): Minimal digital presence, MT-based access only
- Low-Resource (5×10⁶ – 2×10⁹ words): Limited but usable for continual pretraining
- Mid-Resource (2×10⁹ – 10¹¹ words): Substantial resources, light adaptation needed
- High-Resource (> 10¹¹ words): Comprehensive LLM support

Usage:
    python -m byol.language_resource_assessment \\
        --task language-classification \\
        --tgt-lang nya

    python -m byol.language_resource_assessment \\
        --task language-classification \\
        --tgt-lang Chichewa
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# =============================================================================
# CONSTANTS
# =============================================================================

# Classification labels mapped to cluster IDs
CLASSIFICATION_LABELS: dict[int, str] = {
    0: "Extreme-Low-Resource",
    1: "Low-Resource",
    2: "Mid-Resource",
    3: "High-Resource",
}

# Classification descriptions
CLASSIFICATION_DESCRIPTIONS: dict[int, str] = {
    0: "Minimal digital presence, MT-based access is the most practical route",
    1: "Limited but usable data, candidate for continual pretraining",
    2: "Substantial resources, light adaptation can close performance gaps",
    3: "Abundant web-scale corpora, comprehensive LLM support",
}

# Default CSV path (relative to this file's parent package)
_DEFAULT_CSV_FILENAME = "language_digital_presence.csv"

# Well-known aliases: maps common alternative names to ISO-3 codes.
# The CSV uses one canonical name per language, but users may use other names.
_LANGUAGE_ALIASES: dict[str, str] = {
    "chichewa": "nya",
    "chewa": "nya",
    "chicheŵa": "nya",
    "guarani": "gug",
    "guaraní": "gug",
    "paraguayan guarani": "gug",
    "te reo māori": "mri",
    "te reo maori": "mri",
    "māori": "mri",
    "inuktitut": "iku",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LanguageInfo:
    """Information about a language's digital presence and classification."""
    
    name: str
    iso3_code: str
    classification: str
    classification_id: int
    num_speakers: int
    corpus_size: int
    corpus_metric: str
    language_family: str
    script_type: str
    category: str
    country: str
    
    def __str__(self) -> str:
        """Format language info for display."""
        speakers_fmt = f"{self.num_speakers:,}"
        corpus_fmt = f"{self.corpus_size:,}"
        
        lines = [
            "",
            "=" * 60,
            f"  Language Name:      {self.name}",
            f"  ISO-3 Code:         {self.iso3_code}",
            f"  Classification:     {self.classification}",
            f"  Number of Speakers: {speakers_fmt}",
            f"  Corpus Size:        {corpus_fmt} {self.corpus_metric}",
            f"  Language Family:    {self.language_family}",
            f"  Script Type:        {self.script_type}",
            f"  Category:           {self.category}",
            "=" * 60,
            "",
            f"  ℹ️  {CLASSIFICATION_DESCRIPTIONS.get(self.classification_id, '')}",
            "",
        ]
        return "\n".join(lines)


@dataclass
class AmbiguousMatch:
    """Represents an ambiguous language match."""
    
    name: str
    iso3_code: str
    classification: str
    num_speakers: int
    
    
# =============================================================================
# LANGUAGE CLASSIFIER
# =============================================================================

class LanguageClassifier:
    """
    Classifier for language resource levels based on digital presence data.
    
    Loads language metadata from CSV and provides lookup by ISO-3 code or name.
    Handles ambiguous name matches gracefully.
    """
    
    def __init__(self, csv_path: Path | None = None):
        """
        Initialize the classifier with language data.
        
        Args:
            csv_path: Path to CSV file with language data.
                     If None, uses the default bundled CSV.
        """
        if csv_path is None:
            csv_path = self._get_default_csv_path()
        
        self.csv_path = csv_path
        self._data: dict[str, dict[str, Any]] = {}  # iso3_code -> row data
        self._name_to_codes: dict[str, list[str]] = {}  # lowercase name -> list of iso3 codes
        self._load_data()
    
    def _get_default_csv_path(self) -> Path:
        """Get the default CSV path bundled with the package."""
        # CSV is at data/language_digital_presence.csv (repo root)
        repo_root = Path(__file__).parent.parent.parent
        return repo_root / "data" / _DEFAULT_CSV_FILENAME
    
    def _load_data(self) -> None:
        """Load language data from CSV file."""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Language data CSV not found: {self.csv_path}\n"
                f"Please ensure the file exists or provide a valid path."
            )
        
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                iso3 = row.get("lang_code", "").strip().lower()
                name = row.get("language", "").strip()
                
                if not iso3 or not name:
                    continue
                
                # Store by ISO-3 code
                self._data[iso3] = row
                
                # Build name -> codes mapping for ambiguity detection
                name_lower = name.lower()
                if name_lower not in self._name_to_codes:
                    self._name_to_codes[name_lower] = []
                self._name_to_codes[name_lower].append(iso3)
                
                # Also index partial names for fuzzy matching
                # e.g., "Fiji Hindi" should be findable by "Hindi"
                words = name_lower.split()
                for word in words:
                    if len(word) > 2:  # Skip very short words
                        if word not in self._name_to_codes:
                            self._name_to_codes[word] = []
                        if iso3 not in self._name_to_codes[word]:
                            self._name_to_codes[word].append(iso3)
    
    def _find_matching_codes(self, query: str) -> list[str]:
        """
        Find all ISO-3 codes matching a query (name or code).
        
        Args:
            query: Language name or ISO-3 code
            
        Returns:
            List of matching ISO-3 codes
        """
        query_lower = query.strip().lower()
        
        # 0. Check well-known aliases first
        if query_lower in _LANGUAGE_ALIASES:
            alias_code = _LANGUAGE_ALIASES[query_lower]
            if alias_code in self._data:
                return [alias_code]
        
        # 1. Check if it's an exact ISO-3 code match
        if query_lower in self._data:
            return [query_lower]
        
        # 2. Check for exact full-name match
        if query_lower in self._name_to_codes:
            codes = self._name_to_codes[query_lower]
            # Prefer full name matches over partial word matches
            exact_matches = [
                code for code in codes
                if self._data[code]["language"].lower() == query_lower
            ]
            if len(exact_matches) == 1:
                return exact_matches
            elif exact_matches:
                return exact_matches
            return codes
        
        # 3. Check for whole-word matches only (not substring)
        # A query like "chichewa" should NOT match a word "wa" inside it
        query_words = set(query_lower.split())
        matches = set()
        for name_key, codes in self._name_to_codes.items():
            name_words = set(name_key.split())
            # Match if any query word is an indexed key, or any indexed key is a query word
            if query_words & name_words:
                matches.update(codes)
        
        if matches:
            return list(matches)
        
        # 4. Last resort: check if the full query appears as a substring of any
        # language name, or any full language name appears in the query.
        # Require minimum length of 4 to avoid false positives like "Wa" in "Chichewa".
        for code, row in self._data.items():
            lang_name = row.get("language", "").lower()
            if len(lang_name) >= 4 and lang_name in query_lower:
                matches.add(code)
            elif len(query_lower) >= 4 and query_lower in lang_name:
                matches.add(code)
        
        return list(matches)
    
    def lookup(self, query: str) -> LanguageInfo | list[AmbiguousMatch]:
        """
        Look up language information by ISO-3 code or name.
        
        Args:
            query: Language name or ISO-3 code (case-insensitive)
            
        Returns:
            LanguageInfo if unique match found, or list of AmbiguousMatch if ambiguous
            
        Raises:
            ValueError: If no matching language found
        """
        matching_codes = self._find_matching_codes(query)
        
        if not matching_codes:
            raise ValueError(
                f"Language not found: '{query}'\n"
                f"Please check the spelling or use an ISO-3 code."
            )
        
        if len(matching_codes) == 1:
            # Unique match - return full info
            return self._build_language_info(matching_codes[0])
        
        # Multiple matches - return ambiguous results
        return [self._build_ambiguous_match(code) for code in sorted(matching_codes)]
    
    def _build_language_info(self, iso3_code: str) -> LanguageInfo:
        """Build LanguageInfo from stored data."""
        row = self._data[iso3_code]
        cluster_id = int(row.get("cluster_class", 0))
        
        return LanguageInfo(
            name=row.get("language", "Unknown"),
            iso3_code=iso3_code,
            classification=CLASSIFICATION_LABELS.get(cluster_id, "Unknown"),
            classification_id=cluster_id,
            num_speakers=int(row.get("number_of_speakers", 0)),
            corpus_size=int(row.get("corpus_size", 0)),
            corpus_metric=row.get("corpus_metric_used", "words"),
            language_family=row.get("family", "Unknown"),
            script_type=row.get("script_type", "Unknown"),
            category=row.get("category", "Unknown"),
            country=row.get("country", "Unknown"),
        )
    
    def _build_ambiguous_match(self, iso3_code: str) -> AmbiguousMatch:
        """Build AmbiguousMatch from stored data."""
        row = self._data[iso3_code]
        cluster_id = int(row.get("cluster_class", 0))
        
        return AmbiguousMatch(
            name=row.get("language", "Unknown"),
            iso3_code=iso3_code,
            classification=CLASSIFICATION_LABELS.get(cluster_id, "Unknown"),
            num_speakers=int(row.get("number_of_speakers", 0)),
        )
    
    def list_all_languages(self) -> list[LanguageInfo]:
        """Return info for all languages in the database."""
        return [self._build_language_info(code) for code in sorted(self._data.keys())]
    
    def get_languages_by_classification(self, classification_id: int) -> list[LanguageInfo]:
        """Get all languages with a specific classification."""
        return [
            self._build_language_info(code)
            for code in sorted(self._data.keys())
            if int(self._data[code].get("cluster_class", -1)) == classification_id
        ]


# =============================================================================
# CLI DISPLAY HELPERS
# =============================================================================

def format_ambiguous_results(query: str, matches: list[AmbiguousMatch]) -> str:
    """Format ambiguous match results for CLI display."""
    lines = [
        "",
        "=" * 70,
        f"  ⚠️  Ambiguous language name: \"{query}\"",
        "=" * 70,
        "",
        f"  Multiple languages found matching \"{query}\":",
        "",
        f"  {'ISO-3':<8} {'Language Name':<30} {'Classification':<22} {'Speakers':>12}",
        "  " + "─" * 74,
    ]
    
    for match in matches:
        speakers_fmt = f"{match.num_speakers:,}"
        lines.append(
            f"  {match.iso3_code:<8} {match.name:<30} {match.classification:<22} {speakers_fmt:>12}"
        )
    
    lines.extend([
        "",
        "  Please re-run with a specific ISO-3 code:",
        f"    python -m byol.language_resource_assessment --task language-classification --tgt-lang <iso3-code>",
        "",
    ])
    
    return "\n".join(lines)


# =============================================================================
# ENTRY POINTS
# =============================================================================

def run_language_classification(
    tgt_lang: str,
    csv_path: Path | None = None,
    **kwargs: Any,
) -> int:
    """
    Main entry point for language classification task.
    
    Looks up language by ISO-3 code or name and displays classification info.
    Handles ambiguous names by showing all matches.
    
    Args:
        tgt_lang: Target language (ISO-3 code or name)
        csv_path: Optional path to CSV file with language data
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Exit code (0 for success, 1 for ambiguous/error)
    """
    try:
        classifier = LanguageClassifier(csv_path)
        result = classifier.lookup(tgt_lang)
        
        if isinstance(result, LanguageInfo):
            # Unique match - print info
            print(str(result))
            return 0
        else:
            # Ambiguous - print matches and suggest using ISO-3 code
            print(format_ambiguous_results(tgt_lang, result))
            return 1
            
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ❌ Error: {e}\n", file=sys.stderr)
        return 1


def run_digital_presence_analysis(
    tgt_lang: str,
    data_path: Path | None = None,
    output_dir: Path | None = None,
    sources: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """
    Main entry point for language digital presence analysis task.
    
    Note: This is now an alias for run_language_classification for backward compatibility.
    For detailed corpus analysis across multiple sources, use dedicated tools.
    
    Args:
        tgt_lang: Target language name to analyze
        data_path: Path to CSV file with presence data
        output_dir: Output directory for results (unused)
        sources: List of sources to analyze (unused)
        **kwargs: Additional arguments
    """
    run_language_classification(tgt_lang, csv_path=data_path, **kwargs)


__all__ = [
    "LanguageClassifier",
    "LanguageInfo",
    "AmbiguousMatch",
    "CLASSIFICATION_LABELS",
    "CLASSIFICATION_DESCRIPTIONS",
    "run_language_classification",
    "run_digital_presence_analysis",
]
