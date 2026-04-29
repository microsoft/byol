# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Configuration management for language resource assessment.

This module provides:
- Default constants for batch processing, retry logic, and metrics
- YAML config loading with support for repo-level and package-level paths
- Language code resolution (full_name, iso2, iso3, flores200)
- Translator configuration building
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# =============================================================================
# DEFAULT CONSTANTS
# =============================================================================

# Batch processing
BATCH_SIZE: int = 8  # Batch size for local translators
CHECKPOINT_INTERVAL: int = 80  # Save progress every N samples (10 * BATCH_SIZE)
API_MAX_CONCURRENT: int = 32  # Max concurrent API calls per translator

# Retry configuration for rate limiting
MAX_RETRIES: int = 5
INITIAL_BACKOFF: float = 1.0  # seconds
MAX_BACKOFF: float = 60.0  # seconds
BACKOFF_MULTIPLIER: float = 2.0

# Default translators to use when none specified
DEFAULT_TRANSLATORS: list[str] = [
    "gpt-4o",
    "gpt-4.1",
]

# Required metrics for completeness check
REQUIRED_METRICS: list[str] = [
    "src2tgt_similarity_score",
    "tgt2src_similarity_score",
    "tgt2src_sacreBLEU",
    "tgt2src_chrF++",
]

# Core keys in result records (not translator-specific)
CORE_KEYS: set[str] = {"id", "domain", "text", "source_language", "target_language"}

# Metrics configuration
METRICS_BATCH_SIZE: int = 32  # Process 32 entries at once for embeddings

# Plotting configuration
PLOT_METRICS: list[str] = [
    "tgt2src_similarity_score",
    "tgt2src_sacreBLEU",
    "tgt2src_chrF++",
]
SORT_METRIC: str = "tgt2src_sacreBLEU"

# Human-friendly translator display names for plots
TRANSLATOR_DISPLAY_NAMES: dict[str, str] = {
    "azure_translator": "Azure Translator",
    "google_translator": "Google Translate",
    "hf-seamless-m4t-large": "Seamless-M4T",
    "madlad400-7b-mt": "Madlad400-7B-MT",
    "madlad400-3b-mt": "Madlad400-3B-MT",
    "nllb-200-distilled-1.3B": "NLLB-200-1.3B",
    "nllb-200-3.3B": "NLLB-200-3.3B",
    "azure-deepseek-r1": "DeepSeek-R1",
    "azure-deepseek-r1-0528": "DeepSeek-R1-0528",
    "azure-openai-gpt4o": "GPT-4o",
    "azure-openai-gpt4.1": "GPT-4.1",
    "azure-openai-gpt5": "GPT-5",
    "azure-openai-gpt5-chat": "GPT-5-Chat",
    "azure-openai-gpt5-mini": "GPT-5-Mini",
    "azure-openai-gpt5-nano": "GPT-5-Nano",
    "Qwen3-14B": "Qwen3-14B",
    "Qwen3-4B-Instruct-2507": "Qwen3-4B",
    "gemma-3-12b-it": "Gemma-3-12B",
    "gemma-3-4b-it": "Gemma-3-4B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Phi-4-mini-instruct": "Phi-4-Mini",
}


# =============================================================================
# CONFIG PATH RESOLUTION
# =============================================================================

# Package directory (where this file lives)
_PACKAGE_DIR = Path(__file__).parent


def get_config_dir() -> Path:
    """
    Get the configuration directory.
    
    Searches in order:
    1. BYOL_CONFIG_DIR environment variable
    2. Repo-level: <repo_root>/configs/language_resource_assessment/
    3. Package-level fallback: <package_dir>/configs/
    
    Returns:
        Path to the configuration directory
    """
    # 1. Environment variable override
    env_config_dir = os.environ.get("BYOL_CONFIG_DIR")
    if env_config_dir:
        return Path(env_config_dir)
    
    # 2. Repo-level (walk up to find configs/ directory)
    current = _PACKAGE_DIR
    for _ in range(5):  # Max 5 levels up
        repo_config = current / "configs" / "language_resource_assessment"
        if repo_config.exists():
            return repo_config
        current = current.parent
    
    # 3. Package-level fallback
    package_config = _PACKAGE_DIR / "configs"
    if package_config.exists():
        return package_config
    
    # If nothing found, return repo-level path (will fail on load with clear error)
    return _PACKAGE_DIR.parent.parent / "configs" / "language_resource_assessment"


def get_data_dir() -> Path:
    """
    Get the data directory.
    
    Searches in order:
    1. BYOL_DATA_DIR environment variable
    2. Repo-level: <repo_root>/data/
    3. Package-level fallback: <package_dir>/data/
    
    Returns:
        Path to the data directory
    """
    # 1. Environment variable override
    env_data_dir = os.environ.get("BYOL_DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir)
    
    # 2. Repo-level (walk up to find data/ directory)
    current = _PACKAGE_DIR
    for _ in range(5):  # Max 5 levels up
        repo_data = current / "data"
        if repo_data.exists():
            return repo_data
        current = current.parent
    
    # 3. Package-level fallback
    package_data = _PACKAGE_DIR / "data"
    if package_data.exists():
        return package_data
    
    # If nothing found, return repo-level path
    return _PACKAGE_DIR.parent.parent / "data"


# =============================================================================
# CONFIG LOADING
# =============================================================================

def load_config(config_name: str = "translators.yaml") -> dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_name: Name of the config file (default: translators.yaml)
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = get_config_dir() / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Searched in: {get_config_dir()}\n"
            f"Set BYOL_CONFIG_DIR environment variable to override."
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# LANGUAGE CODE RESOLUTION
# =============================================================================

def get_language_codes(config: dict, language_name: str) -> dict[str, str]:
    """
    Get all code formats for a language (case-insensitive).
    
    Args:
        config: The loaded configuration
        language_name: Language name (e.g., "Chichewa", "chichewa", "CHICHEWA")
        
    Returns:
        Dict with keys: full_name, iso2, iso3, flores200
        
    Raises:
        ValueError: If language is not found in configuration
    """
    lang_codes = config.get("language_codes", {})
    input_lower = language_name.lower().strip()
    
    # Build case-insensitive lookup: lowercase name/alias/code -> canonical codes
    for lang, codes in lang_codes.items():
        # Check canonical name (case-insensitive)
        if lang.lower() == input_lower:
            return codes
        # Check ISO / flores codes (iso2, iso3, flores200)
        for code_key in ("iso2", "iso3", "flores200"):
            if codes.get(code_key, "").lower() == input_lower:
                return codes
        # Check aliases (case-insensitive)
        for alias in codes.get("aliases", []):
            if alias.lower() == input_lower:
                return codes
    
    # Not found - raise error with available languages
    available = sorted(lang_codes.keys())
    raise ValueError(
        f"Unknown language: '{language_name}'\n"
        f"Supported languages: {', '.join(available)}\n"
        f"To add a new language, edit 'language_codes' in {get_config_dir() / 'translators.yaml'}"
    )


def get_canonical_language_name(config: dict, language_name: str) -> str:
    """
    Get the canonical (properly cased) language name.
    
    Args:
        config: The loaded configuration
        language_name: Language name in any case (e.g., "CHIchewa")
        
    Returns:
        Canonical language name (e.g., "Chichewa")
    """
    codes = get_language_codes(config, language_name)
    return codes["full_name"]


def resolve_language_code(
    config: dict, language_name: str, format_type: str
) -> str:
    """
    Convert a language name to the required format.
    
    Args:
        config: The loaded configuration
        language_name: Language name (e.g., "Chichewa")
        format_type: One of "full_name", "iso2", "iso3", "flores200"
        
    Returns:
        The language code in the requested format
    """
    codes = get_language_codes(config, language_name)
    return codes.get(format_type, language_name)


# =============================================================================
# TRANSLATOR CONFIGURATION
# =============================================================================

def get_translator_configs(
    config: dict,
    selected_translators: list[str],
    src_lang: str,
    tgt_lang: str,
) -> dict[str, dict]:
    """
    Build translator configurations with resolved language codes.
    
    Args:
        config: The loaded configuration
        selected_translators: List of translator names to configure
        src_lang: Source language name
        tgt_lang: Target language name
        
    Returns:
        Dict mapping translator name to its full configuration
    """
    all_translators = config.get("translators", {})
    result = {}
    
    for name in selected_translators:
        if name not in all_translators:
            print(f"Warning: Unknown translator '{name}', skipping")
            continue
        
        translator_cfg = all_translators[name].copy()
        
        # Resolve language codes based on translator's expected format
        src_format = translator_cfg.get("src_lang_format", "full_name")
        tgt_format = translator_cfg.get("tgt_lang_format", "full_name")
        
        resolved_config = {
            "src_lang": resolve_language_code(config, src_lang, src_format),
            "tgt_lang": resolve_language_code(config, tgt_lang, tgt_format),
            "translator_type": translator_cfg.get("type", "api"),
            "factory_name": translator_cfg.get("factory_name", name),
            **translator_cfg.get("params", {}),
        }
        
        result[name] = resolved_config
    
    return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "BATCH_SIZE",
    "CHECKPOINT_INTERVAL",
    "API_MAX_CONCURRENT",
    "MAX_RETRIES",
    "INITIAL_BACKOFF",
    "MAX_BACKOFF",
    "BACKOFF_MULTIPLIER",
    "DEFAULT_TRANSLATORS",
    "REQUIRED_METRICS",
    "CORE_KEYS",
    "METRICS_BATCH_SIZE",
    "PLOT_METRICS",
    "SORT_METRIC",
    "TRANSLATOR_DISPLAY_NAMES",
    # Path functions
    "get_config_dir",
    "get_data_dir",
    # Config functions
    "load_config",
    "get_language_codes",
    "get_canonical_language_name",
    "resolve_language_code",
    "get_translator_configs",
]
