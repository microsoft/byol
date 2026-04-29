# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
BYOL Language Resource Assessment

Evaluate translation quality for low-resource languages via round-trip translation.
Compare different models (API translators and local LLMs) to find the best one for your language.

Available Tasks:
- find-best-translator: Evaluate translation models (NLLB, Microsoft Translator, GPT, etc.)
- find-best-llm: Evaluate general-purpose LLMs (Qwen, Gemma, etc.) for translation
- language-digital-presence: Analyze online presence of a language (planned)

Both find-best-translator and find-best-llm use the same evaluation pipeline,
just with different config files (translators.yaml vs llms.yaml).

Usage:
    python -m byol.language_resource_assessment --task find-best-translator --tgt-lang Chichewa
    python -m byol.language_resource_assessment --task find-best-llm --tgt-lang Chichewa
"""

__version__ = "0.1.0"
__author__ = "BYOL Team"

# Task entry points
from .find_best_model import run_model_evaluation, run_translator_evaluation
from .language_digital_presence import (
    run_digital_presence_analysis,
    run_language_classification,
    LanguageClassifier,
    LanguageInfo,
    CLASSIFICATION_LABELS,
)

# Shared utilities (for power users)
from .config import load_config, get_language_codes, get_canonical_language_name
from .visualize import generate_plots
from .io import load_jsonl, save_jsonl
from .metrics import compute_metrics
from .embedding import EmbeddingClient

__all__ = [
    # Tasks
    "run_model_evaluation",
    "run_translator_evaluation",  # Backward compatibility alias
    "run_digital_presence_analysis",
    "run_language_classification",
    # Language Classification
    "LanguageClassifier",
    "LanguageInfo",
    "CLASSIFICATION_LABELS",
    # Config
    "load_config",
    "get_language_codes",
    "get_canonical_language_name",
    # Utilities
    "generate_plots",
    "load_jsonl",
    "save_jsonl",
    "compute_metrics",
    "EmbeddingClient",
]
