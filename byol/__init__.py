# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
BYOL - Bring Your Own Language Into LLMs

A modular toolkit for working with low-resource languages in LLM applications.

Subpackages:
    - common: Shared utilities (logging, caching, exceptions)
    - language_resource_assessment: Find the best translator and LLM for your language
    - translation_backends: Unified interface to multiple translation services

Quick Start:
    from byol import translate, list_models
    
    # Translate text
    result = translate("Hello world", tgt_lang="Spanish", model="gpt-5")
    
    # See all available models
    list_models()
    
Logging:
    from byol.common import setup_logging, LogLevel
    setup_logging(level=LogLevel.DEBUG)
"""

__version__ = "0.1.0"
__author__ = "BYOL Team"

# Primary translation API - use these!
from byol.translation_backends import (
    translate,
    translate_batch,
    list_models,
    get_supported_models,
    clear_cache,
    get_cache_stats,
    MODEL_REGISTRY,
    ModelConfig,
    get_model_config,
    BaseTranslator,
)

# Common utilities
from byol.common import (
    get_logger,
    setup_logging,
    LogLevel,
    BYOLError,
    TranslatorNotFoundError,
    LanguageNotSupportedError,
)

# Legacy support (deprecated)
from byol.translation_backends import get_translator

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Primary API
    "translate",
    "translate_batch",
    "list_models",
    "get_supported_models",
    "clear_cache",
    "get_cache_stats",
    # Registry
    "MODEL_REGISTRY",
    "ModelConfig",
    "get_model_config",
    "BaseTranslator",
    # Logging
    "get_logger",
    "setup_logging",
    "LogLevel",
    # Exceptions
    "BYOLError",
    "TranslatorNotFoundError",
    "LanguageNotSupportedError",
    # Legacy
    "get_translator",
]
