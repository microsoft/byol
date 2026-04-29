# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Translation Backends - Unified interface for multiple translation services.

Provides a single `translate()` function that routes to the appropriate backend
based on the model name. No need to instantiate translator classes directly.

Usage:
    from byol.translation_backends import translate, list_models
    
    # Simple translation
    result = translate("Hello world", tgt_lang="Spanish", model="gpt-5")
    
    # Microsoft Translator
    result = translate("Hello", tgt_lang="nya", model="microsoft-translator")
    
    # Local model
    result = translate("Hello", src_lang="eng_Latn", tgt_lang="swh_Latn", model="nllb-200-3.3b")
    
    # List available models
    list_models()
    
Supported Models:
    | Provider   | Models                                                    |
    |------------|-----------------------------------------------------------|
    | Microsoft  | microsoft-translator                                      |
    | Google     | google-translator                                         |
    | DeepSeek   | deepseek-r1, deepseek-r1-0528                            |
    | OpenAI     | gpt-4o, gpt-4.1, gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-chat|
    | Meta       | nllb-200-600m, nllb-200-1.3b, nllb-200-3.3b, seamless-*  |
    | Google     | madlad-400-3b, madlad-400-7b, translategemma, gemma-3-*  |
    | Alibaba    | qwen3-4b, qwen3-8b, qwen3-14b                            |
    | Apertus    | apertus-8b                                                |
    | Helsinki   | marian                                                    |
    | Cohere     | aya-101                                                   |
"""

from typing import Any

# Primary API - use these!
from byol.translation_backends.unified import (
    translate,
    translate_batch,
    list_models,
    get_supported_models,
    clear_cache,
    get_cache_stats,
)

# Registry for programmatic access
from byol.translation_backends.registry import (
    MODEL_REGISTRY,
    ModelConfig,
    get_model_config,
    get_models_by_provider,
    get_models_by_type,
    is_model_supported,
)

# Base class for custom translators
from byol.translation_backends.base import BaseTranslator

__all__ = [
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
    "get_models_by_provider",
    "get_models_by_type",
    "is_model_supported",
    # Base class (for custom implementations)
    "BaseTranslator",
    # Legacy support
    "get_translator",
]


def get_translator(name: str, src_lang: str, tgt_lang: str, **kwargs: Any) -> BaseTranslator:
    """
    Legacy function: Create a translator instance by class name.
    
    .. deprecated::
        Prefer using `translate()` directly:
        ``translate("Hello", src_lang="English", tgt_lang="Spanish", model="gpt-5")``
    
    Args:
        name: Registered name of the translator.
        src_lang: Source language code.
        tgt_lang: Target language code.
        **kwargs: Additional translator parameters.
    
    Returns:
        Translator instance.
    """
    from byol.translation_backends.factory import get_translator as _get_translator
    return _get_translator(name, src_lang, tgt_lang, **kwargs)
