# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Factory functions for creating translator instances.

This module provides the legacy factory interface. Prefer using the unified
`translate()` function from `byol.translation_backends` instead.
"""

from __future__ import annotations

from typing import Any, List, Optional, Type

from byol.common.logging import get_logger
from byol.common.imports import lazy_import_class
from byol.common.exceptions import TranslatorNotFoundError
from byol.translation_backends.base import BaseTranslator
from byol.translation_backends.registry import MODEL_REGISTRY, ModelConfig

logger = get_logger(__name__)


def get_translator(
    name: str, 
    src_lang: str,
    tgt_lang: str,
    **kwargs: Any,
) -> BaseTranslator:
    """
    Create a translator instance by name.
    
    Args:
        name: Model name from MODEL_REGISTRY (case-insensitive).
              Examples: "nllb-200-600m", "gpt-4o", "microsoft-translator"
        src_lang: Source language code.
        tgt_lang: Target language code.
        **kwargs: Additional arguments passed to the translator constructor.
        
    Returns:
        An instance of the requested translator.
        
    Raises:
        TranslatorNotFoundError: If the translator name is not registered.
        
    Example:
        >>> translator = get_translator("gpt-4o", src_lang="English", tgt_lang="Spanish")
        >>> result = translator.translate("Hello world")
    """
    name_lower = name.lower()
    config = MODEL_REGISTRY.get(name_lower)

    if config is None:
        available = list(MODEL_REGISTRY.keys())
        raise TranslatorNotFoundError(name, available=available)

    # Lazily import the backend class
    backend_cls: Type[BaseTranslator] = lazy_import_class(config.backend)
    
    # Build init kwargs from config defaults
    init_kwargs: dict[str, Any] = {}
    if config.default_params:
        init_kwargs.update(config.default_params)
    if config.hf_model_name:
        init_kwargs["model_name"] = config.hf_model_name
    init_kwargs.update(kwargs)

    logger.info(f"Creating translator: {name_lower} (src={src_lang}, tgt={tgt_lang})")
    return backend_cls(src_lang=src_lang, tgt_lang=tgt_lang, **init_kwargs)


def list_translators(translator_type: Optional[str] = None) -> List[str]:
    """
    List all registered translator names.
    
    Args:
        translator_type: Filter by type ("api" or "local"). None returns all.
        
    Returns:
        Sorted list of translator names.
        
    Example:
        >>> list_translators()
        ['aya-101', 'gpt-4o', 'microsoft-translator', 'nllb-200-600m', ...]
        
        >>> list_translators("api")
        ['deepseek-r1', 'google-translator', 'gpt-4o', 'microsoft-translator', ...]
    """
    if translator_type is None:
        return sorted(MODEL_REGISTRY.keys())
    
    return sorted([
        name for name, config in MODEL_REGISTRY.items()
        if config.model_type == translator_type
    ])
