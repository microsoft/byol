# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Unified Translation API - Single function interface for all translation backends.

This is the primary API for translation. Use `translate()` with a model name
and let the system handle backend selection automatically.
"""

from typing import List, Optional, Dict, Any, Type

from byol.common.logging import get_logger
from byol.common.caching import ThreadSafeCache
from byol.common.imports import lazy_import_class
from byol.common.exceptions import TranslatorNotFoundError, TranslationError

from byol.translation_backends.base import BaseTranslator
from byol.translation_backends.registry import (
    MODEL_REGISTRY,
    ModelConfig,
    get_model_config,
    get_models_by_provider,
    get_models_by_type,
    get_auto_detect_models,
)

logger = get_logger(__name__)

# Thread-safe, bounded cache for translator instances
_translator_cache: ThreadSafeCache[str, BaseTranslator] = ThreadSafeCache(
    maxsize=50, 
    name="translator_cache"
)

# Cache for backend classes (to avoid repeated imports)
_backend_class_cache: ThreadSafeCache[str, Type[BaseTranslator]] = ThreadSafeCache(
    maxsize=30,
    name="backend_class_cache"
)


def _get_backend_class(backend_path: str) -> Type[BaseTranslator]:
    """
    Lazily import and return a backend class using dynamic imports.
    
    Args:
        backend_path: Full import path in format "module.path:ClassName".
    
    Returns:
        The backend translator class.
    
    Raises:
        BackendImportError: If the backend cannot be imported.
    """
    def _import() -> Type[BaseTranslator]:
        return lazy_import_class(backend_path)
    
    return _backend_class_cache.get_or_create(backend_path, _import)


def _validate_src_lang(model: str, src_lang: str) -> None:
    """
    Validate that the model supports the given source language.
    
    Raises:
        ValueError: If src_lang is "auto" but model doesn't support auto-detection.
    """
    auto_detect_models = get_auto_detect_models()
    if src_lang == "auto" and model not in auto_detect_models:
        raise ValueError(
            f"Model '{model}' does not support automatic source language detection. "
            f"Please specify an explicit src_lang. "
            f"Models supporting auto-detection: {', '.join(sorted(auto_detect_models))}"
        )


def _get_or_create_translator(
    model: str,
    src_lang: str,
    tgt_lang: str,
    device: Optional[str] = None,
    **kwargs: Any,
) -> BaseTranslator:
    """
    Get a cached translator or create a new one.
    
    Args:
        model: Model name from the registry.
        src_lang: Source language code (or "auto" for supported models).
        tgt_lang: Target language code.
        device: GPU device for local models.
        **kwargs: Additional translator parameters.
    
    Returns:
        Translator instance.
    
    Raises:
        TranslatorNotFoundError: If the model is not in the registry.
        ValueError: If src_lang is "auto" but model doesn't support it.
    """
    config = get_model_config(model)
    if config is None:
        raise TranslatorNotFoundError(model, available=list(MODEL_REGISTRY.keys()))
    
    # Validate auto-detection support
    _validate_src_lang(model, src_lang)
    
    # Build cache key (model + languages for local models that need specific pairs)
    if config.model_type == "local":
        cache_key = f"{model}:{src_lang}:{tgt_lang}"
        if device:
            cache_key += f":{device}"
    else:
        # API translators can be reused across language pairs
        cache_key = f"{model}:{tgt_lang}"
    
    def _create_translator() -> BaseTranslator:
        # Get backend class via dynamic import
        backend_cls = _get_backend_class(config.backend)
        
        # Build initialization kwargs
        init_kwargs: Dict[str, Any] = {}
        
        # Add default params from registry
        if config.default_params:
            init_kwargs.update(config.default_params)
        
        # Add HuggingFace model name for local models
        if config.hf_model_name:
            init_kwargs["model_name"] = config.hf_model_name
        
        # Add device for local models
        if device and config.model_type == "local":
            init_kwargs["device"] = device
        
        # Override with user-provided kwargs
        init_kwargs.update(kwargs)
        
        # Set language parameters
        init_kwargs["src_lang"] = src_lang
        init_kwargs["tgt_lang"] = tgt_lang
        
        logger.info(f"Creating translator: {model} ({config.backend})")
        return backend_cls(**init_kwargs)
    
    return _translator_cache.get_or_create(cache_key, _create_translator)


def translate(
    text: str,
    tgt_lang: str,
    src_lang: str = "auto",
    model: str = "microsoft-translator",
    device: Optional[str] = None,
    raise_on_error: bool = True,
    **kwargs: Any,
) -> str:
    """
    Translate text using the specified model.
    
    This is the primary translation API. Just specify the model name and
    the system handles backend selection automatically.
    
    Args:
        text: Text to translate.
        tgt_lang: Target language (e.g., "Spanish", "nya", "swh_Latn").
        src_lang: Source language or "auto" for detection.
            API models (GPT, DeepSeek, Microsoft/Google Translator) support "auto".
            For local models (NLLB, Marian, etc.), explicit source language is required.
            Use get_auto_detect_models() to see which models support auto-detection.
        model: Model to use. See `list_models()` for options.
        device: GPU device for local models (e.g., "cuda:0").
        raise_on_error: If True (default), raise TranslationError on failure.
            If False, return empty string for backward compatibility.
        **kwargs: Model-specific parameters:
            - temperature: float (for LLM models)
            - max_tokens: int (for LLM models)
            - top_p: float (for LLM models)
            - reasoning_effort: str (for GPT-5: "low", "medium", "high")
            - suppress_thinking: bool (for DeepSeek models)
            - num_beams: int (for local seq2seq models)
    
    Returns:
        Translated text.
    
    Raises:
        TranslatorNotFoundError: If the model is not found in the registry.
        TranslationError: If translation fails and raise_on_error is True.
    
    Examples:
        >>> # API translation with auto-detection (default)
        >>> translate("Hello world", tgt_lang="Spanish", model="gpt-5")
        
        >>> # Microsoft Translator with auto-detection
        >>> translate("Hello", tgt_lang="nya", model="microsoft-translator")
        
        >>> # Local model (requires explicit source language)
        >>> translate("Hello", src_lang="eng_Latn", tgt_lang="swh_Latn", model="nllb-200-3.3B")
        
        >>> # Backward compatible: return empty string on error
        >>> translate("Hello", tgt_lang="es", model="gpt-5", raise_on_error=False)
    """
    translator = _get_or_create_translator(
        model=model,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        device=device,
        **kwargs
    )
    
    try:
        return translator.translate(text)
    except TranslationError:
        if raise_on_error:
            raise
        logger.warning(f"Translation failed for model {model}, returning empty string")
        return ""


def translate_batch(
    texts: List[str],
    tgt_lang: str,
    src_lang: str = "auto",
    model: str = "microsoft-translator",
    batch_size: int = 8,
    device: Optional[str] = None,
    **kwargs: Any,
) -> List[str]:
    """
    Translate multiple texts using the specified model.
    
    Args:
        texts: List of texts to translate.
        tgt_lang: Target language.
        src_lang: Source language or "auto" (for supported API models only).
        model: Model to use.
        batch_size: Batch size for processing.
        device: GPU device for local models.
        **kwargs: Model-specific parameters.
    
    Returns:
        List of translated texts.
    
    Example:
        >>> texts = ["Hello", "World", "How are you?"]
        >>> translate_batch(texts, tgt_lang="Spanish", model="gpt-4o")
    """
    translator = _get_or_create_translator(
        model=model,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        device=device,
        **kwargs
    )
    
    return translator.translate_batch(texts, batch_size=batch_size)


def list_models(model_type: Optional[str] = None, provider: Optional[str] = None) -> None:
    """
    Print available models in a formatted table.
    
    Args:
        model_type: Filter by "api" or "local". None shows all.
        provider: Filter by provider name. None shows all.
    
    Example:
        >>> list_models()  # Show all
        >>> list_models("api")  # API models only
        >>> list_models(provider="Meta")  # Meta models only
    """
    print("\n" + "=" * 70)
    print(" BYOL Translation - Supported Models")
    print("=" * 70)
    
    providers = get_models_by_provider()
    
    for prov, models in sorted(providers.items()):
        if provider and prov.lower() != provider.lower():
            continue
        
        print(f"\n📂 {prov.upper()}")
        print("-" * 50)
        
        for model_name in sorted(models):
            config = MODEL_REGISTRY[model_name]
            
            if model_type and config.model_type != model_type:
                continue
            
            type_badge = "🌐" if config.model_type == "api" else "💻"
            print(f"   {type_badge} {model_name:<25} {config.description}")
    
    print("\n" + "=" * 70)
    print(" Legend: 🌐 = API-based, 💻 = Local model")
    print("=" * 70 + "\n")


def get_supported_models() -> Dict[str, List[str]]:
    """
    Get dictionary of supported models grouped by provider.
    
    Returns:
        Dict mapping provider names to lists of model names.
    
    Example:
        >>> models = get_supported_models()
        >>> print(models["OpenAI"])
        ['gpt-4o', 'gpt-4.1', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-5-chat']
    """
    return get_models_by_provider()


def clear_cache() -> None:
    """Clear all translator and backend class caches."""
    _translator_cache.clear()
    _backend_class_cache.clear()
    logger.info("All caches cleared")


def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the translator cache.
    
    Returns:
        Dict with cache statistics including hits, misses, and hit rate.
    """
    return {
        "translator_cache": _translator_cache.stats(),
        "backend_class_cache": _backend_class_cache.stats(),
    }

