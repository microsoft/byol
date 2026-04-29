# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Common exception hierarchy for BYOL.

Provides a structured exception hierarchy that allows for:
- Programmatic error handling by exception type
- Informative error messages with context
- Consistent error patterns across all BYOL packages

Usage:
    from byol.common.exceptions import TranslatorNotFoundError
    
    raise TranslatorNotFoundError("unknown-model", available=["gpt-5", "nllb-200-3.3b"])
"""

from __future__ import annotations

from typing import Any, Optional, Sequence


class BYOLError(Exception):
    """
    Base exception for all BYOL errors.
    
    All BYOL exceptions inherit from this, making it easy to catch
    any BYOL-related error:
    
        try:
            result = translate(text, target="es", model="gpt-5")
        except BYOLError as e:
            logger.error(f"Translation failed: {e}")
    """
    
    def __init__(self, message: str, **context: Any):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message.
            **context: Additional context that will be available as attributes.
        """
        super().__init__(message)
        self.message = message
        self.context = context
        
        # Set context items as attributes for easy access
        for key, value in context.items():
            setattr(self, key, value)
    
    def __str__(self) -> str:
        return self.message
    
    def __repr__(self) -> str:
        context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        if context_str:
            return f"{self.__class__.__name__}({self.message!r}, {context_str})"
        return f"{self.__class__.__name__}({self.message!r})"


class ConfigurationError(BYOLError):
    """
    Raised when there's an error in configuration.
    
    Examples:
    - Missing required config file
    - Invalid YAML syntax
    - Missing required configuration key
    """
    
    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        missing_key: Optional[str] = None,
        **context: Any,
    ):
        super().__init__(
            message,
            config_file=config_file,
            missing_key=missing_key,
            **context,
        )


class EnvironmentError(BYOLError):
    """
    Raised when a required environment variable is missing or invalid.
    
    Example:
        raise EnvironmentError(
            "Azure OpenAI endpoint not configured",
            variable="AZURE_OPENAI_ENDPOINT",
            required_for="Azure OpenAI translation"
        )
    """
    
    def __init__(
        self,
        message: str,
        variable: Optional[str] = None,
        required_for: Optional[str] = None,
        **context: Any,
    ):
        super().__init__(
            message,
            variable=variable,
            required_for=required_for,
            **context,
        )


def _format_available_models(model_names: Sequence[str]) -> str:
    """Format model names as a grouped table by provider."""
    try:
        from byol.translation_backends.registry import MODEL_REGISTRY
        # Group by provider
        groups: dict[str, list[str]] = {}
        for name in model_names:
            cfg = MODEL_REGISTRY.get(name)
            provider = cfg.provider if cfg else "Other"
            groups.setdefault(provider, []).append(name)
        
        lines = []
        for provider in sorted(groups):
            models_str = ", ".join(sorted(groups[provider]))
            lines.append(f"  {provider:20s} {models_str}")
        return "\n".join(lines)
    except ImportError:
        # Fallback if registry not available
        return "  " + ", ".join(model_names)


class TranslatorNotFoundError(BYOLError):
    """
    Raised when a requested translator/model is not found in the registry.
    
    Includes helpful information about available translators grouped by provider.
    
    Example:
        raise TranslatorNotFoundError(
            "unknown-model",
            available=["gpt-5", "nllb-200-3.3b", "microsoft-translator"]
        )
    """
    
    def __init__(
        self,
        model_name: str,
        available: Optional[Sequence[str]] = None,
        **context: Any,
    ):
        available_list = sorted(available) if available else []
        
        message = f"Model '{model_name}' not found."
        if available_list:
            message += "\n\nAvailable models:\n"
            message += _format_available_models(available_list)
        
        super().__init__(
            message,
            model_name=model_name,
            available=available_list,
            **context,
        )


class BackendImportError(BYOLError):
    """
    Raised when a translation backend cannot be imported.
    
    This typically happens when:
    - The backend module doesn't exist
    - Required dependencies are not installed
    - There's a syntax error in the backend code
    
    Example:
        raise BackendImportError(
            "byol.translation_backends.api.azure_openai:AzureOpenAIGPT4Translator",
            original_error=import_error
        )
    """
    
    def __init__(
        self,
        backend_path: str,
        original_error: Optional[Exception] = None,
        **context: Any,
    ):
        message = f"Failed to import backend: {backend_path}"
        if original_error:
            message += f"\nCause: {original_error}"
        
        super().__init__(
            message,
            backend_path=backend_path,
            original_error=original_error,
            **context,
        )


class LanguageNotSupportedError(BYOLError):
    """
    Raised when a language is not supported by a translator.
    
    Example:
        raise LanguageNotSupportedError(
            language="Klingon",
            translator="microsoft-translator",
            supported=["English", "Spanish", "French", ...]
        )
    """
    
    def __init__(
        self,
        language: str,
        translator: Optional[str] = None,
        supported: Optional[Sequence[str]] = None,
        **context: Any,
    ):
        message = f"Language '{language}' is not supported"
        if translator:
            message += f" by {translator}"
        message += "."
        
        if supported and len(supported) <= 20:
            message += f"\nSupported languages: {', '.join(sorted(supported))}"
        elif supported:
            message += f"\nSupported languages: {len(supported)} available"
        
        super().__init__(
            message,
            language=language,
            translator=translator,
            supported=list(supported) if supported else None,
            **context,
        )


class TranslationError(BYOLError):
    """
    Raised when a translation operation fails.
    
    Example:
        raise TranslationError(
            "API request failed",
            text=text[:100],
            model="gpt-5",
            original_error=api_error
        )
    """
    
    def __init__(
        self,
        message: str,
        text: Optional[str] = None,
        model: Optional[str] = None,
        original_error: Optional[Exception] = None,
        **context: Any,
    ):
        # Truncate text for error message
        text_preview = text[:100] + "..." if text and len(text) > 100 else text
        
        super().__init__(
            message,
            text=text_preview,
            model=model,
            original_error=original_error,
            **context,
        )


class RateLimitError(TranslationError):
    """
    Raised when a rate limit is hit.
    
    Includes retry information when available.
    
    Example:
        raise RateLimitError(
            "Rate limit exceeded",
            model="gpt-5",
            retry_after=60
        )
    """
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        **context: Any,
    ):
        if retry_after:
            message += f" (retry after {retry_after:.1f}s)"
        
        super().__init__(message, retry_after=retry_after, **context)
