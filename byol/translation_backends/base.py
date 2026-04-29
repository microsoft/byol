# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Base translator class for all translation backends.

All translator implementations should inherit from BaseTranslator.
Models are registered in MODEL_REGISTRY in registry.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional

from byol.common.logging import get_logger

logger = get_logger(__name__)


class BaseTranslator(ABC):
    """
    Abstract base class for all translators.
    
    Models are registered in MODEL_REGISTRY (registry.py) with their backend paths.
    The unified API lazily imports backends as needed.
    
    Attributes:
        name: Identifier for the translator (for display/debugging).
        translator_type: Either "api" or "local".
    
    Example:
        class MyTranslator(BaseTranslator):
            name = "my-translator"
            translator_type = "api"
            
            def translate(self, text: str, **kwargs: Any) -> str:
                return translated_text
    """
    
    name: ClassVar[str] = ""
    translator_type: ClassVar[str] = ""  # "api" or "local"

    def __init__(
        self, 
        src_lang: str,
        tgt_lang: str,
        languages: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the translator.
        
        Args:
            src_lang: Source language code.
            tgt_lang: Target language code.
            languages: Optional mapping of language names to codes.
            **kwargs: Additional arguments (ignored, for subclass flexibility).
        """
        self._languages: Dict[str, str] = languages or {}
        self._supported_languages: List[str] = list(self._languages.keys()) if self._languages else []
        self._source: str = self._map_language(src_lang)
        self._target: str = self._map_language(tgt_lang)

    @property
    def source(self) -> str:
        """Source language code."""
        return self._source

    @property
    def target(self) -> str:
        """Target language code."""
        return self._target

    def _map_language(self, lang: str) -> str:
        """Map language name to code if languages dict is provided."""
        if not self._languages:
            return lang
        if lang in self._languages.values():
            return lang
        if lang in self._languages:
            return self._languages[lang]
        # Don't raise - let the specific translator handle unknown languages
        return lang

    def is_language_supported(self, lang: str) -> bool:
        """Check if a language is supported by this translator."""
        if not self._languages:
            return True  # No restrictions defined
        return (
            lang in self._languages or 
            lang in self._languages.values()
        )

    @abstractmethod
    def translate(self, text: str, **kwargs: Any) -> str:
        """
        Translate the given text to the target language.
        
        Args:
            text: Text to translate.
            **kwargs: Additional translator-specific options.
            
        Returns:
            Translated text.
        """
        pass

    def translate_batch(self, texts: List[str], batch_size: int = 8, **kwargs: Any) -> List[str]:
        """
        Translate multiple texts. Default implementation calls translate() in a loop.
        
        Subclasses should override this for more efficient batch processing.
        
        Args:
            texts: List of texts to translate.
            batch_size: Number of texts to process at once.
            **kwargs: Additional translator-specific options.
            
        Returns:
            List of translated texts.
        """
        return [self.translate(text, **kwargs) for text in texts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(src={self._source}, tgt={self._target})"
