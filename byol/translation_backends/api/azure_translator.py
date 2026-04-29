# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Azure Translator Service Backend.

Uses Azure AI Translation service (not Azure OpenAI).
Supports 100+ languages with automatic language detection.

Reference: https://learn.microsoft.com/en-us/azure/ai-services/translator/language-support
"""

import os
from typing import List, Optional

from byol.common.logging import get_logger
from byol.translation_backends.base import BaseTranslator
from byol.translation_backends.config import get_env, ENV

logger = get_logger(__name__)


class AzureTranslator(BaseTranslator):
    """
    Translator using Azure AI Translation service.
    
    Requires:
        - AZURE_TRANSLATOR_ENDPOINT environment variable
        - Azure Entra (AAD) authentication or Cognitive Services User role
        
    Features:
        - Automatic language detection when src_lang is not specified
        - Batch translation (up to 100 texts per request)
        
    Example:
        >>> translator = AzureTranslator(tgt_lang="es")
        >>> result = translator.translate("Hello world")
    """
    
    name = "azure_translator"
    translator_type = "api"
    _dotenv_loaded = False

    def __init__(
        self, 
        src_lang: str,
        tgt_lang: str, 
        **kwargs
    ):
        super().__init__(src_lang, tgt_lang)

        # Lazy load .env file (once per class)
        if not AzureTranslator._dotenv_loaded:
            from dotenv import load_dotenv
            load_dotenv()
            AzureTranslator._dotenv_loaded = True
        
        # Lazy import Azure SDK
        from azure.identity import DefaultAzureCredential
        from azure.ai.translation.text import TextTranslationClient
        
        endpoint = get_env(ENV.AZURE_TRANSLATOR_ENDPOINT, required=True)
        
        # Authenticate using the default identity
        self.credential = DefaultAzureCredential()
        self.client = TextTranslationClient(endpoint=endpoint, credential=self.credential)
        
        # Convert "auto" to None for Azure API auto-detection
        self.src_lang = None if src_lang == "auto" else src_lang
        self.tgt_lang = self._resolve_azure_code(tgt_lang, self.client)
        
        logger.info(f"Initialized AzureTranslator: {src_lang} -> {self.tgt_lang}")

    @staticmethod
    def _resolve_azure_code(lang: str, client) -> str:
        """Resolve a language identifier to the code Azure Translator accepts.

        Azure uses different code formats per language (ISO-2, ISO-3, or BCP-47).
        We try all known code variants for the language and return the first one
        the API accepts.
        """
        from byol.common.translator_support import resolve_language_codes

        codes = resolve_language_codes(lang)
        if not codes:
            return lang

        # Candidate codes in priority order
        candidates = []
        seen = set()
        for key in ("iso3", "iso2", "bcp47"):
            val = codes.get(key, "").strip()
            if val and val not in seen:
                candidates.append(val)
                seen.add(val)
        # Also keep the original input if not already covered
        if lang not in seen:
            candidates.insert(0, lang)

        # Probe with a lightweight translate call
        for code in candidates:
            try:
                client.translate([{"text": "hello"}], to_language=[code])
                return code
            except Exception:
                continue

        # Fallback to original input
        return lang

    def translate(self, text: str, **kwargs) -> str:
        """Translate a single text."""
        content = [{"text": text}]
        params = {"to_language": [self.tgt_lang]}
        
        # Add source language only if specified (otherwise auto-detect)
        if self.src_lang is not None:
            params["from_language"] = self.src_lang
        
        response = self.client.translate(content, **params)
        return response[0].translations[0].text

    def translate_batch(self, texts: List[str], batch_size: int = 100, **kwargs) -> List[str]:
        """
        Translate multiple texts.
        
        Azure supports up to 100 texts per request.
        """
        batch_size = min(batch_size, 100)  # Azure limit
        results = []
        
        params = {"to_language": [self.tgt_lang]}
        if self.src_lang is not None:
            params["from_language"] = self.src_lang
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            content = [{"text": text} for text in batch]
            response = self.client.translate(content, **params)
            results.extend([item.translations[0].text for item in response])
        
        return results

