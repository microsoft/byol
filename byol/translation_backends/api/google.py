# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Google Cloud Translation Backend.

Uses Google Cloud Translation API v3.
Requires a Google Cloud project with Translation API enabled.
"""

import os
from typing import Optional

from byol.common.logging import get_logger
from byol.common.exceptions import TranslationError
from byol.translation_backends.base import BaseTranslator
from byol.translation_backends.config import get_env, ENV

logger = get_logger(__name__)


class GoogleTranslator(BaseTranslator):
    """
    Translator using Google Cloud Translation API v3.
    
    Requires:
        - GOOGLE_APPLICATION_CREDENTIALS environment variable pointing to service account key
        - GOOGLE_CLOUD_PROJECT environment variable with project ID
        
    Example:
        >>> translator = GoogleTranslator(src_lang="en", tgt_lang="es")
        >>> result = translator.translate("Hello world")
    """
    
    name = "google"
    translator_type = "api"
    _dotenv_loaded = False

    def __init__(
        self, 
        src_lang: str,
        tgt_lang: str,
        project_id: Optional[str] = None,
        location: str = "global",
        **kwargs
    ):
        super().__init__(src_lang, tgt_lang)

        # Lazy load .env file (once per class)
        if not GoogleTranslator._dotenv_loaded:
            from dotenv import load_dotenv
            load_dotenv()
            GoogleTranslator._dotenv_loaded = True
        
        # Lazy import Google Cloud SDK
        from google.cloud import translate_v3 as translate
        
        self.project_id = project_id or get_env(ENV.GOOGLE_CLOUD_PROJECT, required=True)
        self.location = location
        self.parent = f"projects/{self.project_id}/locations/{self.location}"
        
        self.client = translate.TranslationServiceClient()
        
        # Google uses empty string for auto-detection
        self.src_lang = "" if src_lang == "auto" else src_lang
        self.tgt_lang = self._resolve_google_code(tgt_lang, self.client, self.parent)
        
        logger.info(f"Initialized GoogleTranslator: {src_lang} -> {self.tgt_lang}")

    @staticmethod
    def _resolve_google_code(lang: str, client, parent: str) -> str:
        """Resolve a language identifier to the code Google Translator accepts.

        Google uses different code formats per language (typically ISO-2 or BCP-47).
        We try all known code variants and return the first one the API accepts.
        """
        from byol.common.translator_support import resolve_language_codes

        codes = resolve_language_codes(lang)
        if not codes:
            return lang

        candidates = []
        seen = set()
        for key in ("iso2", "bcp47", "iso3"):
            val = codes.get(key, "").strip()
            if val and val not in seen:
                candidates.append(val)
                seen.add(val)
        if lang not in seen:
            candidates.insert(0, lang)

        for code in candidates:
            try:
                client.translate_text(
                    request={
                        "parent": parent,
                        "contents": ["hello"],
                        "mime_type": "text/plain",
                        "target_language_code": code,
                    }
                )
                return code
            except Exception:
                continue

        return lang

    def translate(self, text: str, **kwargs) -> str:
        """Translate text using Google Cloud Translation API."""
        request = {
            "parent": self.parent,
            "contents": [text],
            "mime_type": "text/plain",
            "target_language_code": self.tgt_lang,
        }
        # Only add source language if not auto-detect
        if self.src_lang:
            request["source_language_code"] = self.src_lang
            
        response = self.client.translate_text(request=request)
        
        if response.translations:
            return response.translations[0].translated_text
        
        raise TranslationError(
            "No translations returned from Google API",
            text=text,
        )
