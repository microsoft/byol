# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Azure OpenAI Translation Backend.

Provides two translator classes:
- AzureOpenAIGPT4Translator: For GPT-4 series models (gpt-4o, gpt-4.1, gpt-5-chat)
- AzureOpenAIGPT5Translator: For GPT-5 reasoning models (gpt-5, gpt-5-mini, gpt-5-nano)

Uses Azure Entra (AAD) authentication.
"""

import os
from typing import Optional, List

from byol.common.logging import get_logger
from byol.common.exceptions import TranslationError
from byol.translation_backends.base import BaseTranslator
from byol.translation_backends.config import get_translation_prompt, get_env, ENV

logger = get_logger(__name__)


class AzureOpenAIGPT4Translator(BaseTranslator):
    """
    Translator using GPT-4 series models deployed on Azure OpenAI.
    
    Supports: gpt-4o, gpt-4.1-preview, gpt-5-chat (non-reasoning models)
    
    Requires:
        - AZURE_OPENAI_ENDPOINT environment variable
        - Azure Entra (AAD) authentication configured
        
    Example:
        >>> translator = AzureOpenAIGPT4Translator(tgt_lang="Spanish", model_name="gpt-4o")
        >>> result = translator.translate("Hello world")
    """

    name = "azure-openai"
    translator_type = "api"
    _dotenv_loaded = False

    # Default configuration
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_TOP_P = 0.7
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_API_VERSION = "2024-12-01-preview"
    DEFAULT_MODEL_NAME = "gpt-4o"

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_version: str = DEFAULT_API_VERSION,
        model_name: str = DEFAULT_MODEL_NAME,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(src_lang, tgt_lang)

        # Lazy load .env file (once per class)
        if not AzureOpenAIGPT4Translator._dotenv_loaded:
            from dotenv import load_dotenv
            load_dotenv()
            AzureOpenAIGPT4Translator._dotenv_loaded = True

        # Lazy import Azure SDK
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from openai import AzureOpenAI

        self.api_base = get_env(ENV.AZURE_OPENAI_ENDPOINT, required=True)
        self.api_version = api_version
        self.model_name = model_name
        self.system_prompt = system_prompt or get_translation_prompt(src_lang, tgt_lang)

        logger.info(f"Initializing AzureOpenAIGPT4Translator with model: {self.model_name}")

        # Use Entra Authentication
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), 
            "https://cognitiveservices.azure.com/.default"
        )

        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.api_base,
            azure_ad_token_provider=token_provider,
        )

        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def translate(self, text: str, **kwargs) -> str:
        """Translate text using Azure OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        content = response.choices[0].message.content
        if content is None:
            raise TranslationError(
                "Received empty response from Azure OpenAI",
                text=text,
                model=self.model_name,
            )
        return content.strip()

    def translate_batch(self, texts: List[str], batch_size: int = 8, **kwargs) -> List[str]:
        """Translate multiple texts."""
        return [self.translate(text, **kwargs) for text in texts]


class AzureOpenAIGPT5Translator(BaseTranslator):
    """
    Translator using GPT-5 reasoning models deployed on Azure OpenAI.
    
    Supports: gpt-5, gpt-5-mini, gpt-5-nano (reasoning models with reasoning_effort)
    
    Requires:
        - AZURE_OPENAI_ENDPOINT environment variable
        - Azure Entra (AAD) authentication configured
        
    Example:
        >>> translator = AzureOpenAIGPT5Translator(tgt_lang="Spanish", reasoning_effort="medium")
        >>> result = translator.translate("Hello world")
    """

    name = "azure-openai-gpt5"
    translator_type = "api"
    _dotenv_loaded = False

    # Default configuration
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_API_VERSION = "2024-12-01-preview"
    DEFAULT_MODEL_NAME = "gpt-5"
    DEFAULT_REASONING_EFFORT = "low"

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_version: str = DEFAULT_API_VERSION,
        model_name: str = DEFAULT_MODEL_NAME,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(src_lang, tgt_lang)

        # Lazy load .env file (once per class)
        if not AzureOpenAIGPT5Translator._dotenv_loaded:
            from dotenv import load_dotenv
            load_dotenv()
            AzureOpenAIGPT5Translator._dotenv_loaded = True

        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from openai import AzureOpenAI

        self.api_base = get_env(ENV.AZURE_OPENAI_ENDPOINT, required=True)
        self.api_version = api_version
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.system_prompt = system_prompt or get_translation_prompt(src_lang, tgt_lang)

        logger.info(f"Initializing AzureOpenAIGPT5Translator with model: {self.model_name}")

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), 
            "https://cognitiveservices.azure.com/.default"
        )

        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.api_base,
            azure_ad_token_provider=token_provider,
        )

        self.max_tokens = max_tokens

    def translate(self, text: str, **kwargs) -> str:
        """Translate text using Azure OpenAI GPT-5 reasoning model."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
            max_completion_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
        )

        content = response.choices[0].message.content
        if content is None:
            raise TranslationError(
                "Received empty response from Azure OpenAI GPT-5",
                text=text,
                model=self.model_name,
            )
        return content.strip()

    def translate_batch(self, texts: List[str], batch_size: int = 8, **kwargs) -> List[str]:
        """Translate multiple texts."""
        return [self.translate(text, **kwargs) for text in texts]


# =============================================================================
# Backward compatibility aliases
# =============================================================================

AzureOpenAITranslator = AzureOpenAIGPT4Translator
AzureOpenAITranslatorGPT4p1 = AzureOpenAIGPT4Translator
AzureOpenAITranslatorGPT5 = AzureOpenAIGPT5Translator
