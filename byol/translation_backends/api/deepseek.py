# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Azure AI Foundry DeepSeek R1 Translation Backend.

Uses DeepSeek-R1 reasoning model deployed on Azure AI Foundry.
Supports thinking/reasoning suppression for faster responses.
"""

import os
import re
from typing import Optional

from byol.common.logging import get_logger
from byol.common.exceptions import TranslationError
from byol.translation_backends.base import BaseTranslator
from byol.translation_backends.config import get_translation_prompt, get_env, ENV

logger = get_logger(__name__)


class AzureDeepSeekR1Translator(BaseTranslator):
    """
    Translator using DeepSeek-R1 model via Azure AI Foundry.
    
    Requires:
        - AZURE_AI_FOUNDRY_DEEPSEEK_R1_ENDPOINT environment variable
        - AZURE_AI_FOUNDRY_DEEPSEEK_R1_MODEL environment variable
        
    Features:
        - Reasoning model with <think> tags
        - Option to suppress thinking for faster responses
        - Automatic retry with increased token budget on truncation
        
    Example:
        >>> translator = AzureDeepSeekR1Translator(tgt_lang="Spanish")
        >>> result = translator.translate("Hello world")
    """

    name = "azure-deepseek-r1"
    translator_type = "api"
    _dotenv_loaded = False

    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_TOP_P = 0.7
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_prompt: Optional[str] = None,
        suppress_thinking: bool = False,
        **kwargs,
    ):
        super().__init__(src_lang, tgt_lang)

        # Lazy load .env file (once per class)
        if not AzureDeepSeekR1Translator._dotenv_loaded:
            from dotenv import load_dotenv
            load_dotenv()
            AzureDeepSeekR1Translator._dotenv_loaded = True
        
        # Lazy import Azure AI SDK
        from azure.identity import DefaultAzureCredential
        from azure.ai.inference import ChatCompletionsClient
        
        # Build system prompt
        base_prompt = system_prompt or get_translation_prompt(src_lang, tgt_lang)
        if suppress_thinking:
            base_prompt += " Do NOT use <think> tags. Provide the translation directly without any thinking process."
        self.system_prompt = base_prompt
        
        self.endpoint = get_env(ENV.AZURE_AI_FOUNDRY_DEEPSEEK_R1_ENDPOINT, required=True)
        self.model_name = get_env(ENV.AZURE_AI_FOUNDRY_DEEPSEEK_R1_MODEL, required=True)

        self.client = ChatCompletionsClient(
            endpoint=self.endpoint, 
            credential=DefaultAzureCredential()
        )

        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        logger.info(f"Initialized AzureDeepSeekR1Translator with model: {self.model_name}")

    def _extract_non_think_text(self, text: str) -> str:
        """
        Remove <think> tags and their content from the response.
        
        Returns:
            Cleaned text without thinking content.
        """
        # Check for unclosed <think> tag (truncated response)
        if "<think>" in text and "</think>" not in text:
            raise TranslationError(
                f"Response truncated within thinking tags. Token budget ({self.max_tokens}) "
                "may be insufficient. Consider increasing max_tokens.",
                model=self.model_name,
            )
        
        # Remove think tags and their content
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        
        if not cleaned and "<think>" in text:
            logger.warning(
                "Model only generated thinking content without an actual response."
            )
        
        return cleaned

    def translate(self, text: str, **kwargs) -> str:
        """Translate text using DeepSeek R1."""
        from azure.ai.inference.models import SystemMessage, UserMessage
        
        max_retries = 2
        current_max_tokens = self.max_tokens
        
        for attempt in range(max_retries):
            completion = self.client.complete(
                messages=[
                    SystemMessage(content=self.system_prompt),
                    UserMessage(content=text),
                ],
                max_tokens=current_max_tokens,
                model=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            raw_response = completion.choices[0].message.content.strip()
            cleaned = self._extract_non_think_text(raw_response)
            
            if cleaned:
                return cleaned
            
            # Retry with increased token budget
            if attempt < max_retries - 1:
                current_max_tokens = int(current_max_tokens * 1.5)
                logger.info(f"Retrying with increased token budget: {current_max_tokens}")
        
        raise TranslationError(
            "Failed to get valid translation after retries",
            text=text,
            model=self.model_name,
        )
