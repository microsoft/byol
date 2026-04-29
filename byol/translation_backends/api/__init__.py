# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
API-based translation backends.

This module provides translators that use cloud APIs:
- Azure OpenAI (GPT-4o, GPT-5, etc.)
- Azure Translator Service
- Google Cloud Translation
- Azure AI Foundry (DeepSeek)

Classes are lazily imported to avoid loading unnecessary dependencies.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from byol.translation_backends.api.azure_openai import (
        AzureOpenAIGPT4Translator,
        AzureOpenAIGPT5Translator,
    )
    from byol.translation_backends.api.azure_translator import AzureTranslator
    from byol.translation_backends.api.google import GoogleTranslator
    from byol.translation_backends.api.deepseek import AzureDeepSeekR1Translator


__all__ = [
    "AzureOpenAIGPT4Translator",
    "AzureOpenAIGPT5Translator",
    "AzureTranslator",
    "GoogleTranslator",
    "AzureDeepSeekR1Translator",
]


def __getattr__(name: str):
    """Lazy import of translator classes."""
    if name == "AzureOpenAIGPT4Translator":
        from byol.translation_backends.api.azure_openai import AzureOpenAIGPT4Translator
        return AzureOpenAIGPT4Translator
    elif name == "AzureOpenAIGPT5Translator":
        from byol.translation_backends.api.azure_openai import AzureOpenAIGPT5Translator
        return AzureOpenAIGPT5Translator
    elif name == "AzureTranslator":
        from byol.translation_backends.api.azure_translator import AzureTranslator
        return AzureTranslator
    elif name == "GoogleTranslator":
        from byol.translation_backends.api.google import GoogleTranslator
        return GoogleTranslator
    elif name == "AzureDeepSeekR1Translator":
        from byol.translation_backends.api.deepseek import AzureDeepSeekR1Translator
        return AzureDeepSeekR1Translator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
