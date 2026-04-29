# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Local model translation backends.

This module provides translators that run models locally:
- NLLB (No Language Left Behind) - Meta's multilingual translation
- Marian MT - Helsinki-NLP's translation models
- SeamlessM4T - Meta's multimodal translation
- MADLAD-400 - Google's multilingual translation
- Aya-101 - Cohere's multilingual model

Classes are lazily imported to avoid loading heavy ML dependencies.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from byol.translation_backends.local.nllb import NLLBTranslator
    from byol.translation_backends.local.marian import MarianTranslator
    from byol.translation_backends.local.seamless import SeamlessM4TTranslator
    from byol.translation_backends.local.causal_lm import HuggingFaceCausalLM
    from byol.translation_backends.local.madlad import MadladTranslator
    from byol.translation_backends.local.aya101 import Aya101Translator
    from byol.translation_backends.local.translate_gemma import TranslateGemmaTranslator
    from byol.translation_backends.local.gemma3 import Gemma3Translator
    from byol.translation_backends.local.qwen3 import Qwen3Translator
    from byol.translation_backends.local.apertus import ApertusTranslator


__all__ = [
    "NLLBTranslator",
    "MarianTranslator",
    "SeamlessM4TTranslator",
    "HuggingFaceCausalLM",
    "MadladTranslator",
    "Aya101Translator",
    "TranslateGemmaTranslator",
    "Gemma3Translator",
    "Qwen3Translator",
    "ApertusTranslator",
]


def __getattr__(name: str):
    """Lazy import of translator classes."""
    if name == "NLLBTranslator":
        from byol.translation_backends.local.nllb import NLLBTranslator
        return NLLBTranslator
    elif name == "MarianTranslator":
        from byol.translation_backends.local.marian import MarianTranslator
        return MarianTranslator
    elif name == "SeamlessM4TTranslator":
        from byol.translation_backends.local.seamless import SeamlessM4TTranslator
        return SeamlessM4TTranslator
    elif name == "HuggingFaceCausalLM":
        from byol.translation_backends.local.causal_lm import HuggingFaceCausalLM
        return HuggingFaceCausalLM
    elif name == "MadladTranslator":
        from byol.translation_backends.local.madlad import MadladTranslator
        return MadladTranslator
    elif name == "Aya101Translator":
        from byol.translation_backends.local.aya101 import Aya101Translator
        return Aya101Translator
    elif name == "TranslateGemmaTranslator":
        from byol.translation_backends.local.translate_gemma import TranslateGemmaTranslator
        return TranslateGemmaTranslator
    elif name == "Gemma3Translator":
        from byol.translation_backends.local.gemma3 import Gemma3Translator
        return Gemma3Translator
    elif name == "Qwen3Translator":
        from byol.translation_backends.local.qwen3 import Qwen3Translator
        return Qwen3Translator
    elif name == "ApertusTranslator":
        from byol.translation_backends.local.apertus import ApertusTranslator
        return ApertusTranslator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
