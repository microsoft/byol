# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Model Registry - Maps model names to their backends and configurations.

This module defines which backend handles which model, and any model-specific
default configurations.

Backend paths use the format "module.path:ClassName" for lazy importing.
"""

from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field


ModelType = Literal["api", "local"]


@dataclass
class ModelConfig:
    """
    Configuration for a translation model.
    
    Attributes:
        backend: Full import path in format "module.path:ClassName".
        model_type: Either "api" for cloud services or "local" for local models.
        provider: Provider name for display/grouping.
        description: Human-readable description of the model.
        default_params: Default parameters passed to the translator.
        hf_model_name: HuggingFace model name for local models.
        deployment_name: Azure deployment name for API models.
        supports_auto_detect: Whether the model supports automatic source language detection.
    """
    backend: str  # Format: "module.path:ClassName"
    model_type: ModelType
    provider: str
    description: str = ""
    default_params: Dict[str, Any] = field(default_factory=dict)
    hf_model_name: Optional[str] = None
    deployment_name: Optional[str] = None
    supports_auto_detect: bool = False


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # =========================================================================
    # Microsoft Translator
    # =========================================================================
    "microsoft-translator": ModelConfig(
        backend="byol.translation_backends.api.azure_translator:AzureTranslator",
        model_type="api",
        provider="Microsoft",
        description="Azure Translator Service - supports 100+ languages",
        supports_auto_detect=True,
    ),
    
    # =========================================================================
    # Google Translator
    # =========================================================================
    "google-translator": ModelConfig(
        backend="byol.translation_backends.api.google:GoogleTranslator",
        model_type="api",
        provider="Google",
        description="Google Cloud Translation API v3",
        supports_auto_detect=True,
    ),
    
    # =========================================================================
    # DeepSeek Models (via Azure AI Foundry)
    # =========================================================================
    "deepseek-r1": ModelConfig(
        backend="byol.translation_backends.api.deepseek:AzureDeepSeekR1Translator",
        model_type="api",
        provider="DeepSeek",
        description="DeepSeek R1 reasoning model",
        default_params={"suppress_thinking": False},
        supports_auto_detect=True,
    ),
    "deepseek-r1-0528": ModelConfig(
        backend="byol.translation_backends.api.deepseek:AzureDeepSeekR1Translator",
        model_type="api",
        provider="DeepSeek",
        description="DeepSeek R1 (May 2025 version)",
        default_params={"suppress_thinking": False, "model_variant": "0528"},
        supports_auto_detect=True,
    ),
    
    # =========================================================================
    # OpenAI GPT-4 Models (via Azure OpenAI)
    # =========================================================================
    "gpt-4o": ModelConfig(
        backend="byol.translation_backends.api.azure_openai:AzureOpenAIGPT4Translator",
        model_type="api",
        provider="OpenAI",
        description="GPT-4o - fast and efficient",
        default_params={"model_name": "gpt-4o"},
        supports_auto_detect=True,
    ),
    "gpt-4.1": ModelConfig(
        backend="byol.translation_backends.api.azure_openai:AzureOpenAIGPT4Translator",
        model_type="api",
        provider="OpenAI",
        description="GPT-4.1",
        default_params={"model_name": "gpt-4.1"},
        supports_auto_detect=True,
    ),
    
    # =========================================================================
    # OpenAI GPT-5 Models (via Azure OpenAI)
    # =========================================================================
    "gpt-5": ModelConfig(
        backend="byol.translation_backends.api.azure_openai:AzureOpenAIGPT5Translator",
        model_type="api",
        provider="OpenAI",
        description="GPT-5 reasoning model",
        default_params={"model_name": "gpt-5", "reasoning_effort": "low"},
        supports_auto_detect=True,
    ),
    "gpt-5-mini": ModelConfig(
        backend="byol.translation_backends.api.azure_openai:AzureOpenAIGPT5Translator",
        model_type="api",
        provider="OpenAI",
        description="GPT-5 Mini - smaller, faster",
        default_params={"model_name": "gpt-5-mini", "reasoning_effort": "low"},
        supports_auto_detect=True,
    ),
    "gpt-5-nano": ModelConfig(
        backend="byol.translation_backends.api.azure_openai:AzureOpenAIGPT5Translator",
        model_type="api",
        provider="OpenAI",
        description="GPT-5 Nano - smallest, fastest",
        default_params={"model_name": "gpt-5-nano", "reasoning_effort": "low"},
        supports_auto_detect=True,
    ),
    "gpt-5-chat": ModelConfig(
        backend="byol.translation_backends.api.azure_openai:AzureOpenAIGPT4Translator",
        model_type="api",
        provider="OpenAI",
        description="GPT-5 Chat - non-reasoning model",
        default_params={"model_name": "gpt-5-chat"},
        supports_auto_detect=True,
    ),
    
    # =========================================================================
    # NLLB Models (Local)
    # =========================================================================
    "nllb-200-600m": ModelConfig(
        backend="byol.translation_backends.local.nllb:NLLBTranslator",
        model_type="local",
        provider="Meta",
        description="NLLB 200 Distilled 600M - fastest",
        hf_model_name="facebook/nllb-200-distilled-600M",
    ),
    "nllb-200-1.3b": ModelConfig(
        backend="byol.translation_backends.local.nllb:NLLBTranslator",
        model_type="local",
        provider="Meta",
        description="NLLB 200 1.3B - balanced",
        hf_model_name="facebook/nllb-200-1.3B",
    ),
    "nllb-200-3.3b": ModelConfig(
        backend="byol.translation_backends.local.nllb:NLLBTranslator",
        model_type="local",
        provider="Meta",
        description="NLLB 200 3.3B - most accurate",
        hf_model_name="facebook/nllb-200-3.3B",
    ),
    
    # =========================================================================
    # SeamlessM4T Models (Local)
    # =========================================================================
    "seamless-m4t-medium": ModelConfig(
        backend="byol.translation_backends.local.seamless:SeamlessM4TTranslator",
        model_type="local",
        provider="Meta",
        description="SeamlessM4T Medium",
        hf_model_name="facebook/hf-seamless-m4t-medium",
    ),
    "seamless-m4t-large": ModelConfig(
        backend="byol.translation_backends.local.seamless:SeamlessM4TTranslator",
        model_type="local",
        provider="Meta",
        description="SeamlessM4T Large",
        hf_model_name="facebook/hf-seamless-m4t-large",
    ),
    
    # =========================================================================
    # Marian MT Models (Local)
    # =========================================================================
    "marian": ModelConfig(
        backend="byol.translation_backends.local.marian:MarianTranslator",
        model_type="local",
        provider="Helsinki-NLP",
        description="Marian MT - specify src/tgt for model selection",
    ),
    
    # =========================================================================
    # MADLAD Models (Local)
    # =========================================================================
    "madlad-400-3b": ModelConfig(
        backend="byol.translation_backends.local.madlad:MadladTranslator",
        model_type="local",
        provider="Google",
        description="MADLAD 400 3B - 400+ languages",
        hf_model_name="jbochi/madlad400-3b-mt",
    ),
    "madlad-400-7b": ModelConfig(
        backend="byol.translation_backends.local.madlad:MadladTranslator",
        model_type="local",
        provider="Google",
        description="MADLAD 400 7B - larger model",
        hf_model_name="jbochi/madlad400-7b-mt",
    ),
    
    # =========================================================================
    # Aya Models (Local)
    # =========================================================================
    "aya-101": ModelConfig(
        backend="byol.translation_backends.local.aya101:Aya101Translator",
        model_type="local",
        provider="Cohere",
        description="Aya 101 - 101 languages",
        hf_model_name="CohereForAI/aya-101",
    ),
    
    # =========================================================================
    # TranslateGemma (Local)
    # =========================================================================
    "translategemma": ModelConfig(
        backend="byol.translation_backends.local.translate_gemma:TranslateGemmaTranslator",
        model_type="local",
        provider="Google",
        description="TranslateGemma 12B - Google's translation model",
        hf_model_name="google/translategemma-12b-it",
    ),
    
    # =========================================================================
    # Gemma 3 Models (Local)
    # =========================================================================
    "gemma-3-1b-it": ModelConfig(
        backend="byol.translation_backends.local.gemma3:Gemma3Translator",
        model_type="local",
        provider="Google",
        description="Gemma 3 1B - fastest, smallest",
        hf_model_name="google/gemma-3-1b-it",
        default_params={"model_name": "gemma-3-1b-it"},
    ),
    "gemma-3-4b-it": ModelConfig(
        backend="byol.translation_backends.local.gemma3:Gemma3Translator",
        model_type="local",
        provider="Google",
        description="Gemma 3 4B - balanced (default)",
        hf_model_name="google/gemma-3-4b-it",
        default_params={"model_name": "gemma-3-4b-it"},
    ),
    "gemma-3-12b-it": ModelConfig(
        backend="byol.translation_backends.local.gemma3:Gemma3Translator",
        model_type="local",
        provider="Google",
        description="Gemma 3 12B - larger, more accurate",
        hf_model_name="google/gemma-3-12b-it",
        default_params={"model_name": "gemma-3-12b-it"},
    ),
    "gemma-3-27b-it": ModelConfig(
        backend="byol.translation_backends.local.gemma3:Gemma3Translator",
        model_type="local",
        provider="Google",
        description="Gemma 3 27B - largest, best quality",
        hf_model_name="google/gemma-3-27b-it",
        default_params={"model_name": "gemma-3-27b-it"},
    ),
    
    # =========================================================================
    # Qwen3 Models (Local)
    # =========================================================================
    "qwen3-4b": ModelConfig(
        backend="byol.translation_backends.local.qwen3:Qwen3Translator",
        model_type="local",
        provider="Alibaba",
        description="Qwen3 4B - smallest, fastest",
        hf_model_name="Qwen/Qwen3-4B",
        default_params={"model_name": "Qwen/Qwen3-4B"},
    ),
    "qwen3-8b": ModelConfig(
        backend="byol.translation_backends.local.qwen3:Qwen3Translator",
        model_type="local",
        provider="Alibaba",
        description="Qwen3 8B - balanced",
        hf_model_name="Qwen/Qwen3-8B",
        default_params={"model_name": "Qwen/Qwen3-8B"},
    ),
    "qwen3-14b": ModelConfig(
        backend="byol.translation_backends.local.qwen3:Qwen3Translator",
        model_type="local",
        provider="Alibaba",
        description="Qwen3 14B - largest, best quality",
        hf_model_name="Qwen/Qwen3-14B",
        default_params={"model_name": "Qwen/Qwen3-14B"},
    ),
    
    # =========================================================================
    # Apertus Models (Local)
    # =========================================================================
    "apertus-8b": ModelConfig(
        backend="byol.translation_backends.local.apertus:ApertusTranslator",
        model_type="local",
        provider="Swiss AI",
        description="Apertus 8B Instruct - Swiss AI's multilingual model",
        hf_model_name="swiss-ai/Apertus-8B-Instruct-2509",
        default_params={"model_name": "apertus-8b"},
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_models_by_provider() -> Dict[str, List[str]]:
    """Get models grouped by provider."""
    providers: Dict[str, List[str]] = {}
    for model_name, config in MODEL_REGISTRY.items():
        if config.provider not in providers:
            providers[config.provider] = []
        providers[config.provider].append(model_name)
    return providers


def get_models_by_type(model_type: ModelType) -> List[str]:
    """Get models filtered by type ('api' or 'local')."""
    return [
        name for name, config in MODEL_REGISTRY.items()
        if config.model_type == model_type
    ]


def get_auto_detect_models() -> frozenset[str]:
    """Get models that support automatic source language detection."""
    return frozenset(
        name for name, config in MODEL_REGISTRY.items()
        if config.supports_auto_detect
    )


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a model."""
    return MODEL_REGISTRY.get(model_name.lower())


def is_model_supported(model_name: str) -> bool:
    """Check if a model is supported."""
    return model_name.lower() in MODEL_REGISTRY


__all__ = [
    # Types
    "ModelType",
    "ModelConfig",
    # Registry
    "MODEL_REGISTRY",
    # Functions
    "get_models_by_provider",
    "get_models_by_type",
    "get_auto_detect_models",
    "get_model_config",
    "is_model_supported",
]
