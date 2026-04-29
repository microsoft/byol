# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Central configuration for translation backends.

This module provides shared constants, default prompts, and configuration
utilities used across multiple translator implementations.
"""

from typing import Dict, Any
import os
from dataclasses import dataclass, field


# =============================================================================
# Default Generation Parameters
# =============================================================================

@dataclass
class GenerationConfig:
    """Default parameters for text generation."""
    temperature: float = 0.1
    top_p: float = 0.7
    top_k: int = 50
    max_tokens: int = 1024
    num_beams: int = 4
    do_sample: bool = True
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3


# Singleton instance for defaults
DEFAULT_GENERATION_CONFIG = GenerationConfig()


# =============================================================================
# System Prompts
# =============================================================================

def get_translation_prompt(src_lang: str, tgt_lang: str, detailed: bool = True) -> str:
    """
    Get the system prompt for translation tasks.
    
    Args:
        src_lang: Source language name (or "auto" for auto-detection).
        tgt_lang: Target language name.
        detailed: If True, use detailed prompt; if False, use concise version.
        
    Returns:
        System prompt string.
    """
    # Handle auto-detection case - omit source language from prompt
    if src_lang == "auto":
        if detailed:
            return f"""You are a professional translator. Your task is to translate text into {tgt_lang}.
Your goal is to accurately convey the meaning and nuances of the original text while adhering to {tgt_lang} grammar, vocabulary, and cultural sensitivities.
Please translate the provided text into {tgt_lang}.
Produce only the {tgt_lang} translation, without any additional explanations or commentary."""
        else:
            return f"Translate the following text into {tgt_lang}. Produce only the {tgt_lang} translation, without any additional explanations or commentary."
    
    if detailed:
        return f"""You are a professional translator, tasked with providing translations from {src_lang} into {tgt_lang} language. 
Your goal is to accurately convey the meaning and nuances of the original text while adhering to {tgt_lang} grammar, vocabulary, and cultural sensitivities.
Please translate the provided text from {src_lang} into {tgt_lang} language. 
Produce only the {tgt_lang} translation, without any additional explanations or commentary."""
    else:
        return f"Translate the following text from {src_lang} into {tgt_lang} LITERALLY. Produce only the {tgt_lang} translation, without any additional explanations or commentary."


# =============================================================================
# Environment Variables
# =============================================================================

@dataclass 
class EnvironmentConfig:
    """Environment variable names for various services."""
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str = "AZURE_OPENAI_ENDPOINT"
    AZURE_OPENAI_API_KEY: str = "AZURE_OPENAI_API_KEY"
    AZURE_OPENAI_API_VERSION: str = "AZURE_OPENAI_API_VERSION"
    
    # Azure Translator
    AZURE_TRANSLATOR_ENDPOINT: str = "AZURE_TRANSLATOR_ENDPOINT"
    AZURE_TRANSLATOR_KEY: str = "AZURE_TRANSLATOR_KEY"
    
    # Azure AI Foundry (DeepSeek)
    AZURE_AI_FOUNDRY_DEEPSEEK_R1_ENDPOINT: str = "AZURE_AI_FOUNDRY_DEEPSEEK_R1_ENDPOINT"
    AZURE_AI_FOUNDRY_DEEPSEEK_R1_MODEL: str = "AZURE_AI_FOUNDRY_DEEPSEEK_R1_MODEL"
    
    # Google Cloud
    GOOGLE_APPLICATION_CREDENTIALS: str = "GOOGLE_APPLICATION_CREDENTIALS"
    GOOGLE_CLOUD_PROJECT: str = "GOOGLE_CLOUD_PROJECT"
    
    # HuggingFace
    HF_TOKEN: str = "HF_TOKEN"


ENV = EnvironmentConfig()


def get_env(key: str, default: str = None, required: bool = False) -> str:
    """
    Get environment variable with optional default and required check.
    
    Args:
        key: Environment variable name.
        default: Default value if not set.
        required: If True, raise error when not set.
        
    Returns:
        Environment variable value.
        
    Raises:
        EnvironmentError: If required and not set.
    """
    value = os.getenv(key, default)
    if required and not value:
        raise EnvironmentError(f"Required environment variable not set: {key}")
    return value


# =============================================================================
# Model Configurations
# =============================================================================

# NLLB model variants
NLLB_MODELS = {
    "distilled-600M": "facebook/nllb-200-distilled-600M",
    "distilled-1.3B": "facebook/nllb-200-distilled-1.3B",
    "1.3B": "facebook/nllb-200-1.3B",
    "3.3B": "facebook/nllb-200-3.3B",
}

# Default device selection
def get_default_device() -> str:
    """Get the default compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"
