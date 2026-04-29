# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Apertus Translation Backend.

Dedicated translator for Swiss AI's Apertus instruction-tuned model.
Supports model: Apertus-8B-Instruct-2509.
"""

from functools import lru_cache
from typing import List, Optional

from byol.common.logging import get_logger
from byol.translation_backends.base import BaseTranslator
from byol.translation_backends.config import get_translation_prompt
from byol.translation_backends.local._utils import (
    safe_hf_login, 
    get_torch_device,
    normalize_device,
    register_model_cache,
)

logger = get_logger(__name__)

# Supported Apertus model checkpoints
APERTUS_MODELS = {
    "apertus-8b": "swiss-ai/Apertus-8B-Instruct-2509",
}

# Default model if none specified
DEFAULT_APERTUS_MODEL = "apertus-8b"


@register_model_cache
@lru_cache(maxsize=2)
def _load_apertus_model_and_tokenizer(model_name: str, device: str):
    """
    Load and cache Apertus model and tokenizer.
    
    Args:
        model_name: Either a short name (e.g., "apertus-8b") or 
                    full HuggingFace model path (e.g., "swiss-ai/Apertus-8B-Instruct-2509").
        device: Target device string (e.g., "cuda:0", "cuda:3", "cpu").
    
    Returns:
        Tuple of (tokenizer, model).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Device is already normalized by caller
    device = get_torch_device(device)
    safe_hf_login()
    
    # Resolve short name to full HuggingFace path if needed
    if model_name.lower() in APERTUS_MODELS:
        hf_model_name = APERTUS_MODELS[model_name.lower()]
    elif "apertus" in model_name.lower():
        hf_model_name = model_name
    else:
        # Assume it's already a full path
        hf_model_name = model_name
    
    logger.info(f"Loading Apertus model: {hf_model_name} on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Determine device_map based on requested device
    # Using {"": device} ensures the model loads on the specific GPU
    if "cuda" in device:
        device_map = {"": device}
    else:
        device_map = None
    
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        device_map=device_map,
    ).eval()
    
    logger.info(f"Apertus model loaded successfully: {hf_model_name} on {device}")
    
    return tokenizer, model


class ApertusTranslator(BaseTranslator):
    """
    Translator using Swiss AI's Apertus instruction-tuned model.
    
    Apertus is a multilingual instruction-following model.
    
    Supported models:
        - apertus-8b (Apertus-8B-Instruct-2509)
    
    Example:
        >>> translator = ApertusTranslator(
        ...     model_name="apertus-8b",
        ...     tgt_lang="Spanish",
        ...     device="cuda:0",
        ... )
        >>> result = translator.translate("Hello world")
    """

    name = "apertus"
    translator_type = "local"

    DEFAULT_MAX_NEW_TOKENS = 512
    DEFAULT_TEMPERATURE = 0.8
    DEFAULT_TOP_P = 0.9
    DEFAULT_DO_SAMPLE = False  # Greedy decoding by default - more stable
    DEFAULT_REPETITION_PENALTY = 1.0  # No penalty by default

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        model_name: str = DEFAULT_APERTUS_MODEL,
        device: str = "cuda:0",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        do_sample: bool = DEFAULT_DO_SAMPLE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Apertus Translator.
        
        Args:
            src_lang: Source language code.
            tgt_lang: Target language code.
            model_name: Apertus model to use. Can be:
                - Short name: "apertus-8b"
                - Full path: "swiss-ai/Apertus-8B-Instruct-2509"
            device: Device to run on. Accepts:
                - "cuda:0", "cuda:1", etc.
                - "0", "1" (converted to cuda:X)
                - "cpu"
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (default: 0.8).
            top_p: Nucleus sampling parameter (default: 0.9).
            system_prompt: Custom system prompt. If None, uses default translation prompt.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(src_lang, tgt_lang)

        self.device = get_torch_device(device)
        self.model_name = model_name
        self.tokenizer, self.model = _load_apertus_model_and_tokenizer(model_name, self.device)

        self.system_prompt = system_prompt or get_translation_prompt(src_lang, tgt_lang, detailed=False)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        
        logger.info(f"Initialized ApertusTranslator with model: {model_name}, device: {self.device}")

    def _format_messages(self, text: str) -> list:
        """
        Format input as Apertus chat messages.
        
        Apertus uses simple message format with system and user roles.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]
        return messages

    def translate(self, text: str, **kwargs) -> str:
        """
        Translate a single text.
        
        Args:
            text: Text to translate.
            **kwargs: Override generation parameters:
                - max_new_tokens: Maximum tokens to generate (default: 512)
                - temperature: Sampling temperature (default: 0.8)
                - top_p: Nucleus sampling parameter (default: 0.9)
            
        Returns:
            Translated text.
        """
        import torch
        
        messages = self._format_messages(text)
        
        # Apply chat template
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize with add_special_tokens=False as per Apertus docs
        model_inputs = self.tokenizer(
            [prompt_text], 
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)
        
        input_length = model_inputs["input_ids"].shape[1]
        
        with torch.inference_mode():
            do_sample = kwargs.get('do_sample', self.do_sample)
            gen_kwargs = {
                "max_new_tokens": kwargs.get('max_new_tokens', self.max_new_tokens),
                "pad_token_id": self.tokenizer.eos_token_id,
                "do_sample": do_sample,
            }
            
            # Only add sampling params if do_sample is True
            if do_sample:
                gen_kwargs["temperature"] = kwargs.get('temperature', self.temperature)
                gen_kwargs["top_p"] = kwargs.get('top_p', self.top_p)
            
            # Add repetition penalty if > 1.0
            repetition_penalty = kwargs.get('repetition_penalty', self.repetition_penalty)
            if repetition_penalty > 1.0:
                gen_kwargs["repetition_penalty"] = repetition_penalty
            
            outputs = self.model.generate(**model_inputs, **gen_kwargs)
        
        # Get only the generated tokens (exclude input)
        output_ids = outputs[0][input_length:]
        translation = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return translation.strip()

    def translate_batch(self, texts: List[str], batch_size: int = 4, **kwargs) -> List[str]:
        """
        Translate multiple texts.
        
        Note: For Apertus, we process one at a time due to the specific 
        chat template requirements.
        
        Args:
            texts: List of texts to translate.
            batch_size: Ignored for Apertus (processes one at a time).
            **kwargs: Override generation parameters.
            
        Returns:
            List of translated texts.
        """
        translations = []
        for text in texts:
            translation = self.translate(text, **kwargs)
            translations.append(translation)
        return translations
