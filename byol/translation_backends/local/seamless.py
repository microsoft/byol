# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
SeamlessM4T Translation Backend.

Meta's multimodal translation model supporting 100+ languages.
Uses 3-letter language codes.

Reference: https://github.com/facebookresearch/seamless_communication
"""

from functools import lru_cache
from typing import List

from byol.common.logging import get_logger
from byol.translation_backends.base import BaseTranslator
from byol.translation_backends.local._utils import (
    safe_hf_login, 
    get_torch_device, 
    get_torch_dtype,
    normalize_device,
    register_model_cache,
)

logger = get_logger(__name__)


@register_model_cache
@lru_cache(maxsize=2)
def _load_seamless_model_and_processor(model_name: str, device: str):
    """Load and cache SeamlessM4T model and processor."""
    import torch
    from transformers import AutoProcessor, SeamlessM4TForTextToText
    
    # Device is already normalized by caller
    device = get_torch_device(device)
    dtype = get_torch_dtype(device)
    safe_hf_login()
    
    logger.info(f"Loading SeamlessM4T model: {model_name} on {device}")
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Load model directly on target device
    if "cuda" in device:
        model = SeamlessM4TForTextToText.from_pretrained(
            model_name,
            use_safetensors=True,
            dtype=dtype,
            device_map=device,
        )
    else:
        model = SeamlessM4TForTextToText.from_pretrained(
            model_name,
            use_safetensors=True,
            dtype=dtype,
        )
    
    model.eval()
    logger.info(f"SeamlessM4T model loaded on {device} with dtype {dtype}")
    return processor, model


class SeamlessM4TTranslator(BaseTranslator):
    """
    Translator using Meta's SeamlessM4T model for text-to-text translation.
    
    Supports 100+ languages with 3-letter language codes (e.g., "eng", "fra", "swh").
    
    Model variants:
        - facebook/hf-seamless-m4t-medium
        - facebook/hf-seamless-m4t-large (default)
    
    Example:
        >>> translator = SeamlessM4TTranslator(
        ...     src_lang="eng",
        ...     tgt_lang="swh",  # Swahili
        ... )
        >>> result = translator.translate("Hello, how are you?")
    """

    name = "seamlessm4t"
    translator_type = "local"

    DEFAULT_MODEL_NAME = "facebook/hf-seamless-m4t-large"
    DEFAULT_NUM_BEAMS = 5
    DEFAULT_TOP_K = 50
    DEFAULT_TOP_P = 1.0
    DEFAULT_DO_SAMPLE = False
    DEFAULT_LENGTH_PENALTY = 1.0
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_NO_REPEAT_NGRAM_SIZE = 5

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cuda:0",
        num_beams: int = DEFAULT_NUM_BEAMS,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        do_sample: bool = DEFAULT_DO_SAMPLE,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        temperature: float = DEFAULT_TEMPERATURE,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
        **kwargs,
    ):
        if not src_lang:
            raise ValueError("src_lang is required (e.g., 'eng')")
        if not tgt_lang:
            raise ValueError("tgt_lang is required (e.g., 'swh')")

        super().__init__(src_lang, tgt_lang)

        self.device = get_torch_device(device)
        self.processor, self.model = _load_seamless_model_and_processor(model_name, self.device)

        self.src_lang_code = src_lang
        self.tgt_lang_code = tgt_lang

        # Generation parameters
        self.num_beams = num_beams
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.length_penalty = length_penalty
        self.temperature = temperature
        self.no_repeat_ngram_size = no_repeat_ngram_size
        
        logger.info(f"Initialized SeamlessM4TTranslator: {src_lang} -> {tgt_lang}")

    def translate(self, text: str, **kwargs) -> str:
        """Translate a single text."""
        import torch
        
        # Replace double quotes to avoid <unk> tokens
        text_processed = text.replace('"', "'")
        
        inputs = self.processor(
            text=text_processed,
            src_lang=self.src_lang_code,
            return_tensors="pt"
        )
        
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                tgt_lang=self.tgt_lang_code,
                num_beams=kwargs.get('num_beams', self.num_beams),
                top_k=kwargs.get('top_k', self.top_k),
                top_p=kwargs.get('top_p', self.top_p),
                do_sample=kwargs.get('do_sample', self.do_sample),
                length_penalty=kwargs.get('length_penalty', self.length_penalty),
                temperature=kwargs.get('temperature', self.temperature),
                no_repeat_ngram_size=kwargs.get('no_repeat_ngram_size', self.no_repeat_ngram_size),
            )
        
        translation = self.processor.decode(output_tokens[0], skip_special_tokens=True)
        # Replace any <unk> tokens
        translation = translation.replace('<unk>', "'")
        
        return translation.strip()

    def translate_batch(self, texts: List[str], batch_size: int = 8, **kwargs) -> List[str]:
        """
        Translate multiple texts.
        
        Note: SeamlessM4T processes texts individually internally.
        """
        translations = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_translations = [self.translate(text, **kwargs) for text in batch_texts]
            translations.extend(batch_translations)
        
        return translations
