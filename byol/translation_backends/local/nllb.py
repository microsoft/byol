# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
NLLB (No Language Left Behind) Translation Backend.

Facebook's multilingual translation model supporting 200+ languages.
Uses Flores-200 language codes.

Reference: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
"""

from functools import lru_cache
from typing import List, Optional

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
@lru_cache(maxsize=4)
def _load_nllb_model_and_tokenizer(model_name: str, device: str):
    """Load and cache NLLB model and tokenizer."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    
    # Device is already normalized by caller
    device = get_torch_device(device)
    dtype = get_torch_dtype(device)
    
    safe_hf_login()
    
    logger.info(f"Loading NLLB model: {model_name} on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model directly on target device for better memory efficiency
    if "cuda" in device:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,  # Load directly on target GPU
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=dtype,
        )
    
    model.eval()
    logger.info(f"NLLB model loaded on {device} with dtype {dtype}")
    return tokenizer, model


class NLLBTranslator(BaseTranslator):
    """
    Translator using Facebook's NLLB (No Language Left Behind) model.
    
    Supports 200+ languages with Flores-200 language codes.
    
    Model variants:
        - facebook/nllb-200-distilled-600M (fastest, least accurate)
        - facebook/nllb-200-distilled-1.3B
        - facebook/nllb-200-1.3B
        - facebook/nllb-200-3.3B (slowest, most accurate)
    
    Language codes use Flores-200 format: e.g., "eng_Latn", "swh_Latn", "fra_Latn"
    
    Example:
        >>> translator = NLLBTranslator(
        ...     src_lang="eng_Latn",
        ...     tgt_lang="swh_Latn",  # Swahili
        ...     model_name="facebook/nllb-200-1.3B"
        ... )
        >>> result = translator.translate("Hello, how are you?")
    """

    name = "nllb"
    translator_type = "local"

    DEFAULT_MAX_LENGTH = 512
    DEFAULT_MODEL_NAME = "facebook/nllb-200-1.3B"
    DEFAULT_NUM_BEAMS = 5
    DEFAULT_TOP_K = 50
    DEFAULT_TOP_P = 1.0
    DEFAULT_DO_SAMPLE = False
    DEFAULT_LENGTH_PENALTY = 1.0
    DEFAULT_TEMPERATURE = 1.0

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cuda:0",
        num_beams: int = DEFAULT_NUM_BEAMS,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        do_sample: bool = DEFAULT_DO_SAMPLE,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs,
    ):
        if not src_lang:
            raise ValueError("src_lang is required for NLLB (e.g., 'eng_Latn')")
        if not tgt_lang:
            raise ValueError("tgt_lang is required for NLLB (e.g., 'swh_Latn')")

        super().__init__(src_lang, tgt_lang)

        self.device = get_torch_device(device)
        self.tokenizer, self.model = _load_nllb_model_and_tokenizer(model_name, self.device)

        self.max_length = max_length
        self.src_lang_code = src_lang
        self.tgt_lang_code = tgt_lang
        
        # Generation parameters
        self.num_beams = num_beams
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.length_penalty = length_penalty
        self.temperature = temperature
        self.model_name = model_name
        
        logger.info(f"Initialized NLLBTranslator: {src_lang} -> {tgt_lang}")

    def translate(self, text: str, **kwargs) -> str:
        """Translate a single text."""
        import torch
        
        # Set source language for tokenizer
        self.tokenizer.src_lang = self.src_lang_code
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get target language token ID
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang_code)
        
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=kwargs.get('max_length', self.max_length),
                num_beams=kwargs.get('num_beams', self.num_beams),
                top_k=kwargs.get('top_k', self.top_k),
                top_p=kwargs.get('top_p', self.top_p),
                do_sample=kwargs.get('do_sample', self.do_sample),
                length_penalty=kwargs.get('length_penalty', self.length_penalty),
                temperature=kwargs.get('temperature', self.temperature),
            )
        
        translation = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return translation.strip()

    def translate_batch(self, texts: List[str], batch_size: int = 8, **kwargs) -> List[str]:
        """Translate multiple texts efficiently."""
        import torch
        
        self.tokenizer.src_lang = self.src_lang_code
        translations = []
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang_code)
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=kwargs.get('max_length', self.max_length),
                    num_beams=kwargs.get('num_beams', self.num_beams),
                    top_k=kwargs.get('top_k', self.top_k),
                    top_p=kwargs.get('top_p', self.top_p),
                    do_sample=kwargs.get('do_sample', self.do_sample),
                    length_penalty=kwargs.get('length_penalty', self.length_penalty),
                    temperature=kwargs.get('temperature', self.temperature),
                )
            
            batch_translations = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            translations.extend([t.strip() for t in batch_translations])
        
        return translations
