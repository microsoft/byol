# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
MADLAD (Massively Multilingual Domain Adapted Language Translation) Backend.

Google's multilingual translation model supporting 400+ languages.
Uses language code prefix format: <2xx> where xx is the language code.
"""

from functools import lru_cache
from typing import List

from byol.common.logging import get_logger
from byol.translation_backends.base import BaseTranslator
from byol.translation_backends.local._utils import (
    safe_hf_login, 
    get_torch_device,
    normalize_device,
    register_model_cache,
)

logger = get_logger(__name__)


@register_model_cache
@lru_cache(maxsize=2)
def _load_madlad_model_and_tokenizer(model_name: str, device: str):
    """Load and cache MADLAD model and tokenizer."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    
    # Device is already normalized by caller
    device = get_torch_device(device)
    safe_hf_login()
    
    logger.info(f"Loading MADLAD model: {model_name} on {device}")
    
    dtype = torch.bfloat16 if "cuda" in device else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model directly on target device
    if "cuda" in device:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=dtype,
        )
    
    model.eval()
    logger.info(f"MADLAD model loaded on {device} with dtype {dtype}")
    return tokenizer, model


class MadladTranslator(BaseTranslator):
    """
    Translator using Google's MADLAD model.
    
    Supports 400+ languages using prefix format: <2xx> source_text
    
    Example:
        >>> translator = MadladTranslator(tgt_lang="sw")  # Swahili
        >>> result = translator.translate("Hello world")
    """

    name = "madlad"
    translator_type = "local"

    DEFAULT_MAX_LENGTH = 256
    DEFAULT_MODEL_NAME = "jbochi/madlad400-3b-mt"
    DEFAULT_NUM_BEAMS = 4
    DEFAULT_TOP_K = 50
    DEFAULT_TOP_P = 1.0
    DEFAULT_DO_SAMPLE = True
    DEFAULT_LENGTH_PENALTY = 1.0
    DEFAULT_TEMPERATURE = 0.7

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

        super().__init__(src_lang, tgt_lang)

        self.device = get_torch_device(device)
        self.tokenizer, self.model = _load_madlad_model_and_tokenizer(model_name, self.device)

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
        
        logger.info(f"Initialized MadladTranslator: {src_lang} -> {tgt_lang}")

    def translate(self, text: str, **kwargs) -> str:
        """Translate a single text using MADLAD prefix format."""
        import torch
        
        # MADLAD uses prefix format: <2tgt_lang> source_text
        prompt = f"<2{self.tgt_lang_code}> {text}"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        if self.device != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
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
        """Translate multiple texts."""
        import torch
        
        translations = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            prompts = [f"<2{self.tgt_lang_code}> {text}" for text in batch_texts]
            
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
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
