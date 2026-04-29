# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Marian MT Translation Backend.

Helsinki-NLP's Opus-MT models for bilingual translation.
Models available for many language pairs.

Reference: https://huggingface.co/docs/transformers/model_doc/marian
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
@lru_cache(maxsize=4)
def _load_marian_model_and_tokenizer(model_name: str, device: str):
    """Load and cache Marian model and tokenizer."""
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    
    # Device is already normalized by caller
    device = get_torch_device(device)
    safe_hf_login()
    
    logger.info(f"Loading Marian model: {model_name} on {device}")
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    # Load model directly on target device
    if "cuda" in device:
        model = MarianMTModel.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map=device,
        )
    else:
        model = MarianMTModel.from_pretrained(model_name, dtype=torch.float32)
    
    model.eval()
    logger.info(f"Marian model loaded on {device}")
    return tokenizer, model


class MarianTranslator(BaseTranslator):
    """
    Translator using Helsinki-NLP's Marian MT models.
    
    Uses language-pair specific models from the opus-mt collection.
    Model name format: Helsinki-NLP/opus-mt-{src}-{tgt}
    
    Example:
        >>> # English to German
        >>> translator = MarianTranslator(src_lang="en", tgt_lang="de")
        >>> result = translator.translate("Hello world")
        
        >>> # Or specify model directly
        >>> translator = MarianTranslator(
        ...     model_name="Helsinki-NLP/opus-mt-en-ROMANCE"  # Multilingual
        ... )
    """

    name = "marian"
    translator_type = "local"

    DEFAULT_MAX_LENGTH = 512
    DEFAULT_NUM_BEAMS = 4
    DEFAULT_TOP_K = 50
    DEFAULT_TOP_P = 1.0
    DEFAULT_DO_SAMPLE = True
    DEFAULT_LENGTH_PENALTY = 1.0
    DEFAULT_TEMPERATURE = 0.7

    def __init__(
        self,
        src_lang: str = "en",
        tgt_lang: str = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        model_name: str = None,
        device: str = "cuda:0",
        num_beams: int = DEFAULT_NUM_BEAMS,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        do_sample: bool = DEFAULT_DO_SAMPLE,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs,
    ):
        if tgt_lang is None and model_name is None:
            raise ValueError("Either tgt_lang or model_name must be specified")

        super().__init__(src_lang, tgt_lang)

        # Construct model name from language pair if not provided
        if model_name is None:
            model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

        self.device = get_torch_device(device)
        self.tokenizer, self.model = _load_marian_model_and_tokenizer(model_name, self.device)

        self.max_length = max_length
        self.src_lang_code = src_lang
        self.tgt_lang_code = tgt_lang
        self.model_name = model_name

        # Generation parameters
        self.num_beams = num_beams
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.length_penalty = length_penalty
        self.temperature = temperature

        # Check if multilingual target
        self.is_multilingual_target = (
            hasattr(self.tokenizer, 'supported_language_codes') and 
            self.tokenizer.supported_language_codes is not None
        )
        
        logger.info(f"Initialized MarianTranslator with model: {model_name}")

    def translate(self, text: str, **kwargs) -> str:
        """Translate a single text."""
        import torch
        
        # For multilingual models, prepend language code
        if self.is_multilingual_target and self.tgt_lang_code:
            text = f">>{self.tgt_lang_code}<< {text}"
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
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
        """Translate multiple texts efficiently."""
        import torch
        
        # Prepend language codes for multilingual models
        if self.is_multilingual_target and self.tgt_lang_code:
            texts = [f">>{self.tgt_lang_code}<< {text}" for text in texts]
        
        translations = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
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
