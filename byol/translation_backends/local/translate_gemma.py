# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
TranslateGemma Translation Backend.

Google's TranslateGemma 12B model for translation tasks.
Uses ISO 639-1 (2-letter) language codes.

Reference: https://huggingface.co/google/translategemma-12b-it
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
def _load_translate_gemma_model_and_processor(model_name: str, device: str):
    """Load and cache TranslateGemma model and processor."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    
    # Device is already normalized by caller
    device = get_torch_device(device)
    safe_hf_login()
    
    logger.info(f"Loading TranslateGemma model: {model_name} on {device}")
    
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    
    # TranslateGemma uses device_map for device placement
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.bfloat16,
    )
    
    model.eval()
    logger.info(f"TranslateGemma model loaded on {device}")
    return processor, model, device


class TranslateGemmaTranslator(BaseTranslator):
    """
    Translator using Google's TranslateGemma model for text translation.
    
    TranslateGemma is a 12B parameter model designed for translation tasks.
    It uses ISO 639-1 (2-letter) language codes (e.g., "en", "ny", "es", "fr").
    
    Supports 100+ languages. See:
    https://huggingface.co/google/translategemma-12b-it
    
    Example:
        >>> translator = TranslateGemmaTranslator(
        ...     src_lang="en",
        ...     tgt_lang="ny",  # Chichewa
        ... )
        >>> result = translator.translate("Hello, how are you?")
    """

    name = "translategemma"
    translator_type = "local"

    DEFAULT_MODEL_NAME = "google/translategemma-12b-it"
    DEFAULT_DO_SAMPLE = False
    DEFAULT_MAX_NEW_TOKENS = 512

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cuda:0",
        do_sample: bool = DEFAULT_DO_SAMPLE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        **kwargs,
    ):
        """
        Initialize the TranslateGemma translator.
        
        Args:
            src_lang: Source language code (ISO 639-1, e.g., "en")
            tgt_lang: Target language code (ISO 639-1, e.g., "ny")
            model_name: HuggingFace model identifier
            device: Device to use (e.g., "cuda:0", "cuda:1", "cpu", "auto")
            do_sample: Whether to use sampling during generation
            max_new_tokens: Maximum number of new tokens to generate
        """
        if not src_lang:
            raise ValueError("src_lang is required")
        if not tgt_lang:
            raise ValueError("tgt_lang is required")

        super().__init__(src_lang, tgt_lang)

        # Load shared model and processor
        self.processor, self.model, self.device = _load_translate_gemma_model_and_processor(
            model_name, device
        )
        
        self.model_name = model_name
        self.src_lang_code = src_lang
        self.tgt_lang_code = tgt_lang
        
        # Generation parameters
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens

    def translate(self, text: str, **kwargs) -> str:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            **kwargs: Override generation parameters (do_sample, max_new_tokens)
        
        Returns:
            Translated text string
        """
        import torch
        
        # Build the message format for TranslateGemma
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": self.src_lang_code,
                        "target_lang_code": self.tgt_lang_code,
                        "text": text,
                    }
                ],
            }
        ]

        # Process inputs through the chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        
        input_len = len(inputs['input_ids'][0])
        
        # Get generation parameters (allow runtime overrides)
        do_sample = kwargs.get('do_sample', self.do_sample)
        max_new_tokens = kwargs.get('max_new_tokens', self.max_new_tokens)

        # Generate translation
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
            )

        # Extract only the generated tokens (excluding input)
        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        
        return decoded.strip()

    def translate_batch(
        self, 
        texts: List[str], 
        batch_size: int = 8, 
        **kwargs
    ) -> List[str]:
        """
        Translate a batch of texts with proper batched inference.
        
        Args:
            texts: List of texts to translate
            batch_size: Number of texts to process at once
            **kwargs: Additional generation parameters
        
        Returns:
            List of translated texts
        """
        import torch
        
        translations = []
        do_sample = kwargs.get('do_sample', self.do_sample)
        max_new_tokens = kwargs.get('max_new_tokens', self.max_new_tokens)
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Build messages for each text in batch
            batch_messages = []
            for text in batch_texts:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "source_lang_code": self.src_lang_code,
                                "target_lang_code": self.tgt_lang_code,
                                "text": text,
                            }
                        ],
                    }
                ]
                batch_messages.append(messages)
            
            # Process all messages through the chat template
            batch_inputs = [
                self.processor.apply_chat_template(
                    msgs,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                for msgs in batch_messages
            ]
            
            # Pad and batch the inputs
            max_len = max(inp['input_ids'].shape[1] for inp in batch_inputs)
            
            padded_input_ids = []
            padded_attention_mask = []
            input_lengths = []
            
            for inp in batch_inputs:
                seq_len = inp['input_ids'].shape[1]
                input_lengths.append(seq_len)
                
                # Pad on the left (for decoder-only models)
                pad_len = max_len - seq_len
                if pad_len > 0:
                    pad_ids = torch.full((1, pad_len), self.processor.tokenizer.pad_token_id or 0)
                    pad_mask = torch.zeros((1, pad_len), dtype=torch.long)
                    
                    padded_input_ids.append(torch.cat([pad_ids, inp['input_ids']], dim=1))
                    padded_attention_mask.append(torch.cat([pad_mask, inp['attention_mask']], dim=1))
                else:
                    padded_input_ids.append(inp['input_ids'])
                    padded_attention_mask.append(inp['attention_mask'])
            
            # Stack into batch tensors
            batched_inputs = {
                'input_ids': torch.cat(padded_input_ids, dim=0).to(self.model.device, dtype=torch.long),
                'attention_mask': torch.cat(padded_attention_mask, dim=0).to(self.model.device, dtype=torch.long),
            }
            
            # Generate translations for the batch
            with torch.inference_mode():
                generations = self.model.generate(
                    **batched_inputs,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
                )
            
            # Decode each output, excluding the padded input
            for j, gen in enumerate(generations):
                # Remove the padded input portion (max_len tokens)
                output_tokens = gen[max_len:]
                decoded = self.processor.decode(output_tokens, skip_special_tokens=True)
                translations.append(decoded.strip())
        
        return translations
