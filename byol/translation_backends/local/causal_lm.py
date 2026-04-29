# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
HuggingFace CausalLM Translation Backend.

Generic translator for any HuggingFace causal language model.
Useful for LLMs like Llama, Mistral, Qwen, etc.
"""

from functools import lru_cache
from typing import List, Optional

from byol.common.logging import get_logger
from byol.translation_backends.base import BaseTranslator
from byol.translation_backends.config import get_translation_prompt
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
def _load_causal_lm_model_and_tokenizer(model_name: str, device: str):
    """Load and cache CausalLM model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Device is already normalized by caller
    device = get_torch_device(device)
    safe_hf_login()
    
    logger.info(f"Loading CausalLM model: {model_name} on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=get_torch_dtype(device),
        device_map="auto" if "cuda" in device else None,
    )
    
    model.eval()
    return tokenizer, model


class HuggingFaceCausalLM(BaseTranslator):
    """
    Generic translator using any HuggingFace CausalLM model.
    
    Supports chat template formatting for instruction-tuned models.
    
    Example:
        >>> translator = HuggingFaceCausalLM(
        ...     model_name="meta-llama/Llama-3.1-8B-Instruct",
        ...     tgt_lang="Spanish",
        ... )
        >>> result = translator.translate("Hello world")
        
        >>> # With custom prompt
        >>> translator = HuggingFaceCausalLM(
        ...     model_name="Qwen/Qwen2-7B-Instruct",
        ...     tgt_lang="French",
        ...     system_prompt="Translate to French, preserving formal register."
        ... )
    """

    name = "hf-causallm"
    translator_type = "local"

    DEFAULT_MAX_NEW_TOKENS = 512
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_P = 0.9
    DEFAULT_TOP_K = 50
    DEFAULT_DO_SAMPLE = True
    DEFAULT_REPETITION_PENALTY = 1.1

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        model_name: str = None,
        device: str = "cuda:0",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        do_sample: bool = DEFAULT_DO_SAMPLE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        system_prompt: Optional[str] = None,
        use_chat_template: bool = True,
        **kwargs,
    ):
        if model_name is None:
            raise ValueError("model_name is required for HuggingFaceCausalLM")

        super().__init__(src_lang, tgt_lang)

        self.device = get_torch_device(device)
        self.tokenizer, self.model = _load_causal_lm_model_and_tokenizer(model_name, self.device)

        self.model_name = model_name
        self.system_prompt = system_prompt or get_translation_prompt(src_lang, tgt_lang)
        self.use_chat_template = use_chat_template

        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        
        logger.info(f"Initialized HuggingFaceCausalLM with model: {model_name}")

    def _format_prompt(self, text: str) -> str:
        """Format the input with chat template or simple prompt."""
        if self.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, using simple format")
        
        # Fallback to simple format
        return f"{self.system_prompt}\n\nText: {text}\n\nTranslation:"

    def translate(self, text: str, **kwargs) -> str:
        """Translate a single text."""
        import torch
        
        prompt = self._format_prompt(text)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        
        if self.device != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        input_length = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_new_tokens', self.max_new_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                top_p=kwargs.get('top_p', self.top_p),
                top_k=kwargs.get('top_k', self.top_k),
                do_sample=kwargs.get('do_sample', self.do_sample),
                repetition_penalty=kwargs.get('repetition_penalty', self.repetition_penalty),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][input_length:]
        translation = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return translation.strip()

    def translate_batch(self, texts: List[str], batch_size: int = 4, **kwargs) -> List[str]:
        """Translate multiple texts."""
        import torch
        
        translations = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            prompts = [self._format_prompt(text) for text in batch_texts]
            
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_new_tokens', self.max_new_tokens),
                    temperature=kwargs.get('temperature', self.temperature),
                    top_p=kwargs.get('top_p', self.top_p),
                    top_k=kwargs.get('top_k', self.top_k),
                    do_sample=kwargs.get('do_sample', self.do_sample),
                    repetition_penalty=kwargs.get('repetition_penalty', self.repetition_penalty),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode each output, removing the input prompt
            for j, output in enumerate(outputs):
                input_length = inputs['input_ids'][j].ne(self.tokenizer.pad_token_id).sum()
                generated_tokens = output[input_length:]
                translation = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                translations.append(translation.strip())
        
        return translations
