# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Gemma 3 Translation Backend.

Dedicated translator for Google's Gemma 3 instruction-tuned models.
Supports models: gemma-3-1b-it, gemma-3-4b-it, gemma-3-12b-it, gemma-3-27b-it.

Gemma 3 uses a different chat template format with structured content,
which is why it has its own backend separate from the generic CausalLM.
"""

from functools import lru_cache
from typing import List, Optional

from byol.common.logging import get_logger
from byol.translation_backends.base import BaseTranslator
from byol.translation_backends.config import get_translation_prompt
from byol.translation_backends.local._utils import (
    get_torch_device,
    register_model_cache,
    safe_hf_login,
)

logger = get_logger(__name__)

# Supported Gemma 3 model checkpoints
GEMMA3_MODELS = {
    "gemma-3-1b-it": "google/gemma-3-1b-it",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "gemma-3-27b-it": "google/gemma-3-27b-it",
}

# Default model if none specified
DEFAULT_GEMMA3_MODEL = "gemma-3-4b-it"


@register_model_cache
@lru_cache(maxsize=2)
def _load_gemma3_model_and_tokenizer(model_name: str, device: str):
    """
    Load and cache Gemma 3 model and tokenizer.

    Args:
        model_name: Either a short name (e.g., "gemma-3-4b-it") or
                    full HuggingFace model path (e.g., "google/gemma-3-4b-it").
        device: Target device string (e.g., "cuda:0", "cuda:3", "cpu").

    Returns:
        Tuple of (tokenizer, model).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Device is already normalized by caller
    device = get_torch_device(device)
    safe_hf_login()

    # Resolve short name to full HuggingFace path if needed
    if model_name in GEMMA3_MODELS:
        hf_model_name = GEMMA3_MODELS[model_name]
    elif model_name.startswith("google/gemma-3"):
        hf_model_name = model_name
    else:
        # Assume it's already a full path
        hf_model_name = model_name

    logger.info(f"Loading Gemma 3 model: {hf_model_name} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    # Set pad token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    logger.info(f"Gemma 3 model loaded successfully: {hf_model_name} on {device}")

    return tokenizer, model


class Gemma3Translator(BaseTranslator):
    """
    Translator using Google's Gemma 3 instruction-tuned models.

    Gemma 3 uses a specific chat template format with structured content,
    making it more suitable for translation tasks when properly prompted.

    Supported models:
        - gemma-3-1b-it (fastest, smallest)
        - gemma-3-4b-it (balanced, default)
        - gemma-3-12b-it (larger, more accurate)
        - gemma-3-27b-it (largest, best quality)

    Example:
        >>> translator = Gemma3Translator(
        ...     model_name="gemma-3-4b-it",
        ...     tgt_lang="Spanish",
        ...     device="cuda:0",
        ... )
        >>> result = translator.translate("Hello world")

        >>> # With custom system prompt
        >>> translator = Gemma3Translator(
        ...     model_name="gemma-3-12b-it",
        ...     tgt_lang="French",
        ...     system_prompt="Translate to French. Preserve formal register.",
        ... )
    """

    name = "gemma3"
    translator_type = "local"

    DEFAULT_MAX_NEW_TOKENS = 512
    DEFAULT_TEMPERATURE = 1.0  # Use 1.0 when do_sample=False (ignored anyway)
    DEFAULT_TOP_P = 0.9
    DEFAULT_TOP_K = 50
    DEFAULT_DO_SAMPLE = False  # Greedy decoding by default - more stable
    DEFAULT_REPETITION_PENALTY = 1.0  # No penalty by default

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        model_name: str = DEFAULT_GEMMA3_MODEL,
        device: str = "cuda:0",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        do_sample: bool = DEFAULT_DO_SAMPLE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Gemma 3 Translator.

        Args:
            src_lang: Source language code.
            tgt_lang: Target language code.
            model_name: Gemma 3 model to use. Can be:
                - Short name: "gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"
                - Full path: "google/gemma-3-4b-it"
            device: Device to run on. Accepts:
                - "cuda:0", "cuda:1", etc.
                - "0", "1" (converted to cuda:X)
                - "cpu"
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            do_sample: Whether to use sampling (vs greedy).
            repetition_penalty: Penalty for repeating tokens.
            system_prompt: Custom system prompt. If None, uses default translation prompt.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(src_lang, tgt_lang)

        self.device = get_torch_device(device)
        self.model_name = model_name
        self.tokenizer, self.model = _load_gemma3_model_and_tokenizer(model_name, self.device)

        self.system_prompt = system_prompt or get_translation_prompt(src_lang, tgt_lang, detailed=False)

        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty

        logger.info(f"Initialized Gemma3Translator with model: {model_name}, device: {self.device}")

    def _format_messages(self, text: str) -> list:
        """
        Format input as Gemma 3 chat messages with structured content.

        Gemma 3 expects messages wrapped in an extra list (batch format),
        with content as a list of typed objects.
        """
        # Gemma 3 format: list of conversations, each conversation is a list of messages
        # Each message has content as a list of typed objects
        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                },
            ]
        ]
        return messages

    def translate(self, text: str, **kwargs) -> str:
        """
        Translate a single text.

        Args:
            text: Text to translate.
            **kwargs: Override generation parameters:
                - max_new_tokens: Maximum tokens to generate (default: 512)
                - temperature: Sampling temperature (default: 1.0)
                - top_p: Nucleus sampling parameter (default: 0.9)
                - top_k: Top-k sampling parameter (default: 50)
                - do_sample: Whether to use sampling (default: False)
                - repetition_penalty: Penalty for repeating tokens (default: 1.0)

        Returns:
            Translated text.
        """
        import torch

        messages = self._format_messages(text)

        # Apply chat template with Gemma 3's format
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[-1]

        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": kwargs.get('max_new_tokens', self.max_new_tokens),
        }

        # Handle sampling parameters
        do_sample = kwargs.get('do_sample', self.do_sample)
        if do_sample:
            gen_kwargs["do_sample"] = True
            temperature = kwargs.get('temperature', self.temperature)
            gen_kwargs["temperature"] = temperature if temperature > 0 else 1.0
            gen_kwargs["top_p"] = kwargs.get('top_p', self.top_p)
            gen_kwargs["top_k"] = kwargs.get('top_k', self.top_k)

        # Handle repetition penalty
        repetition_penalty = kwargs.get('repetition_penalty', self.repetition_penalty)
        if repetition_penalty and repetition_penalty > 1.0:
            gen_kwargs["repetition_penalty"] = repetition_penalty

        # Set pad_token_id to suppress warning
        gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the generated part (exclude input prompt)
        generated_tokens = outputs[0][input_length:]
        translation = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return translation.strip()

    def translate_batch(self, texts: List[str], batch_size: int = 4, **kwargs) -> List[str]:
        """
        Translate multiple texts.

        Note: For Gemma 3, we process one at a time due to the specific
        chat template format requirements.

        Args:
            texts: List of texts to translate.
            batch_size: Ignored for Gemma 3 (processes one at a time).
            **kwargs: Override generation parameters.

        Returns:
            List of translated texts.
        """
        translations = []
        for text in texts:
            translation = self.translate(text, **kwargs)
            translations.append(translation)
        return translations
