# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Qwen3 Translation Backend.

Dedicated translator for Alibaba's Qwen3 models.
Supports models: Qwen3-4B, Qwen3-8B, Qwen3-14B.

Qwen3 models support a "thinking" mode that can be toggled via the chat template.
For translation tasks, thinking mode is disabled by default for cleaner output.
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

# Supported Qwen3 model checkpoints
QWEN3_MODELS = {
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
}

# Default model if none specified
DEFAULT_QWEN3_MODEL = "qwen3-4b"

# Special token ID for </think> (used when enable_thinking=True)
THINK_END_TOKEN_ID = 151668


@register_model_cache
@lru_cache(maxsize=2)
def _load_qwen3_model_and_tokenizer(model_name: str, device: str):
    """
    Load and cache Qwen3 model and tokenizer.

    Args:
        model_name: Either a short name (e.g., "qwen3-4b") or
                    full HuggingFace model path (e.g., "Qwen/Qwen3-4B").
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
    if model_name.lower() in QWEN3_MODELS:
        hf_model_name = QWEN3_MODELS[model_name.lower()]
    elif model_name.startswith("Qwen/Qwen3"):
        hf_model_name = model_name
    else:
        # Assume it's already a full path
        hf_model_name = model_name

    logger.info(f"Loading Qwen3 model: {hf_model_name} on {device}")

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
        dtype=torch.bfloat16,
        device_map=device_map,
    ).eval()

    logger.info(f"Qwen3 model loaded successfully: {hf_model_name} on {device}")

    return tokenizer, model


class Qwen3Translator(BaseTranslator):
    """
    Translator using Alibaba's Qwen3 models.

    Qwen3 models have a "thinking" mode that can be toggled. For translation,
    thinking mode is disabled by default to produce cleaner output.

    Supported models:
        - qwen3-4b (smallest, fastest)
        - qwen3-8b (balanced)
        - qwen3-14b (largest, best quality)

    Example:
        >>> translator = Qwen3Translator(
        ...     model_name="qwen3-4b",
        ...     tgt_lang="Spanish",
        ...     device="cuda:0",
        ... )
        >>> result = translator.translate("Hello world")

        >>> # With thinking mode enabled (for reasoning)
        >>> translator = Qwen3Translator(
        ...     model_name="qwen3-14b",
        ...     tgt_lang="French",
        ...     enable_thinking=True,
        ... )
    """

    name = "qwen3"
    translator_type = "local"

    DEFAULT_MAX_NEW_TOKENS = 32768
    DEFAULT_ENABLE_THINKING = False
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_P = 0.8
    DEFAULT_TOP_K = 20
    DEFAULT_MIN_P = 0.0
    DEFAULT_DO_SAMPLE = False  # Greedy decoding by default - more stable
    DEFAULT_REPETITION_PENALTY = 1.0  # No penalty by default

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        model_name: str = DEFAULT_QWEN3_MODEL,
        device: str = "cuda:0",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        enable_thinking: bool = DEFAULT_ENABLE_THINKING,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        min_p: float = DEFAULT_MIN_P,
        do_sample: bool = DEFAULT_DO_SAMPLE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Qwen3 Translator.

        Args:
            src_lang: Source language code.
            tgt_lang: Target language code.
            model_name: Qwen3 model to use. Can be:
                - Short name: "qwen3-4b", "qwen3-8b", "qwen3-14b"
                - Full path: "Qwen/Qwen3-4B"
            device: Device to run on. Accepts:
                - "cuda:0", "cuda:1", etc.
                - "0", "1" (converted to cuda:X)
                - "cpu"
            max_new_tokens: Maximum tokens to generate.
            enable_thinking: Whether to enable Qwen3's thinking mode.
                            Default is False for cleaner translation output.
            system_prompt: Custom system prompt. If None, uses default translation prompt.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(src_lang, tgt_lang)

        self.device = get_torch_device(device)
        self.model_name = model_name
        self.tokenizer, self.model = _load_qwen3_model_and_tokenizer(model_name, self.device)

        self.system_prompt = system_prompt or get_translation_prompt(src_lang, tgt_lang, detailed=False)
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty

        logger.info(f"Initialized Qwen3Translator with model: {model_name}, device: {self.device}, thinking: {enable_thinking}")

    def _format_messages(self, text: str) -> list:
        """
        Format input as Qwen3 chat messages.

        Qwen3 uses simple message format with system and user roles.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]
        return messages

    def _parse_thinking_output(self, output_ids: list) -> str:
        """
        Parse output tokens, removing thinking content if present.

        When enable_thinking=True, Qwen3 outputs thinking content before
        the actual response, delimited by </think> token (ID: 151668).

        Args:
            output_ids: List of generated token IDs.

        Returns:
            Decoded text with thinking content removed.
        """
        try:
            # Find </think> token and skip everything before it
            index = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
        except ValueError:
            # No thinking token found, use full output
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)
        return content.strip("\n").strip()

    def translate(self, text: str, **kwargs) -> str:
        """
        Translate a single text.

        Args:
            text: Text to translate.
            **kwargs: Override generation parameters:
                - max_new_tokens: Maximum tokens to generate (default: 512)
                - enable_thinking: Override thinking mode setting

        Returns:
            Translated text.
        """
        import torch

        messages = self._format_messages(text)

        # Apply chat template
        enable_thinking = kwargs.get('enable_thinking', self.enable_thinking)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        # Tokenize
        model_inputs = self.tokenizer(
            [prompt_text],
            return_tensors="pt"
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
                gen_kwargs["top_k"] = kwargs.get('top_k', self.top_k)
                gen_kwargs["min_p"] = kwargs.get('min_p', self.min_p)

            # Add repetition penalty if > 1.0
            repetition_penalty = kwargs.get('repetition_penalty', self.repetition_penalty)
            if repetition_penalty > 1.0:
                gen_kwargs["repetition_penalty"] = repetition_penalty

            outputs = self.model.generate(**model_inputs, **gen_kwargs)

        # Get only the generated tokens (exclude input)
        output_ids = outputs[0][input_length:].tolist()

        # Parse output (handles thinking mode content removal)
        translation = self._parse_thinking_output(output_ids)

        return translation

    def translate_batch(self, texts: List[str], batch_size: int = 4, **kwargs) -> List[str]:
        """
        Translate multiple texts.

        Note: For Qwen3, we process one at a time due to the specific
        chat template and thinking mode requirements.

        Args:
            texts: List of texts to translate.
            batch_size: Ignored for Qwen3 (processes one at a time).
            **kwargs: Override generation parameters.

        Returns:
            List of translated texts.
        """
        translations = []
        for text in texts:
            translation = self.translate(text, **kwargs)
            translations.append(translation)
        return translations
