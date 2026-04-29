# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
LLM Judge Evaluation Utilities

Professional toolkit for LLM-as-a-Judge evaluation with support for:
- HuggingFace models with Flash Attention 2
- Azure OpenAI and GPT-OSS-120B endpoints
- Multi-language evaluation with language detection
- Configurable dataset loading and prompt formatting
"""

import gc
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from datasets import load_dataset as hf_load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _resolve_model_path(model_path: str) -> str:
    """Resolve model path to a local directory or HuggingFace repo ID.

    Resolution order:
        1. If *model_path* is already an existing absolute directory → use it.
        2. If it exists relative to cwd → use the absolute version.
        3. If ``BYOL_WEIGHTS_DIR`` env var is set, try
           ``$BYOL_WEIGHTS_DIR/<model_path>`` (and its basename variant).
        4. Fall through unchanged (assumed to be a HuggingFace repo ID such as
           ``google/gemma-3-4b-it``).

    Set the environment variable before running:
        export BYOL_WEIGHTS_DIR=/path/to/your/weights
    """
    # 1. Already an absolute, existing directory
    if os.path.isabs(model_path) and os.path.isdir(model_path):
        return model_path

    # 2. Exists relative to cwd
    if os.path.isdir(model_path):
        return os.path.abspath(model_path)

    # 3. Try BYOL_WEIGHTS_DIR
    weights_dir = os.environ.get("BYOL_WEIGHTS_DIR", "")
    if weights_dir:
        # Try joining directly (handles both "byol_nya_4b_M" and "sub/byol_nya_4b_M")
        candidate = os.path.join(weights_dir, model_path)
        if os.path.isdir(candidate):
            return candidate
        # Try basename only (in case model_path contains old prefix like "weights/...")
        candidate = os.path.join(weights_dir, os.path.basename(model_path))
        if os.path.isdir(candidate):
            return candidate

    # 4. Assume HuggingFace repo ID
    return model_path


def load_huggingface_model(
    model_path: str,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    use_flash_attention_2: bool = False,
    trust_remote_code: bool = True,
    use_cache: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load HuggingFace model and tokenizer.

    For local models, set the ``BYOL_WEIGHTS_DIR`` environment variable to the
    directory that contains your model folders.  The *model_path* can then be
    just the folder name (e.g. ``byol_nya_4b_M``).

    Args:
        model_path: Local path or HuggingFace repo ID (e.g. google/gemma-3-4b-it)
        device: Target device (e.g., "cuda:0", "cpu")
        dtype: Precision type ("bfloat16", "float16", "float32", "auto")
        use_flash_attention_2: Enable Flash Attention 2 optimization
        trust_remote_code: Allow execution of custom model code
        use_cache: Enable KV cache for faster generation
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_path = _resolve_model_path(model_path)
    logger.info(f"Loading {model_path} on {device} ({dtype})")
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto"
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=trust_remote_code
    )
    
    # Configure padding for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Left padding for decoder-only models
    
    model_kwargs = {
        "trust_remote_code": trust_remote_code
    }
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    if use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        **model_kwargs
    ).to(device)
    
    # Set use_cache after model initialization
    if hasattr(model, 'config'):
        model.config.use_cache = use_cache
    
    logger.info("✅ Model loaded successfully")
    return model, tokenizer


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    generation_params: Dict[str, Any],
    use_chat_template: bool = False
) -> str:
    """
    Generate response from HuggingFace model.
    
    Args:
        model: Loaded HuggingFace model
        tokenizer: Corresponding tokenizer
        prompt: Input prompt text
        generation_params: Generation config (max_tokens, temperature, etc.)
        use_chat_template: Apply chat template formatting
        
    Returns:
        Generated text response
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    ) if use_chat_template else prompt
    
    model_inputs = tokenizer(
        [text], 
        return_tensors="pt", 
        add_special_tokens=False
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, **generation_params)
    
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def batch_generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    generation_params: Dict[str, Any],
    use_chat_template: bool = False
) -> List[str]:
    """
    Generate responses for multiple prompts in batch.
    
    Args:
        model: Loaded HuggingFace model
        tokenizer: Corresponding tokenizer
        prompts: List of input prompts
        generation_params: Generation config (max_tokens, temperature, etc.)
        use_chat_template: Apply chat template formatting
        
    Returns:
        List of generated text responses
    """
    # Ensure model is in eval mode (no gradient tracking)
    model.eval()
    
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True
        ) for p in prompts
    ] if use_chat_template else prompts
    
    model_inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        add_special_tokens=False
    ).to(model.device)
    
    # Store the original input_ids length for each sample (including padding)
    input_ids_lengths = [len(ids) for ids in model_inputs.input_ids]
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs, 
            **generation_params,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True  # Use cache during generation for efficiency
        )
    
    responses = []
    for i, (gen_ids, orig_len) in enumerate(zip(generated_ids, input_ids_lengths)):
        # The generated_ids includes the full input sequence + newly generated tokens
        # We need to extract only the newly generated tokens (after the input)
        # orig_len is the length of input (padding + actual input tokens)
        output_ids = gen_ids[orig_len:]
        responses.append(tokenizer.decode(output_ids, skip_special_tokens=True))
    
    # Critical: Clear model's internal cache and states
    if hasattr(model, 'reset_cache'):
        model.reset_cache()
    # Clear past_key_values if they exist
    for module in model.modules():
        if hasattr(module, 'past_key_values'):
            module.past_key_values = None
    
    # CRITICAL: Synchronize CUDA before cleanup (ensures all operations complete)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Aggressive memory cleanup after batch generation
    del model_inputs, generated_ids, output_ids, input_ids_lengths, texts
    
    if torch.cuda.is_available():
        # Empty cache multiple times with synchronization
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Force garbage collection to free Python objects
    gc.collect()
    
    return responses


def cleanup_model(model: AutoModelForCausalLM) -> None:
    """
    Free model from GPU memory.
    
    Args:
        model: Model to clean up
    """
    del model
    torch.cuda.empty_cache()
    logger.info("✅ Model cleaned up")


def find_model_config(model_name: str, full_config: Dict) -> Optional[Dict]:
    """
    Find and resolve model configuration with family inheritance.
    
    Args:
        model_name: Model identifier (config key or display name)
        full_config: Full configuration dict with models and families
        
    Returns:
        Resolved model config or None if not found
    """
    models = full_config.get("models", {})
    families = full_config.get("model_families", {})
    
    # Try exact key match first
    config_key = model_name
    if model_name not in models:
        # Try matching by display name (case-insensitive)
        model_name_lower = model_name.lower()
        for key, cfg in models.items():
            if cfg.get("name", "").lower() == model_name_lower:
                config_key = key
                break
        else:
            logger.error(f"❌ Model '{model_name}' not found in configuration")
            return None
    
    config = models[config_key]
    family = families.get(config.get("family"), {})
    
    # Resolve type
    model_type = config.get("type") or family.get("type")
    if not model_type:
        logger.error(f"❌ Model '{config_key}' has no type defined")
        return None
    
    # Build resolved config with family inheritance
    result = {
        "id": config_key,
        "name": config.get("name", config_key),
        "type": model_type,
        "generation": {**family.get("generation", {}), **config.get("generation", {})}
    }
    
    # Inherit use_chat_template and batch_size from family or model config
    if model_type == "huggingface":
        result["use_chat_template"] = config.get("use_chat_template", family.get("use_chat_template", False))
        result["batch_size"] = config.get("batch_size", family.get("batch_size", 1))
    
    # Add path based on type
    path_key = "repo" if model_type == "huggingface" else "endpoint"
    path_value = config.get(path_key)
    if not path_value:
        logger.error(f"❌ Model '{config_key}' missing {path_key}")
        return None
    
    result[path_key] = result["path"] = path_value
    return result


def _extract_model_info(comparison: Dict, key: str, default_device: str) -> Tuple[Optional[str], str]:
    """
    Extract model name and device from comparison config.
    
    Args:
        comparison: Comparison configuration dict
        key: Key to extract (model_a or model_b)
        default_device: Default device if not specified
        
    Returns:
        Tuple of (model_name, device)
    """
    model_info = comparison.get(key)
    if isinstance(model_info, str):
        return model_info, default_device
    if isinstance(model_info, dict):
        return model_info.get("name"), model_info.get("device", default_device)
    return None, default_device


def get_comparison_models(model_config: Dict) -> Tuple[Dict, Dict]:
    """
    Extract two models to compare from configuration.
    
    Args:
        model_config: Full model configuration
        
    Returns:
        Tuple of (model_a_config, model_b_config)
        
    Raises:
        ValueError: If models not specified or not found
    """
    comparison = model_config.get("comparison", {})
    if not comparison:
        raise ValueError("Configuration must include 'comparison' section")
    
    model_a_name, model_a_device = _extract_model_info(comparison, "model_a", "cuda:0")
    model_b_name, model_b_device = _extract_model_info(comparison, "model_b", "cuda:1")
    
    if not model_a_name or not model_b_name:
        raise ValueError("Comparison must specify both 'model_a' and 'model_b'")
    
    model_a_config = find_model_config(model_a_name, model_config)
    model_b_config = find_model_config(model_b_name, model_config)
    
    if not model_a_config or not model_b_config:
        raise ValueError(f"Models not found in config: A='{model_a_name}', B='{model_b_name}'")
    
    model_a_config['device'] = model_a_device
    model_b_config['device'] = model_b_device
    
    logger.info(f"✅ Model A: {model_a_config['name']} on {model_a_device}")
    logger.info(f"✅ Model B: {model_b_config['name']} on {model_b_device}")
    
    return model_a_config, model_b_config


def get_judge_config(model_config: Dict) -> Dict:
    """
    Extract judge model configuration.
    
    Args:
        model_config: Full model configuration
        
    Returns:
        Judge model configuration dict
        
    Raises:
        ValueError: If judge model not specified
    """
    judge_config = model_config.get("judge_model", {})
    if not judge_config.get("model"):
        raise ValueError("Configuration must specify 'judge_model.model'")
    
    logger.info(f"✅ Judge: {judge_config['model']}")
    return judge_config


def get_evaluation_config(model_config: Dict) -> Dict:
    """
    Extract evaluation settings from configuration.
    
    Args:
        model_config: Full model configuration
        
    Returns:
        Evaluation configuration dict (no defaults applied)
    """
    eval_config = model_config.get("evaluation", {})
    
    if "enable_language_detection" in eval_config:
        status = "enabled" if eval_config["enable_language_detection"] else "disabled"
        logger.info(f"Language detection: {status}")
    
    return eval_config


def initialize_endpoint_clients() -> Dict[str, Any]:
    """
    Initialize API endpoint clients.
    
    Returns:
        Dict with initialized clients for Azure, DeepSeek, and GPT-OSS-120B
    """
    import os
    import sys
    
    # API folder is at BYOL/api (3 levels up from llm_judge: llm_judge -> byol_eval -> eval -> BYOL)
    byol_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    api_dir = os.path.join(byol_root, 'api')
    if api_dir not in sys.path:
        sys.path.insert(0, api_dir)
    
    empty_clients = {
        "azure_client": None,
        "azure_endpoint": None,
        "deepseek_client": None,
        "deepseek_endpoint": None,
        "gpt_oss_120b_config": None
    }
    
    try:
        from get_azure_api import initialize_client, setup_gpt_oss_120b_client
        
        azure_client, azure_endpoint = initialize_client(
            endpoint_type="eu2", 
            fallback_endpoint=True
        )
        deepseek_client, deepseek_endpoint = initialize_client(
            endpoint_type="deepseek", 
            fallback_endpoint=False
        )
        gpt_oss_120b_config = setup_gpt_oss_120b_client()
        
        # Log initialization status
        logger.info(f"{'✅' if azure_client else '❌'} Azure: {azure_endpoint or 'Failed'}")
        logger.info(f"{'✅' if deepseek_client else '⚠️'} DeepSeek: {deepseek_endpoint or 'Not available'}")
        logger.info(f"{'✅' if gpt_oss_120b_config else '❌'} GPT-OSS-120B: {'Ready' if gpt_oss_120b_config else 'Failed'}")
        
        return {
            "azure_client": azure_client,
            "azure_endpoint": azure_endpoint,
            "deepseek_client": deepseek_client,
            "deepseek_endpoint": deepseek_endpoint,
            "gpt_oss_120b_config": gpt_oss_120b_config
        }
        
    except Exception as e:
        logger.error(f"❌ Client initialization failed: {e}")
        return empty_clients


def _process_answer_field(sample: Dict) -> None:
    """
    Extract answer text from complex answer structures.
    
    Handles different answer formats:
    - {'answers': {'text': ['answer']}}
    - {'answers': {'text': 'answer'}}
    - {'answers': 'answer'}
    
    Args:
        sample: Dataset sample dict (modified in-place)
    """
    if 'answers' not in sample:
        return
    
    answers = sample['answers']
    
    if isinstance(answers, dict):
        if 'text' in answers:
            text_field = answers['text']
            sample['answer_text'] = text_field[0] if isinstance(text_field, list) and text_field else str(text_field)
        else:
            sample['answer_text'] = str(answers)
    else:
        sample['answer_text'] = str(answers)


def load_dataset_samples(dataset_config: Dict) -> List[Dict]:
    """
    Load dataset samples from HuggingFace.
    
    Supports two loading modes:
    - Standard: Single dataset with optional language filtering
    - Multi-language: Load from multiple language subsets
    
    Args:
        dataset_config: Dataset configuration dict
        
    Returns:
        List of dataset samples
    """
    dataset_name = dataset_config.get('name', dataset_config.get('dataset_name', ''))
    if not dataset_name:
        logger.error("❌ Dataset configuration missing 'name' field")
        return []
    
    try:
        # Standard dataset (Aya, Multi-wiki-qa)
        if 'dataset_name' not in dataset_config:
            logger.info(f"Loading {dataset_name}...")
            
            dataset = hf_load_dataset(
                dataset_name,
                dataset_config.get('subset'),
                split=dataset_config.get('split', 'test')
            )
            
            # Apply language filter if specified
            if dataset_config.get('languages_filter'):
                lang_filter = dataset_config['languages_filter']
                dataset = dataset.filter(lambda x: x.get('language') in lang_filter)
            
            # Apply sample limit
            limit = dataset_config.get('limit')
            samples = list(dataset)[:limit] if limit else list(dataset)
            
            # Process answer fields and add language metadata
            for sample in samples:
                _process_answer_field(sample)
                if 'language' not in sample and 'subset' in dataset_config:
                    sample['language'] = dataset_config['subset']
        
        # Multi-language dataset (XLSum-style)
        else:
            logger.info(f"Loading multi-language {dataset_name}...")
            
            samples = []
            languages = dataset_config.get('languages', ['english'])
            num_samples = dataset_config.get('num_samples', 50)
            
            for language in languages:
                lang_dataset = hf_load_dataset(
                    dataset_name,
                    language,
                    split=dataset_config.get('split', 'test')
                )
                lang_samples = list(lang_dataset)[:num_samples]
                
                # Add language metadata
                for sample in lang_samples:
                    sample['language'] = language
                
                samples.extend(lang_samples)
            
            # Apply global limit
            if dataset_config.get('limit'):
                samples = samples[:dataset_config['limit']]
        
        logger.info(f"✅ Loaded {len(samples)} samples from {dataset_name}")
        return samples
        
    except Exception as e:
        logger.error(f"❌ Failed to load {dataset_name}: {e}")
        return []


# API Endpoint Constants
AZURE_MODELS = {
    "gpt-35-turbo", "gpt-35-turbo-16k", 
    "gpt-4-turbo", "gpt-4-vision", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o",
    "gpt-5", "gpt-5-chat", "gpt-5-mini", "gpt-5-nano",
    "o3",
    "text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"
}

GPT5_MODELS = {"gpt-5", "gpt-5-chat", "gpt-5-mini", "gpt-5-nano"}
GPT_OSS_VARIANTS = ("gpt-oss-120b", "openai/gpt-oss-120b", "gpt-oss-120")


def _call_azure_endpoint(endpoint: str, messages: List[Dict], gen_params: Dict, client: Any) -> str:
    """
    Call Azure OpenAI API endpoint.
    
    Args:
        endpoint: Model endpoint name
        messages: Chat messages list
        gen_params: Generation parameters from config
        client: Azure OpenAI client
        
    Returns:
        Generated response text
    """
    api_kwargs = {'model': endpoint, 'messages': messages}
    
    # GPT-5 models have restricted parameters - only max_completion_tokens supported
    # Other models support temperature and top_p
    if endpoint not in GPT5_MODELS:
        # Only add parameters if explicitly provided in config
        if 'temperature' in gen_params:
            api_kwargs['temperature'] = float(gen_params['temperature'])
        
        if 'top_p' in gen_params:
            api_kwargs['top_p'] = float(gen_params['top_p'])
    
    if 'max_new_tokens' in gen_params:
        max_tokens = int(gen_params['max_new_tokens'])
        if max_tokens > 0:
            token_param = 'max_completion_tokens' if endpoint in GPT5_MODELS else 'max_tokens'
            api_kwargs[token_param] = max_tokens
    
    response = client.chat.completions.create(**api_kwargs)
    
    # Debug: Log response structure for GPT-5
    if endpoint in GPT5_MODELS:
        logger.info(f"GPT-5 response structure: choices={len(response.choices) if response.choices else 0}")
        if response.choices:
            choice = response.choices[0]
            logger.info(f"Choice 0: finish_reason={getattr(choice, 'finish_reason', 'N/A')}")
            message = choice.message
            logger.info(f"Message content: '{message.content}'")
            logger.info(f"Message refusal: '{message.refusal}'")
            logger.info(f"Message role: '{message.role}'")
            # Check all text fields
            for field in ['content', 'refusal', 'function_call', 'tool_calls']:
                if hasattr(message, field):
                    val = getattr(message, field)
                    if val:
                        logger.info(f"Found non-empty {field}: {str(val)[:200]}")
    
    if response.choices:
        choice = response.choices[0]
        message = choice.message
        
        # GPT-5 may use reasoning_content instead of content
        content = message.content
        if not content and hasattr(message, 'reasoning_content'):
            content = message.reasoning_content
            logger.info("Using reasoning_content from GPT-5")
        
        return content.strip() if content else "No response generated"
    
    return "No response generated"


def _call_gpt_oss_endpoint(messages: List[Dict], gen_params: Dict, config: Dict) -> str:
    """
    Call GPT-OSS-120B endpoint with exponential backoff retry.
    
    Args:
        messages: Chat messages list
        gen_params: Generation parameters from config
        config: GPT-OSS-120B configuration with url, headers, model
        
    Returns:
        Generated response text or error message
    """
    payload = {"messages": messages, "model": config["model"]}
    
    # Only add parameters if explicitly provided in config
    if 'temperature' in gen_params:
        payload["temperature"] = float(gen_params['temperature'])
    
    if 'do_sample' in gen_params:
        payload["do_sample"] = gen_params['do_sample']
    
    if 'max_new_tokens' in gen_params:
        max_tokens = int(gen_params['max_new_tokens'])
        if max_tokens > 0:
            payload["max_completion_tokens"] = max_tokens
    
    # Retry with exponential backoff: 5s, 10s, 20s, 40s, 80s (max 5 attempts)
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            response = requests.post(
                config["url"],
                headers=config["headers"],
                data=json.dumps(payload),
                timeout=180
            )
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    
                    # Validate response structure
                    if 'choices' not in response_data:
                        logger.error(f"❌ GPT-OSS-120B response missing 'choices' field")
                        logger.error(f"Response keys: {list(response_data.keys())}")
                        logger.debug(f"Full response: {response.text[:500]}")
                        return "Error: Invalid response structure (missing 'choices')"
                    
                    if not response_data['choices']:
                        logger.error(f"❌ GPT-OSS-120B returned empty 'choices' list")
                        return "Error: Empty response choices"
                    
                    choice = response_data['choices'][0]
                    if 'message' not in choice:
                        logger.error(f"❌ GPT-OSS-120B choice missing 'message' field")
                        logger.error(f"Choice keys: {list(choice.keys())}")
                        return "Error: Invalid response structure (missing 'message')"
                    
                    message = choice['message']
                    
                    # Prioritize 'content' over 'reasoning_content'
                    # GPT-OSS-120B may return both fields
                    content = message.get('content')
                    
                    if not content:
                        # Fallback to reasoning_content if content not present
                        content = message.get('reasoning_content')
                        logger.debug(f"Using 'reasoning_content' as fallback (no 'content' field)")
                    
                    if not content:
                        logger.error(f"❌ GPT-OSS-120B message missing both 'content' and 'reasoning_content' fields")
                        logger.error(f"Message keys: {list(message.keys())}")
                        logger.debug(f"Full message: {message}")
                        return "Error: Invalid response structure (no content field found)"
                    
                    return content.strip() if content else "No response generated"
                    
                except (KeyError, IndexError, TypeError) as e:
                    logger.error(f"❌ GPT-OSS-120B parse error: {e}")
                    logger.debug(f"Response: {response.text[:500]}")
                    # Don't retry on parse errors - the API format is wrong
                    return f"Error: Response parsing failed ({e})"
            
            # Retry on non-200 status codes
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            logger.warning(f"⚠️ GPT-OSS-120B request failed: {error_msg}")
            raise requests.RequestException(error_msg)
            
        except (requests.Timeout, requests.ConnectionError) as e:
            # Retry on network errors
            if attempt < max_attempts - 1:
                delay = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s, 80s
                logger.warning(f"⚠️ GPT-OSS-120B retry {attempt + 1}/{max_attempts} in {delay}s: {e}")
                time.sleep(delay)
            else:
                logger.error(f"❌ GPT-OSS-120B failed after {max_attempts} attempts: {e}")
                return f"Error: API call failed after {max_attempts} attempts"
        
        except requests.RequestException as e:
            # Retry on HTTP errors (non-200 status)
            if attempt < max_attempts - 1:
                delay = 5 * (2 ** attempt)
                logger.warning(f"⚠️ GPT-OSS-120B retry {attempt + 1}/{max_attempts} in {delay}s: {e}")
                time.sleep(delay)
            else:
                logger.error(f"❌ GPT-OSS-120B failed after {max_attempts} attempts: {e}")
                return f"Error: API call failed after {max_attempts} attempts"
    
    return "Error: API call failed"


def generate_endpoint_response(
    prompt: str,
    model_config: Dict,
    clients: Dict,
    system_prompt: Optional[str] = None
) -> str:
    """
    Generate response using API endpoint.
    
    Supports:
    - Azure OpenAI (GPT-3.5, GPT-4, GPT-5 series)
    - GPT-OSS-120B
    
    Args:
        prompt: User prompt text
        model_config: Model configuration with model/endpoint and generation params
        clients: Initialized API clients dict (from initialize_endpoint_clients)
        system_prompt: Optional system prompt (if not provided, only user message sent)
        
    Returns:
        Generated response text or error message
    """
    # Get endpoint from model_config - try endpoint, path, or model field
    endpoint = model_config.get('endpoint') or model_config.get('path') or model_config.get('model')
    if not endpoint:
        logger.error("❌ No endpoint/model specified in model config")
        return "Error: No endpoint specified"
    
    # Build messages - only include system prompt if provided
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    gen_params = model_config.get('generation', {})
    
    try:
        # Azure OpenAI
        if endpoint in AZURE_MODELS:
            client = clients.get("azure_client")
            if not client:
                return "Error: Azure client not initialized"
            return _call_azure_endpoint(endpoint, messages, gen_params, client)
        
        # GPT-OSS-120B
        elif endpoint in GPT_OSS_VARIANTS:
            config = clients.get("gpt_oss_120b_config")
            if not config:
                return "Error: GPT-OSS-120B not initialized"
            return _call_gpt_oss_endpoint(messages, gen_params, config)
        
        # Unsupported endpoint
        else:
            logger.error(f"❌ Unsupported endpoint: {endpoint}")
            return f"Error: Unsupported endpoint: {endpoint}"
            
    except Exception as e:
        logger.error(f"❌ Endpoint generation failed: {e}")
        return f"Error: {str(e)[:100]}"


# ISO 639 Language Code Mapping
LANGUAGE_CODE_MAP = {
    'en': 'English', 'eng': 'English',
    'mi': 'Maori', 'mri': 'Maori', 'mao': 'Maori',
    'ny': 'Chichewa', 'nya': 'Chichewa',
    'fr': 'French', 'fra': 'French',
    'es': 'Spanish', 'spa': 'Spanish',
    'de': 'German', 'deu': 'German',
    'ar': 'Arabic', 'ara': 'Arabic',
    'zh': 'Chinese', 'zho': 'Chinese',
    'hi': 'Hindi', 'hin': 'Hindi',
    'pt': 'Portuguese', 'por': 'Portuguese',
    'ru': 'Russian', 'rus': 'Russian',
    'ja': 'Japanese', 'jpn': 'Japanese',
    'ko': 'Korean', 'kor': 'Korean',
}


def get_language_code_mapping() -> Dict[str, str]:
    """
    Get ISO 639 language code to full name mapping.
    
    Returns:
        Dict mapping 2-char and 3-char ISO codes to language names
    """
    return LANGUAGE_CODE_MAP


def detect_response_language(
    response: str,
    target_language: str,
    clients: Dict,
    detection_config: Dict,
    sample: Optional[Dict] = None,
    language_code_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Detect language of response using configured detection model.
    
    Args:
        response: Text to analyze
        target_language: Expected language code (ISO 639)
        clients: Initialized API clients
        detection_config: Language detection configuration from config file
        sample: Optional sample dict with context/question for better detection
        language_code_mapping: Optional custom language code mapping
        
    Returns:
        Dict with detected_language, target_language, is_correct_language, 
        confidence, and full_response
    """
    mapping = language_code_mapping or LANGUAGE_CODE_MAP
    target_name = mapping.get(target_language.lower(), target_language)
    
    # Require detection_prompt_template in config
    if 'detection_prompt_template' not in detection_config:
        logger.error("❌ No detection_prompt_template in detection config")
        return {
            "detected_language": "Error",
            "target_language": target_name,
            "is_correct_language": False,
            "confidence": "Unknown",
            "full_response": "Missing detection_prompt_template in config"
        }
    
    # Extract context and question from sample (if available)
    context = sample.get('context', '') if sample else ''
    question = sample.get('question', '') if sample else ''
    
    detection_prompt = detection_config['detection_prompt_template'].format(
        response=response,
        target_name=target_name,
        context=context,
        question=question
    )
    
    logger.debug(f"Detecting language (target: {target_name})")
    
    error_response = {
        "detected_language": "Unknown",
        "target_language": target_name,
        "is_correct_language": False,
        "confidence": "Unknown",
        "full_response": ""
    }
    
    try:
        # Require model in detection config
        if 'model' not in detection_config:
            logger.error("❌ No model specified in detection config")
            error_response["full_response"] = "Missing model in detection config"
            return error_response
        
        # Use model field directly - generate_endpoint_response will handle it
        detection_model_config = {
            "model": detection_config['model'],
            "generation": detection_config.get("generation", {})
        }
        
        # Format system prompt with target language if it contains {target_name}
        system_prompt = detection_config.get("system_prompt", "")
        if "{target_name}" in system_prompt:
            system_prompt = system_prompt.format(target_name=target_name)
        
        detection_response = generate_endpoint_response(
            prompt=detection_prompt,
            model_config=detection_model_config,
            clients=clients,
            system_prompt=system_prompt
        )
        
        if not detection_response or detection_response.startswith("Error:"):
            logger.warning(f"⚠️ Language detection failed: {detection_response}")
            error_response["full_response"] = detection_response
            return error_response
        
        # Parse structured response - only response language check
        response_has_other = ""
        detected = ""
        confidence = ""
        matches = ""
        
        for line in detection_response.split('\n'):
            line = line.strip()
            if line.startswith("Response_Has_Other_Language:"):
                response_has_other = line.replace("Response_Has_Other_Language:", "").strip()
            elif line.startswith("Detected_Language:"):
                detected = line.replace("Detected_Language:", "").strip()
            elif line.startswith("Confidence:"):
                confidence = line.replace("Confidence:", "").strip()
            elif line.startswith("Matches_Expected:"):
                matches = line.replace("Matches_Expected:", "").strip()
        
        # Determine correctness: response is correct if it does NOT have >10% other language
        if response_has_other:
            is_correct = response_has_other.lower() in ('no', 'false')
        elif matches:
            is_correct = matches.lower() in ('yes', 'true', 'correct')
        else:
            # Fallback: fuzzy match
            is_correct = (
                detected.lower() == target_name.lower() or
                detected.lower() in target_name.lower() or
                target_name.lower() in detected.lower()
            )
        
        return {
            "detected_language": detected or "Unknown",
            "target_language": target_name,
            "is_correct_language": is_correct,
            "response_has_other_language": response_has_other.lower() == 'yes' if response_has_other else False,
            "confidence": confidence or "Unknown",
            "full_response": detection_response
        }
        
    except Exception as e:
        logger.error(f"❌ Language detection error: {e}")
        error_response["detected_language"] = "Error"
        error_response["full_response"] = str(e)
        return error_response


def format_prompt(
    sample: Dict,
    dataset_name: str,
    dataset_config: Dict,
    language_names: Optional[Dict[str, str]] = None
) -> str:
    """
    Format dataset sample into prompt using template.
    
    Args:
        sample: Dataset sample dict
        dataset_name: Dataset identifier (aya, multi-wiki-qa, xlsum)
        dataset_config: Dataset configuration with prompt_template
        language_names: Optional language code to name mapping
        
    Returns:
        Formatted prompt string
    """
    # Require prompt_template in config - no default
    if 'prompt_template' not in dataset_config:
        logger.error(f"❌ No prompt_template in dataset config for {dataset_name}")
        return ""
    
    template = dataset_config['prompt_template']
    lang_code = sample.get('language', '')
    
    # Use LANGUAGE_CODE_MAP by default to convert codes to full names
    mapping = language_names or LANGUAGE_CODE_MAP
    lang_name = mapping.get(lang_code.lower(), lang_code) if lang_code else 'English'
    
    # Dataset-specific formatting
    if dataset_name == 'aya':
        return template.format(
            instruction=sample.get('inputs', ''),
            language=lang_name
        )
    elif dataset_name == 'multi-wiki-qa':
        return template.format(
            context=sample.get('context', ''),
            question=sample.get('question', ''),
            language=lang_name
        )
    elif dataset_name == 'xlsum':
        return template.format(
            title=sample.get('title', ''),
            text=sample.get('text', ''),
            language=lang_name
        )
    else:
        # Generic fallback
        return sample.get('inputs', sample.get('instruction', ''))


def compute_language_accuracy_stats(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute language accuracy statistics from evaluation results.
    
    Args:
        results: List of evaluation results with language_detection_a/b fields
        
    Returns:
        Dict with overall and per-language accuracy statistics
    """
    stats = {
        'a_correct': 0,
        'a_incorrect': 0,
        'b_correct': 0,
        'b_incorrect': 0,
        'by_language': {},
        'skipped': 0
    }
    
    for result in results:
        det_a = result.get('language_detection_a')
        det_b = result.get('language_detection_b')
        
        # Skip if language detection not available
        if not det_a or not det_b:
            stats['skipped'] += 1
            continue
        
        # Skip samples where context/question is in wrong language
        if result.get('excluded_from_language_stats', False):
            stats['skipped'] += 1
            continue
        
        target_lang = det_a.get('target_language', 'Unknown')
        
        # Initialize language stats
        if target_lang not in stats['by_language']:
            stats['by_language'][target_lang] = {
                'a_correct': 0,
                'a_incorrect': 0,
                'b_correct': 0,
                'b_incorrect': 0,
                'total': 0
            }
        
        stats['by_language'][target_lang]['total'] += 1
        
        # Update model A counts
        if det_a.get('is_correct_language', False):
            stats['a_correct'] += 1
            stats['by_language'][target_lang]['a_correct'] += 1
        else:
            stats['a_incorrect'] += 1
            stats['by_language'][target_lang]['a_incorrect'] += 1
        
        # Update model B counts
        if det_b.get('is_correct_language', False):
            stats['b_correct'] += 1
            stats['by_language'][target_lang]['b_correct'] += 1
        else:
            stats['b_incorrect'] += 1
            stats['by_language'][target_lang]['b_incorrect'] += 1
    
    total_samples = len(results) - stats['skipped']
    
    # Build result with overall statistics
    total_correct = stats['a_correct'] + stats['b_correct']
    total_evaluated = total_samples * 2  # Both models evaluated per sample
    
    result = {
        'model_a_correct': stats['a_correct'],
        'model_a_incorrect': stats['a_incorrect'],
        'model_a_accuracy': (stats['a_correct'] / total_samples) if total_samples > 0 else 0.0,
        'model_b_correct': stats['b_correct'],
        'model_b_incorrect': stats['b_incorrect'],
        'model_b_accuracy': (stats['b_correct'] / total_samples) if total_samples > 0 else 0.0,
        'overall_accuracy': (total_correct / total_evaluated) if total_evaluated > 0 else 0.0,
        'total_samples': total_samples,
        'skipped_samples': stats['skipped'],
        'by_language': {}
    }
    
    # Add per-language statistics
    for lang, lang_stats in stats['by_language'].items():
        lang_total = lang_stats['total']
        result['by_language'][lang] = {
            'model_a_correct': lang_stats['a_correct'],
            'model_a_incorrect': lang_stats['a_incorrect'],
            'model_a_accuracy': (lang_stats['a_correct'] / lang_total) if lang_total > 0 else 0.0,
            'model_b_correct': lang_stats['b_correct'],
            'model_b_incorrect': lang_stats['b_incorrect'],
            'model_b_accuracy': (lang_stats['b_correct'] / lang_total) if lang_total > 0 else 0.0,
            'total': lang_total
        }
    
    if stats['skipped'] > 0:
        logger.info(f"Language stats: {total_samples} processed, {stats['skipped']} skipped")
    
    return result


def _format_judge_prompt(
    sample: Dict,
    dataset_name: str,
    template: str,
    language_name: str,
    response_a: str,
    response_b: str
) -> str:
    """
    Format judge prompt based on dataset type.
    
    Args:
        sample: Dataset sample
        dataset_name: Dataset identifier
        template: Judge prompt template
        language_name: Language name for context
        response_a: Model A response
        response_b: Model B response
        
    Returns:
        Formatted judge prompt
    """
    if dataset_name == 'aya':
        return template.format(
            language=language_name,
            instruction=sample.get('inputs', ''),
            completion_1=response_a,
            completion_2=response_b
        )
    elif dataset_name == 'multi-wiki-qa':
        return template.format(
            language=language_name,
            context=sample.get('context', ''),
            question=sample.get('question', ''),
            answers=sample.get('answer_text', ''),
            completion_1=response_a,
            completion_2=response_b
        )
    elif dataset_name == 'xlsum':
        return template.format(
            title=sample.get('title', ''),
            text=sample.get('text', ''),
            summary_a=response_a,
            summary_b=response_b,
            language=language_name
        )
    else:
        # Generic fallback
        return (
            f"Compare these responses:\n"
            f"Prompt: {sample.get('inputs', '')}\n"
            f"Response A: {response_a}\n"
            f"Response B: {response_b}\n"
            f"Which is better?"
        )


def _parse_judge_response(judge_response: str) -> Tuple[str, str, str, float, float]:
    """Parse judge response to extract comparison, preferred, winner, and ratings."""
    comparison, preferred, rating_a, rating_b = "", "", 0.0, 0.0
    
    for line in judge_response.split('\n'):
        if line.startswith("Comparison:"):
            comparison = line.replace("Comparison:", "").strip()
        elif line.startswith("Preferred:"):
            preferred = line.replace("Preferred:", "").strip()
        elif line.startswith("Rating_A:"):
            try:
                rating_str = line.replace("Rating_A:", "").strip().strip('[]')
                # Remove any text in parentheses (e.g., "4 (explanation)")
                if '(' in rating_str:
                    rating_str = rating_str.split('(')[0].strip()
                # Handle formats like "2/5", "2", "[2]"
                if '/' in rating_str:
                    rating_a = float(rating_str.split('/')[0].strip())
                else:
                    rating_a = float(rating_str)
            except:
                pass
        elif line.startswith("Rating_B:"):
            try:
                rating_str = line.replace("Rating_B:", "").strip().strip('[]')
                # Remove any text in parentheses (e.g., "5 (explanation)")
                if '(' in rating_str:
                    rating_str = rating_str.split('(')[0].strip()
                # Handle formats like "4/5", "4", "[4]"
                if '/' in rating_str:
                    rating_b = float(rating_str.split('/')[0].strip())
                else:
                    rating_b = float(rating_str)
            except:
                pass
    
    # Fallback for preferred
    if not preferred and "Preferred:" in judge_response:
        preferred = judge_response.split("Preferred:")[1].strip().split('\n')[0].strip()
    
    # Determine winner
    full_text = f"{comparison} {preferred} {judge_response}".lower()
    if "answer (a)" in preferred.lower():
        winner = "model_a"
    elif "answer (b)" in preferred.lower():
        winner = "model_b"
    else:
        winner = "tie"
    
    return comparison, preferred, winner, rating_a, rating_b


def judge_responses(
    sample: Dict,
    response_a: str,
    response_b: str,
    dataset_name: str,
    dataset_config: Dict,
    judge_config: Dict,
    clients: Dict,
    language_names: Optional[Dict[str, str]] = None,
    detection_config: Optional[Dict] = None
) -> Dict:
    """
    Use judge model to compare two responses.
    
    Args:
        sample: Dataset sample
        response_a: Model A response
        response_b: Model B response
        dataset_name: Dataset identifier
        dataset_config: Dataset configuration
        judge_config: Judge model configuration
        clients: API clients
        language_names: Optional language mapping
        detection_config: Optional language detection config (if None, detection skipped)
        
    Returns:
        Dict with comparison results and optional language detection
    """
    language_code = sample.get('language', '')
    
    # Use LANGUAGE_CODE_MAP by default to convert codes to full names
    mapping = language_names or LANGUAGE_CODE_MAP
    language_name = mapping.get(language_code.lower(), language_code) if language_code else 'English'
    
    # Require judge_prompt_template in config
    if 'judge_prompt_template' not in dataset_config:
        logger.error(f"❌ No judge_prompt_template in dataset config for {dataset_name}")
        return {"comparison": "Error: Missing judge template", "preferred": "TIE", 
                "winner": "tie", "rating_a": 0, "rating_b": 0, "full_response": ""}
    
    template = dataset_config['judge_prompt_template']
    judge_prompt = _format_judge_prompt(sample, dataset_name, template, language_name, response_a, response_b)
    
    try:
        # Optional language detection BEFORE judging - only if detection_config provided
        detection_a = None
        detection_b = None
        
        if detection_config:
            logger.info("Detecting response languages...")
            detection_a = detect_response_language(
                response_a, language_code, clients, detection_config, sample, LANGUAGE_CODE_MAP
            )
            detection_b = detect_response_language(
                response_b, language_code, clients, detection_config, sample, LANGUAGE_CODE_MAP
            )
            
        # Language detection is for analysis only - all samples are judged
        # If detection is disabled, proceed with judging
        logger.info(f"Judging {dataset_name} responses...")
        logger.debug(f"Judge prompt: {judge_prompt[:500]}{'...' if len(judge_prompt) > 500 else ''}")
        
        judge_response = generate_endpoint_response(
            prompt=judge_prompt,
            model_config=judge_config,
            clients=clients,
            system_prompt=judge_config.get("system_prompt")
        )
        
        if not judge_response or judge_response.startswith("Error:"):
            logger.warning(f"Judge error: {judge_response}")
            return {"comparison": "Error in judging", "preferred": "TIE", "winner": "tie",
                   "rating_a": 0, "rating_b": 0, "full_response": judge_response}
        
        comparison, preferred, winner, rating_a, rating_b = _parse_judge_response(judge_response)
        
        result = {
            "comparison": comparison,
            "preferred": preferred,
            "winner": winner,
            "rating_a": rating_a,
            "rating_b": rating_b,
            "full_response": judge_response
        }
        
        # Add language detection results if they were computed
        if detection_config:
            result["language_detection_a"] = detection_a
            result["language_detection_b"] = detection_b
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Judging error: {e}")
        error_result = {
            "comparison": "Error in judging",
            "preferred": "TIE",
            "winner": "tie",
            "rating_a": 0,
            "rating_b": 0,
            "full_response": str(e)
        }
        
        # Add error language detection results if detection_config provided
        if detection_config:
            error_dict = {
                "detected_language": "Error",
                "target_language": language_code,
                "is_correct_language": False,
                "confidence": "Unknown",
                "full_response": str(e)
            }
            error_result["language_detection_a"] = error_dict
            error_result["language_detection_b"] = error_dict.copy()
        
        return error_result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("✅ LLM Judge Utilities - Configuration-Driven Evaluation")
    print("\n📋 All parameters controlled via YAML configuration files")
    print("\nCore Functions:")
    print("  • load_huggingface_model() - Load HF models with Flash Attention 2")
    print("  • generate_response() - Single prompt generation")
    print("  • batch_generate_responses() - Batch generation")
    print("  • cleanup_model() - Free GPU memory")
    print("\nAPI Functions:")
    print("  • initialize_endpoint_clients() - Initialize Azure/GPT-OSS-120B")
    print("  • generate_endpoint_response() - Call API endpoints (config-driven)")
    print("\nDataset Functions:")
    print("  • load_dataset_samples() - Load from HuggingFace datasets")
    print("  • format_prompt() - Format prompts with templates (from config)")
    print("\nEvaluation Functions:")
    print("  • judge_responses() - Compare responses (optional language detection)")
    print("  • detect_response_language() - Language detection (config-driven)")
    print("  • compute_language_accuracy_stats() - Language accuracy metrics")
    print("\nConfiguration Functions:")
    print("  • find_model_config() - Resolve model config with family inheritance")
    print("  • get_comparison_models() - Extract models A & B")
    print("  • get_judge_config() - Extract judge model config")
    print("  • get_evaluation_config() - Extract evaluation settings")
    print("  • get_language_code_mapping() - ISO 639 language code mapping")
    print("\n⚙️  No hardcoded defaults - all behavior controlled by configuration files")
