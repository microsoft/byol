# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Shared utilities for local model translators.

Provides safe HuggingFace authentication, model loading helpers, and cache management.
"""

import os
from functools import lru_cache
from typing import Optional, Tuple, Any, List, Callable

from byol.common.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Model Cache Registry
# =============================================================================
# List of lru_cache-decorated functions that load models.
# Each module should register its loader by appending to this list.
# This allows clear_model_caches() to clear all model caches without
# hardcoding imports for each model type.
_model_caches: List[Callable] = []


def register_model_cache(func: Callable) -> Callable:
    """
    Decorator to register a model loader function for cache management.
    
    Usage:
        @lru_cache(maxsize=2)
        @register_model_cache
        def _load_my_model(model_name: str, device: str):
            ...
    """
    _model_caches.append(func)
    return func


def clear_model_caches() -> None:
    """
    Clear all registered model caches and free GPU memory.
    
    This should be called between local model runs to free VRAM.
    """
    import gc
    
    for cache_func in _model_caches:
        try:
            cache_func.cache_clear()
            logger.debug(f"Cleared cache for {cache_func.__name__}")
        except AttributeError:
            # Not an lru_cache decorated function
            logger.warning(f"{cache_func.__name__} does not have cache_clear()")
    
    # Run garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")
    except ImportError:
        pass


def safe_hf_login() -> bool:
    """
    Safely login to HuggingFace Hub if token is available.
    
    Returns:
        True if login successful, False otherwise.
    """
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(hf_token, add_to_git_credential=False)
            logger.debug("Successfully logged in to HuggingFace Hub")
            return True
        except Exception as e:
            logger.warning(f"HuggingFace login failed: {e}")
    return False


def normalize_device(device: Optional[str]) -> str:
    """
    Normalize device string to a consistent format for cache keys.
    
    This should be called BEFORE passing device to lru_cache-decorated
    model loaders to ensure consistent cache key generation.
    
    Args:
        device: Device string in various formats:
            - None -> "cuda:0"
            - "0", "1", "3" -> "cuda:0", "cuda:1", "cuda:3"
            - "cuda" -> "cuda:0"
            - "cuda:0" -> "cuda:0" (unchanged)
            - "cpu" -> "cpu" (unchanged)
    
    Returns:
        Normalized device string in format "cuda:X" or "cpu".
    """
    if device is None:
        return "cuda:0"
    
    device = str(device).strip().lower()
    
    # Handle plain GPU index (e.g., "3" -> "cuda:3")
    if device.isdigit():
        return f"cuda:{device}"
    
    # Handle "cuda" without index -> "cuda:0"
    if device == "cuda":
        return "cuda:0"
    
    return device


def get_torch_device(requested_device: Optional[str] = "cuda:0") -> str:
    """
    Get an available torch device, falling back to CPU if needed.
    
    Args:
        requested_device: Preferred device. Accepts multiple formats:
            - "cuda:0", "cuda:1" - full CUDA device string
            - "0", "1", "3" - just the GPU index (converted to cuda:X)
            - "cuda" - default CUDA device (cuda:0)
            - "cpu" - CPU device
        
    Returns:
        Available device string in format "cuda:X" or "cpu".
    """
    # Normalize first to ensure consistent format
    device = normalize_device(requested_device)
    
    if "cuda" in device:
        try:
            import torch
            if torch.cuda.is_available():
                # Validate the device index exists
                device_idx = int(device.split(":")[-1]) if ":" in device else 0
                if device_idx < torch.cuda.device_count():
                    return device
                else:
                    logger.warning(f"CUDA device {device_idx} not available (have {torch.cuda.device_count()} GPUs) - using cuda:0")
                    return "cuda:0"
            logger.warning("CUDA requested but not available - using CPU")
        except ImportError:
            pass
    return "cpu"


def get_torch_dtype(device: str):
    """Get appropriate torch dtype for device."""
    import torch
    if "cuda" in device:
        return torch.float16
    return torch.float32
