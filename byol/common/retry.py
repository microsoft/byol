# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Retry logic with exponential backoff for API rate limits.

Provides utilities for handling transient failures and rate limits
in API calls with configurable retry behavior.

This module is the single source of truth for all retry-related constants
and functions across the BYOL package.
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable, TypeVar

from byol.common.logging import get_logger
from byol.common.exceptions import RateLimitError

logger = get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# Default Configuration
# =============================================================================

# Retry settings for rate limiting
MAX_RETRIES: int = 5
INITIAL_BACKOFF: float = 1.0  # seconds
MAX_BACKOFF: float = 60.0  # seconds
BACKOFF_MULTIPLIER: float = 2.0

# Common rate limit error patterns (case-insensitive)
RATE_LIMIT_PATTERNS = [
    "rate limit",
    "ratelimit",
    "429",
    "too many requests",
    "quota exceeded",
    "throttl",
    "retry after",
    "capacity",
    "overloaded",
    "server busy",
]


# =============================================================================
# Error Detection
# =============================================================================

def is_rate_limit_error(error: Exception) -> bool:
    """
    Check if an exception is a rate limit error.
    
    Args:
        error: The exception to check.
        
    Returns:
        True if the error appears to be a rate limit error.
    """
    # Check if it's already a RateLimitError from our exception hierarchy
    if isinstance(error, RateLimitError):
        return True
    
    error_str = str(error).lower()
    return any(pattern in error_str for pattern in RATE_LIMIT_PATTERNS)


def is_transient_error(error: Exception) -> bool:
    """
    Check if an exception is a transient error worth retrying.
    
    Includes rate limits and temporary network/server errors.
    
    Args:
        error: The exception to check.
        
    Returns:
        True if the error is likely transient and worth retrying.
    """
    if is_rate_limit_error(error):
        return True
    
    # Additional transient error patterns
    error_str = str(error).lower()
    transient_patterns = [
        "connection",
        "timeout",
        "temporary",
        "unavailable",
        "503",
        "502",
        "500",
    ]
    return any(pattern in error_str for pattern in transient_patterns)


# =============================================================================
# Retry Functions
# =============================================================================

def retry_with_backoff(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF,
    max_backoff: float = MAX_BACKOFF,
    backoff_multiplier: float = BACKOFF_MULTIPLIER,
    verbose: bool = False,
    **kwargs: Any,
) -> T:
    """
    Execute a function with exponential backoff retry for transient errors.
    
    Args:
        func: The function to call.
        *args: Positional arguments to pass to the function.
        max_retries: Maximum number of retry attempts.
        initial_backoff: Initial wait time in seconds.
        max_backoff: Maximum wait time in seconds.
        backoff_multiplier: Multiplier for exponential backoff.
        verbose: Whether to log retry messages.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        The function's return value.
        
    Raises:
        Exception: If all retries are exhausted.
        RateLimitError: If rate limit retries are exhausted.
        
    Example:
        >>> result = retry_with_backoff(api_call, text, max_retries=3)
    """
    last_exception: Exception | None = None
    backoff = initial_backoff
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt >= max_retries:
                logger.warning(f"All {max_retries + 1} attempts failed: {e}")
                # Wrap rate limit errors in RateLimitError
                if is_rate_limit_error(e) and not isinstance(e, RateLimitError):
                    raise RateLimitError(
                        f"Rate limit exceeded after {max_retries + 1} retries: {e}",
                        original_error=e,
                    ) from e
                raise last_exception
            
            if not is_transient_error(e):
                # Non-transient errors should not be retried
                logger.error(f"Non-transient error, not retrying: {e}")
                raise
            
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0, backoff * 0.5)
            sleep_time = min(backoff + jitter, max_backoff)
            
            if verbose:
                logger.info(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed, "
                    f"retrying in {sleep_time:.1f}s: {str(e)[:100]}"
                )
            else:
                logger.debug(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed, "
                    f"retrying in {sleep_time:.1f}s: {str(e)[:100]}"
                )
            
            time.sleep(sleep_time)
            backoff = min(backoff * backoff_multiplier, max_backoff)
    
    # Should not reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in retry_with_backoff")


def translate_with_retry(
    translator: Any,
    text: str,
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF,
    max_backoff: float = MAX_BACKOFF,
    backoff_multiplier: float = BACKOFF_MULTIPLIER,
    verbose: bool = True,
) -> str:
    """
    Translate text with exponential backoff retry for rate limits.
    
    This is a convenience wrapper specifically for translator objects
    that have a .translate() method.
    
    Args:
        translator: The translator object (must have a .translate() method).
        text: Text to translate.
        max_retries: Maximum number of retry attempts.
        initial_backoff: Initial wait time in seconds.
        max_backoff: Maximum wait time in seconds.
        backoff_multiplier: Multiplier for exponential backoff.
        verbose: Whether to log retry messages at INFO level (vs DEBUG).
        
    Returns:
        Translated text.
        
    Raises:
        Exception: If all retries are exhausted.
        RateLimitError: If rate limit retries are exhausted.
        
    Example:
        >>> from byol.translation_backends import get_translator
        >>> translator = get_translator("gpt-4o", tgt_lang="Spanish")
        >>> result = translate_with_retry(translator, "Hello world")
    """
    last_exception: Exception | None = None
    backoff = initial_backoff
    
    for attempt in range(max_retries + 1):
        try:
            return translator.translate(text=text)
        except Exception as e:
            last_exception = e
            
            if attempt >= max_retries:
                # Wrap rate limit errors in RateLimitError
                if is_rate_limit_error(e) and not isinstance(e, RateLimitError):
                    raise RateLimitError(
                        f"Rate limit exceeded after {max_retries + 1} retries: {e}",
                        original_error=e,
                    ) from e
                raise last_exception
            
            if is_rate_limit_error(e):
                # Add jitter to prevent thundering herd
                jitter = random.uniform(0, backoff * 0.5)
                sleep_time = min(backoff + jitter, max_backoff)
                
                if verbose:
                    logger.warning(
                        f"[Rate limit] Attempt {attempt + 1}/{max_retries + 1} failed. "
                        f"Retrying in {sleep_time:.1f}s... Error: {str(e)[:100]}"
                    )
                else:
                    logger.debug(
                        f"[Rate limit] Attempt {attempt + 1}/{max_retries + 1} failed. "
                        f"Retrying in {sleep_time:.1f}s... Error: {str(e)[:100]}"
                    )
                
                time.sleep(sleep_time)
                backoff = min(backoff * backoff_multiplier, max_backoff)
            else:
                # Non-rate-limit errors: shorter backoff
                sleep_time = min(backoff * 0.5, 5.0)
                
                if verbose:
                    logger.warning(
                        f"[Error] Attempt {attempt + 1}/{max_retries + 1} failed. "
                        f"Retrying in {sleep_time:.1f}s... Error: {str(e)[:100]}"
                    )
                else:
                    logger.debug(
                        f"[Error] Attempt {attempt + 1}/{max_retries + 1} failed. "
                        f"Retrying in {sleep_time:.1f}s... Error: {str(e)[:100]}"
                    )
                
                time.sleep(sleep_time)
                backoff = min(backoff * backoff_multiplier, max_backoff)
    
    # Should not reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in translate_with_retry")


__all__ = [
    # Configuration
    "MAX_RETRIES",
    "INITIAL_BACKOFF",
    "MAX_BACKOFF",
    "BACKOFF_MULTIPLIER",
    # Functions
    "is_rate_limit_error",
    "is_transient_error",
    "retry_with_backoff",
    "translate_with_retry",
]
