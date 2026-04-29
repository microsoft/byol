# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Text normalization utilities."""

from __future__ import annotations

from functools import lru_cache

from sacremoses import MosesPunctNormalizer


@lru_cache(maxsize=1)
def _get_normalizer() -> MosesPunctNormalizer:
    """Get a cached MosesPunctNormalizer instance."""
    return MosesPunctNormalizer()


def normalize_text(text: str) -> str:
    """
    Normalize text using Moses punctuation normalizer.
    
    This handles Unicode punctuation normalization for consistent
    text comparison across different sources.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return text
    
    mpn = _get_normalizer()
    return mpn.normalize(text)


def normalize_dataset(
    dataset: list[dict], 
    text_field: str = "text",
) -> list[dict]:
    """
    Normalize the text field in a dataset.
    
    Args:
        dataset: List of dictionaries containing text
        text_field: Name of the field to normalize
        
    Returns:
        New list with normalized text
    """
    normalized = []
    for sample in dataset:
        normalized_sample = sample.copy()
        if text_field in normalized_sample:
            normalized_sample[text_field] = normalize_text(sample[text_field])
        normalized.append(normalized_sample)
    return normalized


__all__ = [
    "normalize_text",
    "normalize_dataset",
]
