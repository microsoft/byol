# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LLM-as-Judge evaluation module for BYOL.

This module provides LLM-as-Judge evaluation functionality,
allowing comparison of two language models using an LLM as arbiter.
"""

from .run_llm_judge import run_evaluation

__all__ = ["run_evaluation"]
