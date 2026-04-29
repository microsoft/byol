# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Secrets management — load API keys from environment variables.

All credentials should be set via environment variables (loaded from ``.env``
by each CLI entry point using ``python-dotenv``).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .constants import ENV_HF_TOKEN, ENV_WANDB_API_KEY, ENV_WANDB_PROJECT

logger = logging.getLogger("byol-train")


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from ``HF_TOKEN`` environment variable."""
    token = os.environ.get(ENV_HF_TOKEN)
    if not token:
        logger.warning(
            "HuggingFace token not found. Set HF_TOKEN in your .env file."
        )
    return token


def get_wandb_key() -> Optional[str]:
    """Get W&B API key from ``WANDB_API_KEY`` environment variable."""
    return os.environ.get(ENV_WANDB_API_KEY)


def get_wandb_project() -> Optional[str]:
    """Get W&B project name from ``WANDB_PROJECT`` environment variable."""
    return os.environ.get(ENV_WANDB_PROJECT)


def setup_environment() -> None:
    """Ensure tokens are in ``os.environ`` for subprocess calls (LlamaFactory)."""
    hf_token = get_hf_token()
    if hf_token:
        os.environ[ENV_HF_TOKEN] = hf_token
        logger.info("HuggingFace token configured")

    wandb_key = get_wandb_key()
    if wandb_key:
        os.environ[ENV_WANDB_API_KEY] = wandb_key

    wandb_project = get_wandb_project()
    if wandb_project:
        os.environ[ENV_WANDB_PROJECT] = wandb_project


def mask_token(token: Optional[str], visible_chars: int = 4) -> str:
    """Mask token for safe logging (e.g., ``hf_Ab...xYz``)."""
    if not token:
        return "<not set>"
    if len(token) <= visible_chars * 2:
        return "*" * len(token)
    return f"{token[:visible_chars]}...{token[-visible_chars:]}"
