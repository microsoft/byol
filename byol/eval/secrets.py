# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Secrets management for BYOL Evaluation Framework.

All credentials should be set via environment variables (loaded from ``.env``
by each CLI entry point using ``python-dotenv``).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

HF_TOKEN_ENV: str = "HF_TOKEN"
HF_TOKEN_ALT_ENV: str = "HUGGING_FACE_HUB_TOKEN"


def get_hf_token() -> str | None:
    """Get HuggingFace token from environment variables.

    Checks ``HF_TOKEN`` then ``HUGGING_FACE_HUB_TOKEN``.
    """
    token = os.environ.get(HF_TOKEN_ENV) or os.environ.get(HF_TOKEN_ALT_ENV)
    if not token:
        logger.warning("No HuggingFace token found. Set HF_TOKEN in your .env file.")
    return token


def setup_hf_environment(token: str | None = None) -> None:
    """Set up HuggingFace environment variables for subprocess calls (lm-eval).

    Args:
        token: Optional token to use.  If not provided, reads from env.
    """
    if token is None:
        token = get_hf_token()

    if token:
        os.environ[HF_TOKEN_ENV] = token
        logger.info("HuggingFace token configured")

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def mask_token(token: str | None) -> str:
    """Mask a token for safe logging."""
    if not token:
        return "<not set>"
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}...{token[-4:]}"
