# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BYOL Training Framework — LlamaFactory wrapper.

A typed interface for training LLMs with LlamaFactory backend.
Supports CPT, SFT, and DPO training stages.

Example:
    >>> from byol.train import TrainConfig, TrainingRunner
    >>> config = TrainConfig.from_yaml("configs/train/sft.yaml")
    >>> runner = TrainingRunner(config)
    >>> result = runner.run()
"""

__version__ = "0.1.0"

from .config import DatasetMixConfig, LoraConfig, TrainConfig
from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_RANK,
    SUPPORTED_STAGES,
)
# Lazy imports for heavy modules (torch, transformers) to avoid
# initialising CUDA before CUDA_VISIBLE_DEVICES is set by the runner.
from .runner import TrainingRunner, TrainResult
from .secrets import get_hf_token, get_wandb_key, setup_environment


def __getattr__(name: str):
    """Lazy-load merge symbols so torch/transformers aren't imported eagerly."""
    if name in ("MergeConfig", "merge_lora"):
        from .merge import MergeConfig, merge_lora  # noqa: F811
        globals()["MergeConfig"] = MergeConfig
        globals()["merge_lora"] = merge_lora
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Version
    "__version__",
    # Config
    "TrainConfig",
    "LoraConfig",
    "DatasetMixConfig",
    # Runner
    "TrainingRunner",
    "TrainResult",
    # Merge
    "MergeConfig",
    "merge_lora",
    # Secrets
    "get_hf_token",
    "get_wandb_key",
    "setup_environment",
    # Constants
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_LORA_RANK",
    "SUPPORTED_STAGES",
]
