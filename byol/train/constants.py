# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Constants and default values for BYOL Training Framework."""

from __future__ import annotations

from pathlib import Path as _Path

# =============================================================================
# Package / Repo Paths
# =============================================================================
TRAIN_PACKAGE_DIR: _Path = _Path(__file__).resolve().parent
REPO_ROOT: _Path = TRAIN_PACKAGE_DIR.parent.parent  # byol/train → repo root

DEFAULT_DATA_DIR: str = str(REPO_ROOT / "data" / "train")
DEFAULT_CONFIGS_DIR: str = "configs/train"
DEFAULT_OUTPUT_DIR: str = "results/train"
DEFAULT_EXAMPLES_DIR: str = str(TRAIN_PACKAGE_DIR / "examples")

# =============================================================================
# Training Defaults
# =============================================================================
DEFAULT_BATCH_SIZE: int = 4
DEFAULT_GRADIENT_ACCUMULATION_STEPS: int = 4
DEFAULT_EPOCHS: int = 3
DEFAULT_LEARNING_RATE: float = 5e-5
DEFAULT_WARMUP_RATIO: float = 0.1
DEFAULT_MAX_GRAD_NORM: float = 1.0
DEFAULT_WEIGHT_DECAY: float = 0.01

# =============================================================================
# LoRA Defaults
# =============================================================================
DEFAULT_LORA_RANK: int = 16
DEFAULT_LORA_ALPHA: int = 32
DEFAULT_LORA_DROPOUT: float = 0.05
DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = ("q_proj", "v_proj")

# =============================================================================
# Model Defaults
# =============================================================================
DEFAULT_DTYPE: str = "bfloat16"
DEFAULT_MAX_LENGTH: int = 8192
DEFAULT_CUTOFF_LEN: int = 8192

# =============================================================================
# Default Config Paths (relative to repo root)
# =============================================================================
DEFAULT_CONFIG_CPT: str = "configs/train/cpt.yaml"
DEFAULT_CONFIG_SFT: str = "configs/train/sft.yaml"
DEFAULT_CONFIG_DPO: str = "configs/train/dpo.yaml"

# =============================================================================
# Supported Values
# =============================================================================
SUPPORTED_STAGES: tuple[str, ...] = ("pt", "cpt", "sft", "dpo")
SUPPORTED_DTYPES: tuple[str, ...] = ("bfloat16", "float16", "float32", "auto")
SUPPORTED_MIX_STRATEGIES: tuple[str, ...] = ("concat", "interleave_under", "interleave_over")
SUPPORTED_TEMPLATES: tuple[str, ...] = ("gemma", "llama3", "mistral", "qwen2", "default")

# =============================================================================
# Environment Variables
# =============================================================================
ENV_HF_TOKEN: str = "HF_TOKEN"
ENV_WANDB_API_KEY: str = "WANDB_API_KEY"
ENV_WANDB_PROJECT: str = "WANDB_PROJECT"
ENV_CUDA_VISIBLE_DEVICES: str = "CUDA_VISIBLE_DEVICES"

# =============================================================================
# File Patterns
# =============================================================================
TEMP_CONFIG_PREFIX: str = "byol_train_"
TEMP_CONFIG_SUFFIX: str = ".yaml"
