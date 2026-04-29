# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
BYOL Evaluation Framework — benchmark LLMs with lm-evaluation-harness and LLM-as-Judge.

Submodules:
    - cli:          Command-line interface (``byol-eval`` entry-point)
    - config:       Dataclass-based YAML configuration (``EvalConfig``, ``TaskConfig``, …)
    - runner:       Orchestrates lm-eval subprocess runs
    - judge:        LLM-as-Judge evaluation wrapper
    - secrets:      HuggingFace token resolution
    - constants:    Centralised defaults and magic values
    - extract_benchmarks: Parse lm-eval log files into benchmark tables

Usage:
    # CLI
    byol-eval --model google/gemma-3-4b-pt --type base --tgt-lang mri --gpus 0
    byol-eval judge --model-config configs/judge_models.yaml --dataset-config configs/judge_datasets.yaml

    # Python API
    from byol.eval import EvalConfig, EvaluationRunner
    config = EvalConfig.from_yaml("configs/eval/benchmark_base_eng.yaml")
    runner = EvaluationRunner(config)
    results = runner.run_all()

    # Extract benchmark scores
    python -m byol.eval.extract_benchmarks results/ --type base --tgt-lang eng
"""

__version__ = "0.1.0"
__author__ = "BYOL Team"

# Config classes
from .config import EvalConfig, ModelConfig, TaskConfig

# Runner
from .runner import EvaluationRunner, EvalResult

# Secrets management
from .secrets import get_hf_token, setup_hf_environment, mask_token

# Constants
from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DTYPE,
    DEFAULT_GPUS,
    DEFAULT_OUTPUT_DIR,
    STATUS_SUCCESS,
    STATUS_FAILED,
    STATUS_SKIPPED,
)

__all__ = [
    # Config classes
    "EvalConfig",
    "ModelConfig",
    "TaskConfig",
    # Runner
    "EvaluationRunner",
    "EvalResult",
    # Secrets management
    "get_hf_token",
    "setup_hf_environment",
    "mask_token",
    # Constants
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DTYPE",
    "DEFAULT_GPUS",
    "DEFAULT_OUTPUT_DIR",
    "STATUS_SUCCESS",
    "STATUS_FAILED",
    "STATUS_SKIPPED",
    # CLI (lazy)
    "main",
]


def __getattr__(name: str):
    """Lazy import for CLI entry point to avoid module-order issues."""
    if name == "main":
        from .cli import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
