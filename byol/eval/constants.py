# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Constants for BYOL Evaluation Framework.

Centralises all magic numbers and default values for maintainability.
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

# =============================================================================
# Eval Package Directory (computed from this file's location)
# =============================================================================
EVAL_PACKAGE_DIR: _Path = _Path(__file__).resolve().parent
REPO_ROOT: _Path = EVAL_PACKAGE_DIR.parent.parent  # <repo>/byol/eval -> <repo>

# =============================================================================
# GPU Configuration
# =============================================================================
DEFAULT_GPUS: str = "0"

# =============================================================================
# Batch Size Defaults
# =============================================================================
DEFAULT_BATCH_SIZE: str = "auto:4"

# =============================================================================
# Model Configuration Defaults
# =============================================================================
DEFAULT_DTYPE: str = "bfloat16"
VALID_DTYPES: frozenset[str] = frozenset({"bfloat16", "float16", "float32", "auto"})
DEFAULT_MAX_LENGTH: int = 8192
DEFAULT_TRUST_REMOTE_CODE: bool = True

# =============================================================================
# Evaluation Defaults
# =============================================================================
DEFAULT_OUTPUT_DIR: str = "results/eval"
DEFAULT_JUDGE_OUTPUT_DIR: str = "results/eval/judge"
DEFAULT_LOG_SAMPLES: bool = False
DEFAULT_APPLY_CHAT_TEMPLATE: bool = False

# =============================================================================
# Status Codes
# =============================================================================
STATUS_SUCCESS: str = "success"
STATUS_FAILED: str = "failed"
STATUS_SKIPPED: str = "skipped"

STATUS_ICONS: dict[str, str] = {
    STATUS_SUCCESS: "✅",
    STATUS_FAILED: "❌",
    STATUS_SKIPPED: "⏭️",
}

# =============================================================================
# Model Types and Languages
# =============================================================================
VALID_TYPES: frozenset[str] = frozenset({"base", "instruct", "merged"})

# Known languages with pre-built eval configs.  The eval CLI accepts ANY
# language code; this mapping is only used for display names and hints.
KNOWN_LANGS: dict[str, str] = {
    "eng": "English",
    "mri": "Māori",
    "nya": "Chichewa",
    "gug": "Guaraní",
}
# Backward-compatible alias
VALID_LANGS: frozenset[str] = frozenset(KNOWN_LANGS)
LANG_NAMES: dict[str, str] = KNOWN_LANGS

# =============================================================================
# Config Paths
# =============================================================================
DEFAULT_CONFIGS_DIR: str = "configs/eval"
CONFIG_FILENAME_TEMPLATE: str = "benchmark_{type}_{lang}.yaml"

# =============================================================================
# Task Paths (resolved relative to the eval package directory)
# =============================================================================
DEFAULT_TASKS_PATH: str = str(EVAL_PACKAGE_DIR / "tasks")
DEFAULT_DATA_PATH: str = str(REPO_ROOT / "data" / "eval")
DEFAULT_PATCHES_PATH: str = str(EVAL_PACKAGE_DIR / "patches")
DEFAULT_DATASETS_PATH: str = _os.environ.get("BYOL_DATASETS_PATH", "datasets")
DEFAULT_SHARED_DATASETS_PATH: str = _os.environ.get("BYOL_SHARED_DATASETS_PATH", "datasets/shared")

# =============================================================================
# Unsafe Tasks (require code execution confirmation)
# =============================================================================
UNSAFE_TASKS: frozenset[str] = frozenset({
    "humaneval",
    "humaneval_instruct",
    "humaneval_plus",
    "mbpp",
    "mbpp_plus",
})
