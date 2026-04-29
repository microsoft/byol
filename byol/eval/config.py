# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Configuration classes for BYOL Evaluation Framework.

Provides :class:`EvalConfig`, :class:`ModelConfig`, :class:`TaskConfig` and
:class:`TaskRegistry` for describing lm-evaluation-harness runs.  All classes
are plain ``@dataclass`` objects backed by YAML I/O.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .constants import (
    DEFAULT_APPLY_CHAT_TEMPLATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONFIGS_DIR,
    DEFAULT_DTYPE,
    DEFAULT_GPUS,
    DEFAULT_LOG_SAMPLES,
    DEFAULT_MAX_LENGTH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRUST_REMOTE_CODE,
    EVAL_PACKAGE_DIR,
    VALID_DTYPES,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Model Configuration
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for a single model to evaluate.

    Attributes:
        name: Human-readable model identifier.
        path: Local path or HuggingFace model ID.
        dtype: Data type for model weights (bfloat16, float16, float32, auto).
        trust_remote_code: Whether to trust remote code from HuggingFace.
        max_length: Maximum sequence length for evaluation.
    """

    name: str
    path: str
    dtype: str = DEFAULT_DTYPE
    trust_remote_code: bool = DEFAULT_TRUST_REMOTE_CODE
    max_length: int = DEFAULT_MAX_LENGTH

    def __post_init__(self) -> None:
        """Validate configuration after initialisation."""
        if not self.path:
            raise ValueError("Model path cannot be empty")
        if self.dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid dtype: {self.dtype}. Must be one of {VALID_DTYPES}")

        # Resolve local paths (skip HuggingFace IDs)
        if "/" in self.path and not self.path.startswith(("http://", "https://")):
            resolved = Path(self.path).expanduser()
            if resolved.exists():
                self.path = str(resolved.resolve())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Create :class:`ModelConfig` from a plain dictionary.

        Args:
            data: Dictionary containing model configuration.

        Returns:
            Configured ``ModelConfig`` instance.

        Raises:
            KeyError: If required ``path`` key is missing.
        """
        return cls(
            name=data.get("name", "model"),
            path=data["path"],
            dtype=data.get("dtype", DEFAULT_DTYPE),
            trust_remote_code=data.get("trust_remote_code", DEFAULT_TRUST_REMOTE_CODE),
            max_length=data.get("max_length", DEFAULT_MAX_LENGTH),
        )


# =============================================================================
# Task Configuration
# =============================================================================


@dataclass
class TaskConfig:
    """Configuration for an evaluation task.

    Attributes:
        name: Task name or comma-separated task names.
        num_fewshot: Number of few-shot examples (``None`` uses the task default).
        limit: Maximum samples to evaluate (``None`` for all).
        batch_size: Batch size override for this task.
        apply_chat_template: Whether to apply chat template for this task.
    """

    name: str
    num_fewshot: int | None = None
    limit: int | None = None
    batch_size: str | None = None
    apply_chat_template: bool = DEFAULT_APPLY_CHAT_TEMPLATE

    def __post_init__(self) -> None:
        """Validate configuration after initialisation."""
        if not self.name:
            raise ValueError("Task name cannot be empty")
        if self.num_fewshot is not None and self.num_fewshot < 0:
            raise ValueError("num_fewshot must be non-negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskConfig:
        """Create :class:`TaskConfig` from a plain dictionary.

        Args:
            data: Dictionary containing task configuration.

        Returns:
            Configured ``TaskConfig`` instance.

        Raises:
            KeyError: If required ``name`` key is missing.
        """
        return cls(
            name=data["name"],
            num_fewshot=data.get("num_fewshot"),
            limit=data.get("limit"),
            batch_size=data.get("batch_size"),
            apply_chat_template=data.get("apply_chat_template", DEFAULT_APPLY_CHAT_TEMPLATE),
        )


# =============================================================================
# Main Evaluation Configuration
# =============================================================================


@dataclass
class EvalConfig:
    """Top-level evaluation configuration loaded from YAML.

    Attributes:
        models: List of models to evaluate.
        tasks: List of evaluation tasks.
        output_dir: Directory for evaluation results.
        tasks_path: Path to custom task definitions.
        gpus: Comma-separated GPU device IDs.
        batch_size: Default batch size for all tasks.
        log_samples: Whether to log evaluation samples.
        apply_chat_template: Global chat template setting.
        hf_token: HuggingFace API token.
    """

    models: list[ModelConfig] = field(default_factory=list)
    tasks: list[TaskConfig] = field(default_factory=list)
    output_dir: str = DEFAULT_OUTPUT_DIR
    tasks_path: str | None = None
    gpus: str = DEFAULT_GPUS
    batch_size: str = DEFAULT_BATCH_SIZE
    log_samples: bool = DEFAULT_LOG_SAMPLES
    apply_chat_template: bool = DEFAULT_APPLY_CHAT_TEMPLATE
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))

    def __post_init__(self) -> None:
        """Validate configuration and create output directory."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if self.tasks_path:
            tasks_path = Path(self.tasks_path).expanduser()
            if not tasks_path.is_absolute():
                # Try CWD first, then resolve relative to the eval package dir
                cwd_path = (Path.cwd() / tasks_path).resolve()
                pkg_path = (EVAL_PACKAGE_DIR / tasks_path).resolve()
                if cwd_path.exists():
                    tasks_path = cwd_path
                elif pkg_path.exists():
                    tasks_path = pkg_path
                else:
                    tasks_path = tasks_path.resolve()
            else:
                tasks_path = tasks_path.resolve()
            if not tasks_path.exists():
                raise ValueError(f"Tasks path does not exist: {self.tasks_path}")
            self.tasks_path = str(tasks_path)

    # --------------------------------------------------------------------- #
    # YAML I/O
    # --------------------------------------------------------------------- #

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Configured ``EvalConfig`` instance.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
        """
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalConfig:
        """Create :class:`EvalConfig` from a plain dictionary.

        Args:
            data: Dictionary containing evaluation configuration.

        Returns:
            Configured ``EvalConfig`` instance.
        """
        models = [ModelConfig.from_dict(m) for m in data.get("models", [])]
        tasks = [
            TaskConfig.from_dict(t) for t in data.get("tasks", []) if t.get("enabled", True)
        ]

        eval_settings = data.get("evaluation", {})
        lm_eval_settings = data.get("lm_eval", {})

        global_chat_template = eval_settings.get(
            "apply_chat_template", DEFAULT_APPLY_CHAT_TEMPLATE
        )

        return cls(
            models=models,
            tasks=tasks,
            output_dir=eval_settings.get("results_dir", DEFAULT_OUTPUT_DIR),
            tasks_path=lm_eval_settings.get("include_path"),
            gpus=str(eval_settings.get("gpus", DEFAULT_GPUS)),
            batch_size=str(eval_settings.get("batch_size", DEFAULT_BATCH_SIZE)),
            log_samples=lm_eval_settings.get("log_samples", DEFAULT_LOG_SAMPLES),
            apply_chat_template=global_chat_template,
        )

    def to_yaml(self, path: str | Path) -> None:
        """Serialise configuration to a YAML file.

        Args:
            path: Destination file path.
        """
        data = {
            "evaluation": {
                "results_dir": self.output_dir,
                "gpus": self.gpus,
                "batch_size": self.batch_size,
                "apply_chat_template": self.apply_chat_template,
            },
            "models": [
                {"name": m.name, "path": m.path, "dtype": m.dtype} for m in self.models
            ],
            "lm_eval": {
                "include_path": self.tasks_path,
                "log_samples": self.log_samples,
            },
            "tasks": [
                {
                    "name": t.name,
                    "num_fewshot": t.num_fewshot,
                    "apply_chat_template": t.apply_chat_template,
                    "enabled": True,
                }
                for t in self.tasks
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Task Registry
# =============================================================================


class TaskRegistry:
    """Registry that resolves task settings from benchmark config files.

    Scans all ``benchmark_*.yaml`` files in the configs directory and builds
    a lookup table mapping individual task names to their configured
    :class:`TaskConfig` (num_fewshot, batch_size, apply_chat_template, etc.).

    When a user runs ``byol-eval --tasks copa`` without ``--config``, the
    registry supplies the canonical settings so the user doesn't have to
    remember or re-specify them.
    """

    def __init__(self, configs_dir: str | Path | None = None) -> None:
        if configs_dir is not None:
            self._configs_dir = Path(configs_dir).expanduser().resolve()
        else:
            self._configs_dir = self._find_configs_dir()
        self._registry: dict[str, TaskConfig] = {}
        self._loaded: bool = False

    # ------------------------------------------------------------------ #
    # Config directory resolution
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_configs_dir() -> Path:
        """Locate the configs directory by walking up from the package."""
        pkg_dir = Path(__file__).resolve().parent  # byol/eval/
        candidates = [
            pkg_dir.parent.parent / DEFAULT_CONFIGS_DIR,  # repo/configs/eval/
            pkg_dir.parent.parent / "configs" / "eval",   # alternate
            Path.cwd() / DEFAULT_CONFIGS_DIR,              # cwd/configs/eval/
            Path.cwd() / "configs",                        # cwd/configs/ (flat)
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return pkg_dir.parent.parent / DEFAULT_CONFIGS_DIR  # best guess

    # ------------------------------------------------------------------ #
    # Lazy loading
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        """Lazy-load all benchmark configs on first access."""
        if self._loaded:
            return
        self._loaded = True
        if not self._configs_dir.is_dir():
            logger.debug(f"Configs directory not found: {self._configs_dir}")
            return
        for yaml_path in sorted(self._configs_dir.glob("benchmark_*.yaml")):
            self._load_config(yaml_path)
        if self._registry:
            logger.debug(
                f"TaskRegistry loaded {len(self._registry)} tasks from {self._configs_dir}"
            )

    def _load_config(self, path: Path) -> None:
        """Parse a single benchmark YAML and register its tasks."""
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data or "tasks" not in data:
                return
            for task_data in data["tasks"]:
                if not task_data.get("enabled", True):
                    continue
                if "name" not in task_data:
                    continue
                raw_name = task_data["name"]
                task_cfg = TaskConfig.from_dict(task_data)
                for individual_name in raw_name.split(","):
                    individual_name = individual_name.strip()
                    if individual_name and individual_name not in self._registry:
                        self._registry[individual_name] = TaskConfig(
                            name=individual_name,
                            num_fewshot=task_cfg.num_fewshot,
                            limit=task_cfg.limit,
                            batch_size=task_cfg.batch_size,
                            apply_chat_template=task_cfg.apply_chat_template,
                        )
        except Exception as exc:
            logger.debug(f"Failed to parse {path}: {exc}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def lookup(self, task_name: str) -> TaskConfig | None:
        """Look up settings for a single task name.

        Args:
            task_name: Exact task name (not comma-separated).

        Returns:
            ``TaskConfig`` with the registered settings, or ``None`` if unknown.
        """
        self._ensure_loaded()
        return self._registry.get(task_name)

    def resolve_tasks(
        self,
        task_names: str,
        cli_num_fewshot: int | None = None,
        cli_limit: int | None = None,
    ) -> list[TaskConfig]:
        """Resolve comma-separated task names into :class:`TaskConfig` objects.

        For each task name:
            1. If *cli_num_fewshot* is explicitly set, use it (user override).
            2. Else if the task is found in the registry, use registered settings.
            3. Else leave ``num_fewshot=None`` (let lm-eval use its default).

        Tasks that share the same (num_fewshot, batch_size, apply_chat_template)
        are grouped into a single ``TaskConfig`` with comma-separated names for
        efficiency.

        Args:
            task_names: Comma-separated task names from the CLI.
            cli_num_fewshot: Explicit ``--num-fewshot`` value (``None`` if unset).
            cli_limit: Explicit ``--limit`` value.

        Returns:
            List of ``TaskConfig`` objects grouped by compatible settings.
        """
        self._ensure_loaded()
        individual_names = [n.strip() for n in task_names.split(",") if n.strip()]

        resolved: list[TaskConfig] = []
        for name in individual_names:
            registered = self._registry.get(name)
            if cli_num_fewshot is not None:
                fewshot = cli_num_fewshot
                batch_size = registered.batch_size if registered else None
                chat_template = registered.apply_chat_template if registered else False
            elif registered is not None:
                fewshot = registered.num_fewshot
                batch_size = registered.batch_size
                chat_template = registered.apply_chat_template
                logger.info(
                    f"Task '{name}': using registered settings "
                    f"(num_fewshot={fewshot}, batch_size={batch_size})"
                )
            else:
                fewshot = None
                batch_size = None
                chat_template = False
                logger.info(
                    f"Task '{name}': no registered settings found, using lm-eval defaults"
                )
            resolved.append(
                TaskConfig(
                    name=name,
                    num_fewshot=fewshot,
                    limit=cli_limit,
                    batch_size=batch_size,
                    apply_chat_template=chat_template,
                )
            )

        # Group tasks with identical settings
        groups: dict[tuple, list[str]] = {}
        settings_map: dict[tuple, TaskConfig] = {}
        for tc in resolved:
            key = (tc.num_fewshot, tc.batch_size, tc.apply_chat_template)
            groups.setdefault(key, []).append(tc.name)
            settings_map.setdefault(key, tc)

        grouped: list[TaskConfig] = []
        for key, names in groups.items():
            ref = settings_map[key]
            grouped.append(
                TaskConfig(
                    name=",".join(names),
                    num_fewshot=ref.num_fewshot,
                    limit=ref.limit,
                    batch_size=ref.batch_size,
                    apply_chat_template=ref.apply_chat_template,
                )
            )
        return grouped


# =============================================================================
# Module-level singleton
# =============================================================================

_default_registry: TaskRegistry | None = None


def get_task_registry() -> TaskRegistry:
    """Return the module-level :class:`TaskRegistry` singleton (lazy-initialised)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = TaskRegistry()
    return _default_registry
