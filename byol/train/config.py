# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Configuration dataclasses for training, LoRA, and dataset mixing."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CUTOFF_LEN,
    DEFAULT_EPOCHS,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_LORA_TARGET_MODULES,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
    SUPPORTED_DTYPES,
    SUPPORTED_MIX_STRATEGIES,
    SUPPORTED_STAGES,
)


def _is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def _next_power_of_2(n: int) -> int:
    """Find the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


@dataclass
class LoraConfig:
    """LoRA (Low-Rank Adaptation) configuration."""

    rank: int = DEFAULT_LORA_RANK
    alpha: int = DEFAULT_LORA_ALPHA
    dropout: float = DEFAULT_LORA_DROPOUT
    target_modules: List[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES))

    def __post_init__(self) -> None:
        """Validate LoRA configuration after initialization."""
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout}")
        if not self.target_modules:
            raise ValueError("LoRA target_modules cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LlamaFactory config.

        Returns:
            Dictionary with LoRA parameters.
        """
        return {
            "lora_rank": self.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "lora_target": ",".join(self.target_modules),
        }


@dataclass
class DatasetMixConfig:
    """Dataset mixing configuration: concat, interleave_under, interleave_over."""

    datasets: List[str] = field(default_factory=list)
    strategy: str = "concat"
    probabilities: Optional[List[float]] = None

    def __post_init__(self) -> None:
        """Validate dataset mix configuration."""
        if self.strategy not in SUPPORTED_MIX_STRATEGIES:
            raise ValueError(
                f"Invalid mix strategy: {self.strategy}. "
                f"Must be one of {SUPPORTED_MIX_STRATEGIES}"
            )
        if self.probabilities is not None:
            if len(self.probabilities) != len(self.datasets):
                raise ValueError(
                    f"Probabilities length ({len(self.probabilities)}) must match "
                    f"datasets length ({len(self.datasets)})"
                )
            if abs(sum(self.probabilities) - 1.0) > 1e-6:
                raise ValueError(f"Probabilities must sum to 1.0, got {sum(self.probabilities)}")


@dataclass
class TrainConfig:
    """Main training configuration aggregating model, data, and optimization settings."""

    # Model
    model_name_or_path: str = ""
    stage: str = "sft"
    dataset: str = ""
    eval_dataset: Optional[str] = None  # Evaluation dataset (defaults to dataset if not set)
    output_dir: str = DEFAULT_OUTPUT_DIR

    # Training
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    learning_rate: float = DEFAULT_LEARNING_RATE
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    weight_decay: float = DEFAULT_WEIGHT_DECAY

    # Data
    cutoff_len: int = DEFAULT_CUTOFF_LEN
    template: str = "gemma"

    # Hardware
    gpus: str = "0"
    bf16: bool = True

    # Optional components
    lora: Optional[LoraConfig] = None
    dataset_mix: Optional[DatasetMixConfig] = None

    # Logging
    wandb_project: Optional[str] = None

    # Passthrough: Extra YAML fields passed directly to LlamaFactory
    passthrough_args: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate training configuration."""
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path is required")
        if self.stage not in SUPPORTED_STAGES:
            raise ValueError(
                f"Invalid stage: {self.stage}. Must be one of {SUPPORTED_STAGES}"
            )
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            TrainConfig instance.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        """Create configuration from dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            TrainConfig instance.
        """
        lora_data = data.get("lora")
        lora = LoraConfig(**lora_data) if lora_data else None

        mix_data = data.get("dataset_mix")
        dataset_mix = DatasetMixConfig(**mix_data) if mix_data else None

        # Capture all extra YAML fields not handled by TrainConfig
        known_keys = {
            "model_name_or_path", "stage", "dataset", "eval_dataset", "output_dir", "epochs",
            "batch_size", "gradient_accumulation_steps", "learning_rate",
            "warmup_ratio", "max_grad_norm", "weight_decay", "cutoff_len",
            "template", "gpus", "bf16", "lora", "dataset_mix", "wandb_project",
            # Also ignore these as they map to other fields
            "num_train_epochs", "per_device_train_batch_size",
        }
        passthrough_args = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            model_name_or_path=data.get("model_name_or_path", ""),
            stage=data.get("stage", "sft"),
            dataset=data.get("dataset", ""),
            output_dir=data.get("output_dir", DEFAULT_OUTPUT_DIR),
            # Accept both TrainConfig names and LlamaFactory-native names
            epochs=data.get("epochs") or data.get("num_train_epochs", DEFAULT_EPOCHS),
            batch_size=data.get("batch_size") or data.get("per_device_train_batch_size", DEFAULT_BATCH_SIZE),
            gradient_accumulation_steps=data.get(
                "gradient_accumulation_steps", DEFAULT_GRADIENT_ACCUMULATION_STEPS
            ),
            learning_rate=data.get("learning_rate", DEFAULT_LEARNING_RATE),
            warmup_ratio=data.get("warmup_ratio", DEFAULT_WARMUP_RATIO),
            max_grad_norm=data.get("max_grad_norm", DEFAULT_MAX_GRAD_NORM),
            weight_decay=data.get("weight_decay", DEFAULT_WEIGHT_DECAY),
            cutoff_len=data.get("cutoff_len", DEFAULT_CUTOFF_LEN),
            template=data.get("template", "gemma"),
            gpus=str(data.get("gpus", "0")),
            bf16=data.get("bf16", True),
            lora=lora,
            dataset_mix=dataset_mix,
            eval_dataset=data.get("eval_dataset"),
            wandb_project=data.get("wandb_project"),
            passthrough_args=passthrough_args if passthrough_args else None,
        )

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save the YAML file.
        """
        data = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all configuration values.
        """
        data = {
            "model_name_or_path": self.model_name_or_path,
            "stage": self.stage,
            "dataset": self.dataset,
            "output_dir": self.output_dir,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "cutoff_len": self.cutoff_len,
            "template": self.template,
            "gpus": self.gpus,
            "bf16": self.bf16,
            "wandb_project": self.wandb_project,
        }
        if self.lora:
            data["lora"] = {
                "rank": self.lora.rank,
                "alpha": self.lora.alpha,
                "dropout": self.lora.dropout,
                "target_modules": self.lora.target_modules,
            }
        if self.dataset_mix:
            data["dataset_mix"] = {
                "datasets": self.dataset_mix.datasets,
                "strategy": self.dataset_mix.strategy,
                "probabilities": self.dataset_mix.probabilities,
            }
        return data

    def to_llamafactory(self, output_dir: str) -> Dict[str, Any]:
        """Convert to LlamaFactory configuration format.

        Args:
            output_dir: Output directory for this training run.

        Returns:
            Dictionary in LlamaFactory config format.
        """
        from .constants import DEFAULT_DATA_DIR

        config: Dict[str, Any] = {
            "model_name_or_path": self.model_name_or_path,
            "stage": "pt" if self.stage == "cpt" else self.stage,
            "do_train": True,
            "dataset": self.dataset,
            "dataset_dir": DEFAULT_DATA_DIR,
            "template": self.template,
            "output_dir": output_dir,
            "num_train_epochs": self.epochs,
            "per_device_train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "cutoff_len": self.cutoff_len,
            "bf16": self.bf16,
            "logging_steps": 10,
            "save_steps": 500,
            "save_total_limit": 3,
        }

        # Add eval_dataset if specified
        if self.eval_dataset:
            config["eval_dataset"] = self.eval_dataset
            config["do_eval"] = True

        # Add LoRA config
        if self.lora:
            config["finetuning_type"] = "lora"
            config.update(self.lora.to_dict())
        else:
            config["finetuning_type"] = "full"

        # Add dataset mixing
        if self.dataset_mix and self.dataset_mix.datasets:
            config["dataset"] = ",".join(self.dataset_mix.datasets)
            config["mix_strategy"] = self.dataset_mix.strategy
            if self.dataset_mix.probabilities:
                config["dataset_probs"] = ",".join(
                    str(p) for p in self.dataset_mix.probabilities
                )

        # Add W&B logging
        if self.wandb_project:
            config["report_to"] = "wandb"
            config["run_name"] = f"{Path(self.model_name_or_path).name}_{self.stage}"

        # Add passthrough args (extra YAML fields passed directly to LlamaFactory)
        if self.passthrough_args:
            passthrough = self.passthrough_args.copy()

            # Expand ~ in dataset_dir so LlamaFactory receives an absolute path
            if "dataset_dir" in passthrough and passthrough["dataset_dir"]:
                passthrough["dataset_dir"] = str(
                    Path(passthrough["dataset_dir"]).expanduser().resolve()
                )

            # Handle deepspeed: strip if empty, check if installed, auto-set env var
            if "deepspeed" in passthrough:
                ds_val = passthrough["deepspeed"]
                if not ds_val:
                    # Empty/None — user wants no deepspeed, remove the key entirely
                    del passthrough["deepspeed"]
                else:
                    import importlib.util
                    if importlib.util.find_spec("deepspeed") is None:
                        print(
                            "\033[33m⚠  DeepSpeed is specified in config but not "
                            "installed in the current environment.\n"
                            "   Training will continue without DeepSpeed "
                            "(single-GPU / standard DDP).\n"
                            "   To enable it: pip install deepspeed\033[0m"
                        )
                        del passthrough["deepspeed"]
                    else:
                        # Auto-set FORCE_TORCHRUN so LlamaFactory uses torchrun launcher
                        os.environ.setdefault("FORCE_TORCHRUN", "1")
                        # Resolve relative paths for deepspeed config
                        ds_path = passthrough["deepspeed"]
                        if not os.path.isabs(ds_path):
                            if os.path.exists(ds_path):
                                passthrough["deepspeed"] = str(
                                    Path(ds_path).resolve()
                                )
                            else:
                                from .constants import TRAIN_PACKAGE_DIR, REPO_ROOT
                                resolved = TRAIN_PACKAGE_DIR / ds_path
                                if resolved.exists():
                                    passthrough["deepspeed"] = str(resolved)
                                else:
                                    resolved = REPO_ROOT / ds_path
                                    if resolved.exists():
                                        passthrough["deepspeed"] = str(resolved)

            config.update(passthrough)

        return config
