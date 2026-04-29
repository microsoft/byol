# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for byol.train subpackage integration.

Tests cover:
1. Import smoke tests — every module imports without error
2. Path resolution tests — all default paths exist on disk
3. Config file existence and structure
4. Dataclass / validation tests — config objects reject invalid inputs
5. CLI argument parsing
6. CLI dry-run integration test
7. Constants sanity checks

Run with: conda run -n byol pytest tests/test_train/ -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

# =============================================================================
# Paths — resolved relative to the main BYOL repo root
# =============================================================================

REPO_DIR = Path(__file__).resolve().parent.parent.parent  # byol repo root
CONFIGS_DIR = REPO_DIR / "configs" / "train"
TRAIN_PKG_DIR = REPO_DIR / "byol" / "train"
CLI_MODULE = "byol.train"

EXPECTED_CONFIGS = [
    "cpt.yaml",
    "sft.yaml",
    "dpo.yaml",
    "merge.yaml",
]


# =============================================================================
# Helpers
# =============================================================================


def load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML config from configs/train/."""
    path = CONFIGS_DIR / name
    assert path.exists(), f"Config file missing: {path}"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# 1. Import Smoke Tests
# =============================================================================


class TestImportSmoke:
    """Verify the train package and its key classes are importable."""

    def test_import_train_package(self) -> None:
        import byol.train
        assert hasattr(byol.train, "__version__")

    def test_import_train_config(self) -> None:
        from byol.train.config import TrainConfig, LoraConfig, DatasetMixConfig
        assert TrainConfig is not None
        assert LoraConfig is not None
        assert DatasetMixConfig is not None

    def test_import_runner(self) -> None:
        from byol.train.runner import TrainingRunner, TrainResult
        assert TrainingRunner is not None
        assert TrainResult is not None

    def test_import_constants(self) -> None:
        from byol.train.constants import (
            DEFAULT_BATCH_SIZE,
            DEFAULT_EPOCHS,
            DEFAULT_LEARNING_RATE,
            DEFAULT_OUTPUT_DIR,
            REPO_ROOT,
            TRAIN_PACKAGE_DIR,
            SUPPORTED_STAGES,
        )
        assert DEFAULT_BATCH_SIZE == 4
        assert DEFAULT_EPOCHS == 3
        assert SUPPORTED_STAGES == ("pt", "cpt", "sft", "dpo")
        assert DEFAULT_OUTPUT_DIR == "results/train"

    def test_import_secrets(self) -> None:
        from byol.train.secrets import get_hf_token, setup_environment, mask_token
        assert callable(get_hf_token)
        assert callable(setup_environment)
        assert callable(mask_token)

    def test_import_merge(self) -> None:
        from byol.train.merge import MergeConfig, merge_lora
        assert MergeConfig is not None
        assert callable(merge_lora)

    def test_import_cli(self) -> None:
        from byol.train.cli import main, create_parser
        assert callable(main)
        assert callable(create_parser)

    def test_import_via_top_level(self) -> None:
        from byol.train import TrainConfig, TrainingRunner, TrainResult
        assert TrainConfig is not None
        assert TrainingRunner is not None
        assert TrainResult is not None


# =============================================================================
# 2. Config files exist
# =============================================================================


class TestConfigFilesExist:
    @pytest.mark.parametrize("filename", EXPECTED_CONFIGS)
    def test_config_exists(self, filename: str) -> None:
        assert (CONFIGS_DIR / filename).exists(), f"{filename} should exist"


# =============================================================================
# 3. Path resolution
# =============================================================================


class TestPathResolution:
    """Verify that paths computed by constants.py point to real directories."""

    def test_train_package_dir_exists(self) -> None:
        from byol.train.constants import TRAIN_PACKAGE_DIR
        assert TRAIN_PACKAGE_DIR.exists()
        assert TRAIN_PACKAGE_DIR.is_dir()

    def test_repo_root_has_pyproject(self) -> None:
        from byol.train.constants import REPO_ROOT
        assert (REPO_ROOT / "pyproject.toml").exists()

    def test_default_configs_dir_exists(self) -> None:
        from byol.train.constants import REPO_ROOT, DEFAULT_CONFIGS_DIR
        configs = REPO_ROOT / DEFAULT_CONFIGS_DIR
        assert configs.exists()
        assert configs.is_dir()

    def test_examples_deepspeed_exist(self) -> None:
        from byol.train.constants import DEFAULT_EXAMPLES_DIR
        examples = Path(DEFAULT_EXAMPLES_DIR) / "deepspeed"
        assert examples.exists()
        assert (examples / "ds_z2_config.json").exists()
        assert (examples / "ds_z3_config.json").exists()


# =============================================================================
# 4. Config structure & loading
# =============================================================================


class TestConfigStructure:
    """Verify YAML configs load correctly and have required fields."""

    @pytest.mark.parametrize("filename", ["cpt.yaml", "sft.yaml", "dpo.yaml"])
    def test_training_config_has_required_fields(self, filename: str) -> None:
        data = load_yaml(filename)
        assert "model_name_or_path" in data
        assert "output_dir" in data

    def test_merge_config_has_required_fields(self) -> None:
        data = load_yaml("merge.yaml")
        assert "model_name_or_path" in data
        assert "export_dir" in data or "output_dir" in data

    @pytest.mark.parametrize("filename", ["cpt.yaml", "sft.yaml", "dpo.yaml"])
    def test_deepspeed_paths_resolve(self, filename: str) -> None:
        """Verify deepspeed config paths in YAMLs point to real files."""
        data = load_yaml(filename)
        ds_path = data.get("deepspeed")
        if ds_path:
            # Path is relative to repo root
            resolved = REPO_DIR / ds_path
            assert resolved.exists(), f"DeepSpeed config not found: {resolved}"

    @pytest.mark.parametrize("filename", ["cpt.yaml", "sft.yaml", "dpo.yaml"])
    def test_output_dir_under_results(self, filename: str) -> None:
        """Ensure output dirs point to results/train/..."""
        data = load_yaml(filename)
        output_dir = data.get("output_dir", "")
        assert output_dir.startswith("results/train"), (
            f"{filename}: output_dir should start with results/train, got {output_dir}"
        )


# =============================================================================
# 5. Dataclass validation
# =============================================================================


class TestDataclassValidation:
    """Verify config dataclasses reject invalid inputs."""

    def test_train_config_requires_model(self) -> None:
        from byol.train.config import TrainConfig
        with pytest.raises(ValueError, match="model_name_or_path"):
            TrainConfig(model_name_or_path="")

    def test_train_config_rejects_bad_stage(self) -> None:
        from byol.train.config import TrainConfig
        with pytest.raises(ValueError, match="Invalid stage"):
            TrainConfig(model_name_or_path="test-model", stage="invalid")

    def test_train_config_rejects_negative_epochs(self) -> None:
        from byol.train.config import TrainConfig
        with pytest.raises(ValueError, match="epochs"):
            TrainConfig(model_name_or_path="test-model", epochs=-1)

    def test_train_config_rejects_negative_batch_size(self) -> None:
        from byol.train.config import TrainConfig
        with pytest.raises(ValueError, match="batch_size"):
            TrainConfig(model_name_or_path="test-model", batch_size=0)

    def test_train_config_rejects_negative_lr(self) -> None:
        from byol.train.config import TrainConfig
        with pytest.raises(ValueError, match="learning_rate"):
            TrainConfig(model_name_or_path="test-model", learning_rate=-0.01)

    def test_lora_config_rejects_bad_rank(self) -> None:
        from byol.train.config import LoraConfig
        with pytest.raises(ValueError, match="rank"):
            LoraConfig(rank=0)

    def test_lora_config_rejects_bad_dropout(self) -> None:
        from byol.train.config import LoraConfig
        with pytest.raises(ValueError, match="dropout"):
            LoraConfig(dropout=1.5)

    def test_lora_config_rejects_empty_targets(self) -> None:
        from byol.train.config import LoraConfig
        with pytest.raises(ValueError, match="target_modules"):
            LoraConfig(target_modules=[])

    def test_dataset_mix_rejects_bad_strategy(self) -> None:
        from byol.train.config import DatasetMixConfig
        with pytest.raises(ValueError, match="Invalid mix strategy"):
            DatasetMixConfig(strategy="bad")

    def test_dataset_mix_rejects_mismatched_probs(self) -> None:
        from byol.train.config import DatasetMixConfig
        with pytest.raises(ValueError, match="Probabilities length"):
            DatasetMixConfig(
                datasets=["a", "b"],
                probabilities=[0.5],
            )

    def test_dataset_mix_rejects_probs_not_summing_to_1(self) -> None:
        from byol.train.config import DatasetMixConfig
        with pytest.raises(ValueError, match="sum to 1.0"):
            DatasetMixConfig(
                datasets=["a", "b"],
                probabilities=[0.3, 0.3],
            )

    def test_train_config_valid(self) -> None:
        from byol.train.config import TrainConfig
        c = TrainConfig(model_name_or_path="test-model", stage="sft")
        assert c.model_name_or_path == "test-model"
        assert c.stage == "sft"

    def test_train_config_from_yaml_missing_file(self) -> None:
        from byol.train.config import TrainConfig
        with pytest.raises(FileNotFoundError):
            TrainConfig.from_yaml("/nonexistent/config.yaml")


# =============================================================================
# 6. Config round-trip (dict → TrainConfig → dict)
# =============================================================================


class TestConfigRoundTrip:
    """Verify config serialization works correctly."""

    def test_to_dict_roundtrip(self) -> None:
        from byol.train.config import TrainConfig
        c1 = TrainConfig(model_name_or_path="test-model", stage="sft", epochs=5)
        d = c1.to_dict()
        c2 = TrainConfig.from_dict(d)
        assert c2.model_name_or_path == "test-model"
        assert c2.epochs == 5

    def test_to_llamafactory_format(self) -> None:
        from byol.train.config import TrainConfig
        c = TrainConfig(model_name_or_path="test-model", stage="cpt")
        lf = c.to_llamafactory("/tmp/out")
        # cpt stage maps to 'pt'
        assert lf["stage"] == "pt"
        assert lf["do_train"] is True

    def test_lora_in_llamafactory_format(self) -> None:
        from byol.train.config import TrainConfig, LoraConfig
        c = TrainConfig(
            model_name_or_path="test-model",
            lora=LoraConfig(rank=32, alpha=64),
        )
        lf = c.to_llamafactory("/tmp/out")
        assert lf["finetuning_type"] == "lora"
        assert lf["lora_rank"] == 32
        assert lf["lora_alpha"] == 64


# =============================================================================
# 7. CLI argument parsing
# =============================================================================


class TestCLIParsing:
    """Verify CLI argument parser produces expected namespace."""

    def test_sft_basic(self) -> None:
        from byol.train.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(["sft", "--model", "test-model", "--dataset", "alpaca"])
        assert args.stage == "sft"
        assert args.model == "test-model"
        assert args.dataset == "alpaca"

    def test_cpt_with_device(self) -> None:
        from byol.train.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(["cpt", "--model", "m", "--device", "0,1,2,3"])
        assert args.gpus == "0,1,2,3"

    def test_gpus_alias_works(self) -> None:
        from byol.train.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(["sft", "--model", "m", "--gpus", "1,2"])
        assert args.gpus == "1,2"

    def test_lora_flag(self) -> None:
        from byol.train.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(["sft", "--model", "m", "--lora", "--lora-rank", "64"])
        assert args.lora is True
        assert args.lora_rank == 64

    def test_dry_run_flag(self) -> None:
        from byol.train.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(["sft", "--model", "m", "--dry-run"])
        assert args.dry_run is True

    def test_output_dir_default(self) -> None:
        from byol.train.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(["sft", "--model", "m"])
        # Default is None (sentinel); actual default applied during config build
        assert args.output_dir is None

    def test_output_dir_explicit(self) -> None:
        from byol.train.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(["sft", "--model", "m", "--output-dir", "my/output"])
        assert args.output_dir == "my/output"

    def test_merge_subcommand(self) -> None:
        from byol.train.cli import create_parser
        parser = create_parser()
        args = parser.parse_args([
            "merge", "--base-model", "base", "--adapter", "lora", "--output", "out"
        ])
        assert args.stage == "merge"
        assert args.base_model == "base"
        assert args.adapter == "lora"

    def test_override_parsing(self) -> None:
        from byol.train.cli import parse_overrides
        overrides = parse_overrides(["epochs=10", "lr=1e-5", "bf16=true", "name=my_run"])
        assert overrides["epochs"] == 10
        assert overrides["lr"] == 1e-5
        assert overrides["bf16"] is True
        assert overrides["name"] == "my_run"


# =============================================================================
# 8. CLI dry-run integration
# =============================================================================


class TestCLIDryRun:
    """End-to-end dry-run tests via subprocess."""

    def test_sft_dry_run(self) -> None:
        result = subprocess.run(
            [
                sys.executable, "-m", CLI_MODULE,
                "sft", "--model", "test-model", "--dataset", "alpaca", "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_DIR),
            timeout=30,
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"

    def test_cpt_dry_run(self) -> None:
        result = subprocess.run(
            [
                sys.executable, "-m", CLI_MODULE,
                "cpt", "--model", "test-model", "--dataset", "test_cpt", "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_DIR),
            timeout=30,
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"

    def test_merge_dry_run(self) -> None:
        result = subprocess.run(
            [
                sys.executable, "-m", CLI_MODULE,
                "merge",
                "--base-model", "test-base",
                "--adapter", "test-adapter",
                "--output", "/tmp/byol_test_merge_out",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_DIR),
            timeout=30,
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"

    def test_no_stage_prints_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", CLI_MODULE],
            capture_output=True,
            text=True,
            cwd=str(REPO_DIR),
            timeout=30,
        )
        assert result.returncode == 1


# =============================================================================
# 9. Secrets module
# =============================================================================


class TestSecrets:
    """Verify secrets helpers work correctly."""

    def test_mask_token_short(self) -> None:
        from byol.train.secrets import mask_token
        assert mask_token(None) == "<not set>"
        assert mask_token("") == "<not set>"

    def test_mask_token_long(self) -> None:
        from byol.train.secrets import mask_token
        masked = mask_token("hf_AbCdEfGhIjKlMnOp")
        assert masked.startswith("hf_A")
        assert masked.endswith("MnOp")
        assert "..." in masked

    def test_mask_token_very_short(self) -> None:
        from byol.train.secrets import mask_token
        masked = mask_token("ab")
        assert masked == "**"


# =============================================================================
# 10. Data file checks
# =============================================================================
