# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for byol.eval subpackage integration.

Tests cover:
1. Import smoke tests
2. Config file existence and structure  
3. Config language purity (each config has only tasks for its language)
4. Base vs instruct settings correctness
5. No tasks lost across configs
6. CLI argument parsing and dry-run
7. TaskRegistry lookup
8. Constants and path resolution
9. Data and task file existence

Run with: conda run -n byol pytest tests/test_eval/ -v
"""

from __future__ import annotations

import os
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
CONFIGS_DIR = REPO_DIR / "configs" / "eval"
EVAL_PKG_DIR = REPO_DIR / "byol" / "eval"
CLI_MODULE = "byol.eval"

EXPECTED_CONFIGS = [
    "benchmark_base_eng.yaml",
    "benchmark_base_mri.yaml",
    "benchmark_base_nya.yaml",
    "benchmark_instruct_eng.yaml",
    "benchmark_instruct_mri.yaml",
    "benchmark_instruct_nya.yaml",
]


# =============================================================================
# Helpers
# =============================================================================


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIGS_DIR / name
    assert path.exists(), f"Config file missing: {path}"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_task_names(data: dict[str, Any]) -> list[str]:
    """Extract all individual task names from a config dict."""
    names: list[str] = []
    for t in data.get("tasks", []):
        if "name" not in t:
            continue
        for n in t["name"].split(","):
            n = n.strip()
            if n:
                names.append(n)
    return names


# =============================================================================
# Classification helpers
# =============================================================================

MRI_SUFFIXES = ("_mri", "_mri_Latn", "_en_mri", "_mri_en")
NYA_SUFFIXES = ("_ny", "_nya", "_nya_Latn", "_en_ny", "_ny_en")

ENGLISH_ONLY_TASKS = {
    "winogrande", "copa", "xnli_en", "belebele_eng_Latn",
    "mgsm_direct_en", "hellaswag", "arc_challenge", "arc_easy",
    "piqa", "xstorycloze_en", "gpqa_diamond_n_shot", "gpqa_diamond_zeroshot",
    "bbh_fewshot", "bbh_zeroshot", "ifeval", "humaneval_instruct",
    "arc_challenge_chat", "toxigen", "bbq",
    "realtoxicityprompts", "realtoxicitypromptsllama",
}


def is_mri_task(name: str) -> bool:
    return any(name.endswith(s) for s in MRI_SUFFIXES) or name in {"flores_en_mri", "flores_mri_en"} or "_mri" in name


def is_nya_task(name: str) -> bool:
    return any(name.endswith(s) for s in NYA_SUFFIXES) or name in {"flores_en_ny", "flores_ny_en"} or "_nya" in name or "_ny" in name


def is_english_task(name: str) -> bool:
    if name.startswith("flores_") and name.endswith("_en"):
        parts = name.split("_")
        if len(parts) == 3 and parts[1] not in ("en",):
            return False
    if name in ENGLISH_ONLY_TASKS:
        return True
    if name.startswith("global_mmlu_en"):
        return True
    if name.endswith("_en"):
        return True
    if "eng_" in name:
        return True
    return False


# =============================================================================
# 1. Import Smoke Tests
# =============================================================================


class TestImportSmoke:
    """Verify the eval package and its key classes are importable."""

    def test_import_eval_package(self) -> None:
        import byol.eval
        assert hasattr(byol.eval, "__version__")

    def test_import_eval_config(self) -> None:
        from byol.eval.config import EvalConfig, ModelConfig, TaskConfig
        assert EvalConfig is not None
        assert ModelConfig is not None
        assert TaskConfig is not None

    def test_import_runner(self) -> None:
        from byol.eval.runner import EvaluationRunner, EvalResult
        assert EvaluationRunner is not None
        assert EvalResult is not None

    def test_import_constants(self) -> None:
        from byol.eval.constants import (
            DEFAULT_BATCH_SIZE,
            DEFAULT_DTYPE,
            DEFAULT_GPUS,
            EVAL_PACKAGE_DIR,
            VALID_TYPES,
            VALID_LANGS,
        )
        assert DEFAULT_BATCH_SIZE == "auto:4"
        assert DEFAULT_DTYPE == "bfloat16"
        assert VALID_TYPES == frozenset({"base", "instruct"})
        assert VALID_LANGS == frozenset({"eng", "mri", "nya", "gug"})

    def test_import_secrets(self) -> None:
        from byol.eval.secrets import get_hf_token, setup_hf_environment, mask_token
        assert callable(get_hf_token)
        assert callable(setup_hf_environment)
        assert callable(mask_token)

    def test_import_cli(self) -> None:
        from byol.eval.cli import main, parse_args
        assert callable(main)
        assert callable(parse_args)

    def test_import_via_top_level(self) -> None:
        from byol.eval import EvalConfig, EvaluationRunner, EvalResult
        assert EvalConfig is not None
        assert EvaluationRunner is not None
        assert EvalResult is not None


# =============================================================================
# 2. Config files exist
# =============================================================================


class TestConfigFilesExist:
    @pytest.mark.parametrize("filename", EXPECTED_CONFIGS)
    def test_config_exists(self, filename: str) -> None:
        assert (CONFIGS_DIR / filename).exists(), f"{filename} should exist"

    def test_safety_config_exists(self) -> None:
        assert (CONFIGS_DIR / "benchmark_safety.yaml").exists()


# =============================================================================
# 3. Constants and path resolution
# =============================================================================


class TestPathResolution:
    """Verify that paths computed by constants.py point to real directories."""

    def test_eval_package_dir_exists(self) -> None:
        from byol.eval.constants import EVAL_PACKAGE_DIR
        assert EVAL_PACKAGE_DIR.exists()
        assert EVAL_PACKAGE_DIR.is_dir()

    def test_default_tasks_path_exists(self) -> None:
        from byol.eval.constants import DEFAULT_TASKS_PATH
        assert Path(DEFAULT_TASKS_PATH).exists()
        assert Path(DEFAULT_TASKS_PATH).is_dir()

    def test_default_data_path_exists(self) -> None:
        from byol.eval.constants import DEFAULT_DATA_PATH
        assert Path(DEFAULT_DATA_PATH).exists()
        assert Path(DEFAULT_DATA_PATH).is_dir()

    def test_default_patches_path_exists(self) -> None:
        from byol.eval.constants import DEFAULT_PATCHES_PATH
        assert Path(DEFAULT_PATCHES_PATH).exists()
        assert Path(DEFAULT_PATCHES_PATH).is_dir()

    def test_configs_dir_exists(self) -> None:
        assert CONFIGS_DIR.exists()
        assert CONFIGS_DIR.is_dir()


# =============================================================================
# 4. Data and task files
# =============================================================================


class TestDataAndTaskFiles:
    """Verify that eval data and task definition files exist."""

    def test_chichewa_data_dir_has_files(self) -> None:
        data_dir = REPO_DIR / "data" / "nya" / "eval"
        assert data_dir.exists(), "data/nya/eval/ should exist"
        jsonl_files = list(data_dir.glob("*.jsonl"))
        assert len(jsonl_files) > 0, "chichewa data should contain .jsonl files"

    def test_maori_data_dir_has_files(self) -> None:
        data_dir = REPO_DIR / "data" / "mri" / "eval"
        assert data_dir.exists(), "data/mri/eval/ should exist"
        jsonl_files = list(data_dir.glob("*.jsonl"))
        assert len(jsonl_files) > 0, "maori data should contain .jsonl files"

    def test_task_dirs_exist(self) -> None:
        tasks_dir = EVAL_PKG_DIR / "tasks"
        expected_tasks = ["arc", "flores", "hellaswag", "xcopa", "xnli", "piqa", "xwinograd"]
        for task in expected_tasks:
            assert (tasks_dir / task).exists(), f"tasks/{task}/ should exist"

    def test_task_yamls_exist(self) -> None:
        tasks_dir = EVAL_PKG_DIR / "tasks"
        yaml_files = list(tasks_dir.rglob("*.yaml"))
        assert len(yaml_files) > 10, f"Should have >10 task YAML files, found {len(yaml_files)}"

    def test_patches_dir_has_apply_script(self) -> None:
        patches_dir = EVAL_PKG_DIR / "patches"
        assert (patches_dir / "apply_lmeval_patches.py").exists()


# =============================================================================
# 5. Config language purity
# =============================================================================


class TestLanguagePurity:
    """Each config should ONLY contain tasks for its designated language."""

    def test_base_eng_has_no_mri_or_nya(self) -> None:
        names = get_task_names(load_yaml("benchmark_base_eng.yaml"))
        assert len(names) > 0
        for name in names:
            assert not is_mri_task(name), f"English config has Māori task: {name}"
            assert not is_nya_task(name), f"English config has Chichewa task: {name}"

    def test_base_mri_has_no_eng_or_nya(self) -> None:
        names = get_task_names(load_yaml("benchmark_base_mri.yaml"))
        assert len(names) > 0
        for name in names:
            assert not is_english_task(name), f"Māori config has English task: {name}"
            assert not is_nya_task(name), f"Māori config has Chichewa task: {name}"

    def test_base_nya_has_no_eng_or_mri(self) -> None:
        names = get_task_names(load_yaml("benchmark_base_nya.yaml"))
        assert len(names) > 0
        for name in names:
            assert not is_english_task(name), f"Chichewa config has English task: {name}"
            assert not is_mri_task(name), f"Chichewa config has Māori task: {name}"

    def test_instruct_eng_has_no_mri_or_nya(self) -> None:
        names = get_task_names(load_yaml("benchmark_instruct_eng.yaml"))
        assert len(names) > 0
        for name in names:
            assert not is_mri_task(name), f"English instruct has Māori task: {name}"
            assert not is_nya_task(name), f"English instruct has Chichewa task: {name}"

    def test_instruct_mri_has_no_eng_or_nya(self) -> None:
        names = get_task_names(load_yaml("benchmark_instruct_mri.yaml"))
        assert len(names) > 0
        for name in names:
            assert not is_english_task(name), f"Māori instruct has English task: {name}"
            assert not is_nya_task(name), f"Māori instruct has Chichewa task: {name}"

    def test_instruct_nya_has_no_eng_or_mri(self) -> None:
        names = get_task_names(load_yaml("benchmark_instruct_nya.yaml"))
        assert len(names) > 0
        for name in names:
            assert not is_english_task(name), f"Chichewa instruct has English task: {name}"
            assert not is_mri_task(name), f"Chichewa instruct has Māori task: {name}"


# =============================================================================
# 6. Base vs instruct settings
# =============================================================================


class TestBaseVsInstruct:
    """Base configs use few-shot prompting; instruct configs use 0-shot + chat template."""

    def test_base_configs_no_chat_template(self) -> None:
        for name in ["benchmark_base_eng.yaml", "benchmark_base_mri.yaml", "benchmark_base_nya.yaml"]:
            data = load_yaml(name)
            eval_section = data.get("evaluation", {})
            assert not eval_section.get("apply_chat_template", False), \
                f"{name} should NOT have apply_chat_template=true"

    def test_instruct_configs_have_chat_template(self) -> None:
        for name in ["benchmark_instruct_eng.yaml", "benchmark_instruct_mri.yaml", "benchmark_instruct_nya.yaml"]:
            data = load_yaml(name)
            eval_section = data.get("evaluation", {})
            assert eval_section.get("apply_chat_template", False), \
                f"{name} should have apply_chat_template=true"

    def test_base_configs_have_fewshot_values(self) -> None:
        for name in ["benchmark_base_eng.yaml", "benchmark_base_mri.yaml", "benchmark_base_nya.yaml"]:
            data = load_yaml(name)
            fewshot_values = [t.get("num_fewshot", 0) for t in data["tasks"] if t.get("enabled", True)]
            assert any(v > 0 for v in fewshot_values if v is not None), \
                f"{name} should have some tasks with fewshot > 0"


# =============================================================================
# 7. No tasks lost
# =============================================================================


class TestNoTasksLost:
    def test_base_mri_plus_eng_covers_enough_tasks(self) -> None:
        eng = set(get_task_names(load_yaml("benchmark_base_eng.yaml")))
        mri = set(get_task_names(load_yaml("benchmark_base_mri.yaml")))
        combined = eng | mri
        assert len(combined) >= 20, f"Combined eng+mri should have >=20 tasks, got {len(combined)}"

    def test_instruct_mri_plus_eng_covers_enough_tasks(self) -> None:
        eng = set(get_task_names(load_yaml("benchmark_instruct_eng.yaml")))
        mri = set(get_task_names(load_yaml("benchmark_instruct_mri.yaml")))
        combined = eng | mri
        assert len(combined) >= 20, f"Combined eng+mri should have >=20 tasks, got {len(combined)}"


# =============================================================================
# 8. ModelConfig validation
# =============================================================================


class TestModelConfig:
    def test_valid_model_config(self) -> None:
        from byol.eval.config import ModelConfig
        mc = ModelConfig(name="test_model", path="google/gemma-3-4b-pt")
        assert mc.name == "test_model"
        assert mc.dtype == "bfloat16"

    def test_invalid_dtype_raises(self) -> None:
        from byol.eval.config import ModelConfig
        with pytest.raises(ValueError, match="Invalid dtype"):
            ModelConfig(name="test", path="some/model", dtype="invalid")

    def test_empty_path_raises(self) -> None:
        from byol.eval.config import ModelConfig
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelConfig(name="test", path="")

    def test_model_from_dict(self) -> None:
        from byol.eval.config import ModelConfig
        data = {"name": "test", "path": "google/gemma-3-4b-pt", "dtype": "float16"}
        mc = ModelConfig.from_dict(data)
        assert mc.dtype == "float16"


# =============================================================================
# 9. TaskConfig validation
# =============================================================================


class TestTaskConfig:
    def test_valid_task_config(self) -> None:
        from byol.eval.config import TaskConfig
        tc = TaskConfig(name="copa", num_fewshot=5)
        assert tc.name == "copa"
        assert tc.num_fewshot == 5

    def test_empty_name_raises(self) -> None:
        from byol.eval.config import TaskConfig
        with pytest.raises(ValueError, match="cannot be empty"):
            TaskConfig(name="")

    def test_negative_fewshot_raises(self) -> None:
        from byol.eval.config import TaskConfig
        with pytest.raises(ValueError, match="non-negative"):
            TaskConfig(name="copa", num_fewshot=-1)

    def test_task_from_dict(self) -> None:
        from byol.eval.config import TaskConfig
        data = {"name": "copa,xcopa_mri", "num_fewshot": 3}
        tc = TaskConfig.from_dict(data)
        assert "copa" in tc.name
        assert tc.num_fewshot == 3


# =============================================================================
# 10. EvalConfig YAML loading
# =============================================================================


class TestEvalConfigLoading:
    def test_load_base_eng_config(self) -> None:
        from byol.eval.config import EvalConfig
        config = EvalConfig.from_yaml(str(CONFIGS_DIR / "benchmark_base_eng.yaml"))
        assert len(config.tasks) > 0
        assert len(config.models) > 0

    def test_load_instruct_mri_config(self) -> None:
        from byol.eval.config import EvalConfig
        config = EvalConfig.from_yaml(str(CONFIGS_DIR / "benchmark_instruct_mri.yaml"))
        assert len(config.tasks) > 0

    def test_nonexistent_config_raises(self) -> None:
        from byol.eval.config import EvalConfig
        with pytest.raises(FileNotFoundError):
            EvalConfig.from_yaml("nonexistent.yaml")

    def test_tasks_path_resolves_to_eval_package(self) -> None:
        """When config has include_path: 'tasks', it should resolve to byol/eval/tasks/."""
        from byol.eval.config import EvalConfig
        config = EvalConfig.from_yaml(str(CONFIGS_DIR / "benchmark_base_mri.yaml"))
        if config.tasks_path:
            tp = Path(config.tasks_path)
            assert tp.exists(), f"tasks_path should exist: {tp}"
            assert "byol/eval/tasks" in str(tp) or "byol\\eval\\tasks" in str(tp)


# =============================================================================
# 11. TaskRegistry
# =============================================================================


class TestTaskRegistry:
    """Verify the TaskRegistry works with the split config files."""

    def test_registry_loads_mri_tasks(self) -> None:
        from byol.eval.config import TaskRegistry
        registry = TaskRegistry(configs_dir=CONFIGS_DIR)
        tc = registry.lookup("xcopa_mri")
        assert tc is not None, "xcopa_mri should be in registry"

    def test_registry_loads_eng_tasks(self) -> None:
        from byol.eval.config import TaskRegistry
        registry = TaskRegistry(configs_dir=CONFIGS_DIR)
        tc = registry.lookup("copa")
        assert tc is not None, "copa should be in registry"

    def test_registry_loads_nya_tasks(self) -> None:
        from byol.eval.config import TaskRegistry
        registry = TaskRegistry(configs_dir=CONFIGS_DIR)
        tc = registry.lookup("xcopa_ny")
        assert tc is not None, "xcopa_ny should be in registry"

    def test_registry_returns_none_for_unknown(self) -> None:
        from byol.eval.config import TaskRegistry
        registry = TaskRegistry(configs_dir=CONFIGS_DIR)
        assert registry.lookup("nonexistent_task_xyz") is None

    def test_registry_resolve_tasks(self) -> None:
        from byol.eval.config import TaskRegistry
        registry = TaskRegistry(configs_dir=CONFIGS_DIR)
        tasks = registry.resolve_tasks("copa,xcopa_mri")
        assert len(tasks) > 0
        all_names = ",".join(tc.name for tc in tasks)
        assert "copa" in all_names
        assert "xcopa_mri" in all_names


# =============================================================================
# 12. Secrets module
# =============================================================================


class TestSecrets:
    def test_mask_token_full(self) -> None:
        from byol.eval.secrets import mask_token
        assert mask_token("hf_abcdefghijklmnop") == "hf_a...mnop"

    def test_mask_token_short(self) -> None:
        from byol.eval.secrets import mask_token
        assert mask_token("short") == "*****"

    def test_mask_token_none(self) -> None:
        from byol.eval.secrets import mask_token
        assert mask_token(None) == "<not set>"


# =============================================================================
# 13. CLI argument parsing
# =============================================================================


class TestCLIParsing:
    def test_parse_base_args(self) -> None:
        from byol.eval.cli import parse_args
        args = parse_args([
            "--model", "google/gemma-3-4b-pt",
            "--type", "base",
            "--tgt-lang", "mri",
        ])
        assert args.model == "google/gemma-3-4b-pt"
        assert args.type == "base"
        assert args.tgt_lang == "mri"

    def test_parse_instruct_args(self) -> None:
        from byol.eval.cli import parse_args
        args = parse_args([
            "--model", "google/gemma-3-4b-it",
            "--type", "instruct",
            "--tgt-lang", "eng",
            "--gpus", "0,1",
            "--dry-run",
        ])
        assert args.type == "instruct"
        assert args.tgt_lang == "eng"
        assert args.gpus == "0,1"
        assert args.dry_run is True

    def test_parse_with_tasks_filter(self) -> None:
        from byol.eval.cli import parse_args
        args = parse_args([
            "--model", "google/gemma-3-4b-pt",
            "--type", "base",
            "--tgt-lang", "mri",
            "--tasks", "xcopa_mri,copa",
        ])
        assert args.tasks == "xcopa_mri,copa"


# =============================================================================
# 14. CLI dry-run integration
# =============================================================================


class TestCLIDryRun:
    """Test the CLI with --dry-run (no GPU needed)."""

    def _run_cli(self, extra_args: list[str]) -> subprocess.CompletedProcess:
        cmd = [
            sys.executable, "-m", CLI_MODULE,
            "--model", "google/gemma-3-4b-pt",
            "--dry-run",
            *extra_args,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        return subprocess.run(
            cmd, capture_output=True, text=True, env=env,
            cwd=str(REPO_DIR),
        )

    def test_type_base_loads_config(self) -> None:
        result = self._run_cli(["--type", "base", "--tgt-lang", "mri"])
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        combined = result.stdout + result.stderr
        assert "BYOL Evaluation" in combined or "Task" in combined

    def test_type_instruct_loads_config(self) -> None:
        result = self._run_cli([
            "--type", "instruct", "--tgt-lang", "eng",
            "--model", "google/gemma-3-4b-it",
        ])
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

    def test_type_with_specific_task_filter(self) -> None:
        result = self._run_cli(["--type", "base", "--tgt-lang", "mri", "--tasks", "xcopa_mri"])
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        combined = result.stdout + result.stderr
        assert "xcopa_mri" in combined, f"Expected xcopa_mri in output: {combined}"

    def test_missing_type_errors(self) -> None:
        result = self._run_cli(["--tgt-lang", "mri"])
        assert result.returncode != 0, "Should fail without --type"

    def test_missing_tgt_lang_errors(self) -> None:
        result = self._run_cli(["--type", "base"])
        assert result.returncode != 0, "Should fail without --tgt-lang"

    def test_module_entry_point(self) -> None:
        """Test python -m byol.eval works."""
        cmd = [
            sys.executable, "-m", "byol.eval",
            "--model", "google/gemma-3-4b-pt",
            "--type", "base",
            "--tgt-lang", "eng",
            "--dry-run",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env,
            cwd=str(REPO_DIR),
        )
        assert result.returncode == 0, f"Module entry point failed: {result.stderr}"


# =============================================================================
# 15. EvalResult dataclass
# =============================================================================


class TestEvalResult:
    def test_eval_result_defaults(self) -> None:
        from byol.eval.runner import EvalResult
        result = EvalResult(model="test", task="copa", status="success")
        assert result.model == "test"
        assert result.error is None
        assert result.metrics == {}
        assert result.duration_seconds == 0.0

    def test_eval_result_with_metrics(self) -> None:
        from byol.eval.runner import EvalResult
        result = EvalResult(
            model="gemma",
            task="copa",
            status="success",
            metrics={"acc,none": 0.75, "acc_norm,none": 0.80},
        )
        assert result.metrics["acc,none"] == 0.75
