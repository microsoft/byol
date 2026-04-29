# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Config tests — YAML loading, validation, derived paths, defaults."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from byol.data_prep.config import (
    CPTDataPrepConfig,
    FineWeb2Config,
    FineWebEduConfig,
    LLMStepConfig,
    SubsetConfig,
)
from byol.data_prep.constants import LANG_NAMES


class TestCPTDataPrepConfig:
    """CPTDataPrepConfig unit tests."""

    def test_minimal_creation(self):
        cfg = CPTDataPrepConfig(tgt_lang_code="nya")
        assert cfg.tgt_lang_code == "nya"
        assert cfg.tgt_lang_name == "Chichewa"

    def test_auto_resolve_lang_name(self):
        cfg = CPTDataPrepConfig(tgt_lang_code="mri")
        assert cfg.tgt_lang_name == "Māori"

    def test_unknown_lang_code_uses_code_as_name(self):
        cfg = CPTDataPrepConfig(tgt_lang_code="xxx")
        assert cfg.tgt_lang_name == "xxx"

    def test_missing_lang_code_raises(self):
        with pytest.raises(ValueError, match="tgt_lang_code is required"):
            CPTDataPrepConfig(tgt_lang_code="")

    def test_default_step_flags_all_true(self):
        cfg = CPTDataPrepConfig(tgt_lang_code="nya")
        assert cfg.download_tgt_lang_fineweb2 is True
        assert cfg.refine_tgt_lang is True
        assert cfg.download_eng_finewebedu is True
        assert cfg.refine_eng is True
        assert cfg.translate_eng_to_tgt_lang is True

    def test_derived_paths(self):
        cfg = CPTDataPrepConfig(tgt_lang_code="nya")
        assert "nya" in cfg.lang_data_dir
        assert cfg.fineweb2_raw_jsonl.endswith("nya_train.jsonl")
        assert "nya_train_refined" in cfg.fineweb2_refined_jsonl
        assert "eng2nya_translated" in cfg.finewebedu_translated_jsonl

    def test_subset_jsonl_name(self):
        cfg = CPTDataPrepConfig(tgt_lang_code="nya")
        assert cfg.finewebedu_subset_jsonl.endswith("eng_train.jsonl")

    def test_from_dict(self):
        data = {
            "tgt_lang_code": "mri",
            "steps": {
                "download_tgt_lang_fineweb2": True,
                "refine_tgt_lang": False,
                "download_eng_finewebedu": True,
                "refine_eng": False,
                "translate_eng_to_tgt_lang": False,
            },
            "subset": {"target_tokens": 100_000_000},
        }
        cfg = CPTDataPrepConfig.from_dict(data)
        assert cfg.tgt_lang_code == "mri"
        assert cfg.refine_tgt_lang is False
        assert cfg.translate_eng_to_tgt_lang is False
        assert cfg.subset.target_tokens == 100_000_000

    def test_from_yaml_roundtrip(self, tmp_path):
        yaml_content = {
            "tgt_lang_code": "nya",
            "tgt_lang_name": "Chichewa",
            "steps": {
                "download_tgt_lang_fineweb2": True,
                "refine_tgt_lang": True,
                "download_eng_finewebedu": True,
                "refine_eng": True,
                "translate_eng_to_tgt_lang": True,
            },
            "fineweb2": {"dataset_name": "nya_Latn", "splits": ["train"]},
            "subset": {"seed": 42},
        }
        path = tmp_path / "test.yaml"
        path.write_text(yaml.dump(yaml_content))
        cfg = CPTDataPrepConfig.from_yaml(path)
        assert cfg.tgt_lang_code == "nya"
        assert cfg.fineweb2.dataset_name == "nya_Latn"

    def test_from_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            CPTDataPrepConfig.from_yaml("/nonexistent/path.yaml")


class TestExampleConfig:
    """Validate the shipped example config file."""

    @pytest.fixture
    def nya_config_path(self):
        from byol.data_prep.constants import REPO_ROOT
        return REPO_ROOT / "configs" / "data_prep" / "cpt" / "nya.yaml"

    def test_example_config_exists(self, nya_config_path):
        assert nya_config_path.exists(), f"Example config not found: {nya_config_path}"

    def test_example_config_loads(self, nya_config_path):
        cfg = CPTDataPrepConfig.from_yaml(nya_config_path)
        assert cfg.tgt_lang_code == "nya"
        assert cfg.tgt_lang_name == "Chichewa"
        assert cfg.download_tgt_lang_fineweb2 is True

    def test_example_config_subset_tokens(self, nya_config_path):
        cfg = CPTDataPrepConfig.from_yaml(nya_config_path)
        assert cfg.subset.target_tokens is None  # auto-computed from tgt_lang


class TestLLMStepConfig:
    """LLMStepConfig unit tests."""

    def test_defaults(self):
        sc = LLMStepConfig()
        assert sc.model_name == "gpt-5"
        assert sc.reasoning_effort == "low"
        assert sc.batch_size > 0
        assert sc.concurrency > 0

    def test_custom_values(self):
        sc = LLMStepConfig(model_name="gpt-5-mini", batch_size=32, concurrency=64)
        assert sc.model_name == "gpt-5-mini"
        assert sc.batch_size == 32


class TestSubsetConfig:
    """SubsetConfig unit tests."""

    def test_defaults(self):
        sc = SubsetConfig()
        assert sc.target_tokens is None  # auto-computed
        assert sc.max_token_count == 1024
        assert sc.seed == 42
