# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CLI parsing tests — argument parsing and dry-run behaviour."""

import pytest

from byol.data_prep.cli import create_parser, _build_cpt_config


class TestCLIParsing:
    """Verify CLI argument parsing."""

    def test_stage_required(self):
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_cpt_stage_parse(self):
        parser = create_parser()
        args = parser.parse_args(["--stage", "cpt", "--tgt-lang", "nya"])
        assert args.stage == "cpt"
        assert args.tgt_lang == "nya"

    def test_dry_run_flag(self):
        parser = create_parser()
        args = parser.parse_args(["--stage", "cpt", "--tgt-lang", "mri", "--dry-run"])
        assert args.dry_run is True

    def test_verbose_flag(self):
        parser = create_parser()
        args = parser.parse_args(["--stage", "cpt", "--tgt-lang", "nya", "-v"])
        assert args.verbose is True

    def test_no_refine_flags(self):
        parser = create_parser()
        args = parser.parse_args([
            "--stage", "cpt", "--tgt-lang", "nya",
            "--no-refine-tgt-lang", "--no-refine-eng", "--no-translate",
        ])
        assert args.refine_tgt_lang is False
        assert args.refine_eng is False
        assert args.translate_eng_to_tgt_lang is False
        assert args.download_tgt_lang_fineweb2 is True  # default

    def test_max_samples_override(self):
        parser = create_parser()
        args = parser.parse_args([
            "--stage", "cpt", "--tgt-lang", "nya",
            "--max-samples", "100",
        ])
        assert args.max_samples == 100


class TestBuildConfig:
    """Verify config construction from parsed args."""

    def test_build_from_cli_args(self):
        parser = create_parser()
        args = parser.parse_args(["--stage", "cpt", "--tgt-lang", "nya"])
        cfg = _build_cpt_config(args)
        assert cfg.tgt_lang_code == "nya"
        assert cfg.tgt_lang_name == "Chichewa"

    def test_tgt_lang_required_without_config(self):
        parser = create_parser()
        args = parser.parse_args(["--stage", "cpt"])
        with pytest.raises(SystemExit):
            _build_cpt_config(args)

    def test_config_file_override(self, tmp_path):
        import yaml
        cfg_data = {
            "tgt_lang_code": "mri",
            "tgt_lang_name": "Māori",
            "steps": {"download_tgt_lang_fineweb2": False},
        }
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml.dump(cfg_data))

        parser = create_parser()
        args = parser.parse_args(["--stage", "cpt", "--config", str(cfg_path)])
        cfg = _build_cpt_config(args)
        assert cfg.tgt_lang_code == "mri"

    def test_step_toggle_overrides(self):
        parser = create_parser()
        args = parser.parse_args([
            "--stage", "cpt", "--tgt-lang", "nya",
            "--no-download-tgt-lang-fineweb2",
            "--no-download-eng-finewebedu",
        ])
        cfg = _build_cpt_config(args)
        assert cfg.download_tgt_lang_fineweb2 is False
        assert cfg.download_eng_finewebedu is False
        assert cfg.refine_tgt_lang is True  # still default


class TestDryRun:
    """Verify dry-run doesn't execute any pipeline steps."""

    def test_dry_run_returns_success(self):
        from byol.data_prep.cpt_data_prep_runner import CPTDataPrepRunner
        from byol.data_prep.config import CPTDataPrepConfig

        cfg = CPTDataPrepConfig(tgt_lang_code="nya")
        runner = CPTDataPrepRunner(cfg, dry_run=True)
        result = runner.run()
        assert result.success is True
        assert result.output_files == {}
