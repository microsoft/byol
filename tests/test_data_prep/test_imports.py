# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Import tests — every data_prep module should import without error."""

import pytest


class TestImports:
    """Verify all submodules import cleanly."""

    def test_import_data_prep(self):
        import byol.data_prep

    def test_import_config(self):
        from byol.data_prep.config import CPTDataPrepConfig, LLMStepConfig

    def test_import_constants(self):
        from byol.data_prep.constants import SUPPORTED_STAGES, LANG_NAMES, REPO_ROOT

    def test_import_prompts(self):
        from byol.data_prep.prompts import (
            get_refine_tgt_lang_prompt,
            REFINE_ENG_PROMPT,
            get_translate_prompt,
        )

    def test_import_runner(self):
        from byol.data_prep.cpt_data_prep_runner import CPTDataPrepRunner, CPTDataPrepResult

    def test_import_cli(self):
        from byol.data_prep.cli import create_parser, main

    def test_import_steps_common(self):
        from byol.data_prep.steps.common import (
            load_jsonl,
            save_jsonl,
            chunked,
            parse_model_json,
            build_user_payload,
        )

    def test_import_steps_download_tgt_lang_fineweb2(self):
        from byol.data_prep.steps.download_tgt_lang_fineweb2 import download_tgt_lang_fineweb2

    def test_import_steps_download_finewebedu(self):
        from byol.data_prep.steps.download_finewebedu import download_and_extract_finewebedu

    def test_import_steps_extract_subset(self):
        """extract_subset.py still importable as a utility."""
        from byol.data_prep.steps.extract_subset import extract_subset

    def test_import_steps_refine(self):
        from byol.data_prep.steps.refine import run_refine

    def test_import_steps_translate(self):
        from byol.data_prep.steps.translate import run_translate
