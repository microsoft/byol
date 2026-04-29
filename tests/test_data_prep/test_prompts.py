# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for system prompts."""

import pytest

from byol.data_prep.prompts import (
    REFINE_ENG_PROMPT,
    get_refine_tgt_lang_prompt,
    get_translate_prompt,
)


class TestPrompts:
    """Verify prompts render correctly."""

    def test_refine_tgt_lang_prompt_contains_language(self):
        prompt = get_refine_tgt_lang_prompt("Chichewa", "nya")
        assert "Chichewa" in prompt
        assert "nya" in prompt
        assert "refined" in prompt

    def test_refine_tgt_lang_prompt_different_lang(self):
        prompt = get_refine_tgt_lang_prompt("Māori", "mri")
        assert "Māori" in prompt
        assert "mri" in prompt

    def test_refine_eng_prompt_not_empty(self):
        assert len(REFINE_ENG_PROMPT) > 100
        assert "English" in REFINE_ENG_PROMPT
        assert "cleaned" in REFINE_ENG_PROMPT

    def test_translate_prompt_contains_language(self):
        prompt = get_translate_prompt("Chichewa", "nya")
        assert "Chichewa" in prompt
        assert "nya" in prompt
        assert "translation" in prompt

    def test_translate_prompt_has_json_schema(self):
        prompt = get_translate_prompt("Māori", "mri")
        assert '"items"' in prompt
        assert '"id"' in prompt
        assert '"translation"' in prompt

    def test_refine_tgt_lang_prompt_preserves_religious_texts(self):
        prompt = get_refine_tgt_lang_prompt("Chichewa", "nya")
        assert "religious scriptures" in prompt

    def test_refine_eng_prompt_preserves_religious_texts(self):
        assert "religious scriptures" in REFINE_ENG_PROMPT
