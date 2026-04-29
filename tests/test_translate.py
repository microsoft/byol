# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Tests for the unified translate() function.
"""

import pytest
from unittest.mock import MagicMock, patch

from byol.common.exceptions import TranslatorNotFoundError, TranslationError
from byol.translation_backends.unified import (
    translate,
    translate_batch,
    clear_cache,
    get_cache_stats,
    _get_or_create_translator,
)


class TestTranslate:
    """Tests for the translate() function."""

    def test_translate_unknown_model_raises_error(self):
        """Test that unknown model raises TranslatorNotFoundError."""
        with pytest.raises(TranslatorNotFoundError) as exc_info:
            translate("Hello", tgt_lang="es", model="nonexistent-xyz")
        
        assert "nonexistent-xyz" in str(exc_info.value)

    def test_translate_auto_src_without_support_raises_error(self):
        """Test that auto source language fails for unsupported models."""
        # NLLB doesn't support auto-detection
        with pytest.raises(ValueError) as exc_info:
            translate(
                "Hello",
                tgt_lang="swh_Latn",
                src_lang="auto",
                model="nllb-200-3.3B"
            )
        
        assert "automatic source language detection" in str(exc_info.value)

    @patch("byol.translation_backends.unified._get_backend_class")
    def test_translate_calls_backend(self, mock_get_backend):
        """Test that translate() properly calls the backend."""
        # Create mock translator
        mock_translator = MagicMock()
        mock_translator.translate.return_value = "Hola mundo"
        
        # Create mock class that returns our translator
        mock_cls = MagicMock(return_value=mock_translator)
        mock_get_backend.return_value = mock_cls
        
        result = translate(
            "Hello world",
            tgt_lang="Spanish",
            src_lang="English",
            model="microsoft-translator"
        )
        
        assert result == "Hola mundo"
        mock_translator.translate.assert_called_once_with("Hello world")

    @patch("byol.translation_backends.unified._get_backend_class")
    def test_translate_caches_translator(self, mock_get_backend):
        """Test that translator instances are cached."""
        mock_translator = MagicMock()
        mock_translator.translate.return_value = "cached"
        mock_cls = MagicMock(return_value=mock_translator)
        mock_get_backend.return_value = mock_cls
        
        clear_cache()
        
        # Call translate twice with same parameters
        translate("text1", tgt_lang="es", model="microsoft-translator")
        translate("text2", tgt_lang="es", model="microsoft-translator")
        
        # Backend class should only be instantiated once
        assert mock_cls.call_count == 1

    @patch("byol.translation_backends.unified._get_backend_class")
    def test_translate_with_raise_on_error_false(self, mock_get_backend):
        """Test that translate() respects raise_on_error=False."""
        mock_translator = MagicMock()
        mock_translator.translate.side_effect = TranslationError("API failed")
        mock_cls = MagicMock(return_value=mock_translator)
        mock_get_backend.return_value = mock_cls
        
        clear_cache()
        
        # Should return empty string instead of raising
        result = translate(
            "Hello",
            tgt_lang="es",
            model="microsoft-translator",
            raise_on_error=False
        )
        
        assert result == ""

    @patch("byol.translation_backends.unified._get_backend_class")
    def test_translate_with_raise_on_error_true(self, mock_get_backend):
        """Test that translate() raises by default."""
        mock_translator = MagicMock()
        mock_translator.translate.side_effect = TranslationError("API failed")
        mock_cls = MagicMock(return_value=mock_translator)
        mock_get_backend.return_value = mock_cls
        
        clear_cache()
        
        # Should raise TranslationError
        with pytest.raises(TranslationError):
            translate("Hello", tgt_lang="es", model="microsoft-translator")


class TestTranslateBatch:
    """Tests for translate_batch() function."""

    @patch("byol.translation_backends.unified._get_backend_class")
    def test_translate_batch_calls_backend(self, mock_get_backend):
        """Test that translate_batch() properly calls the backend."""
        mock_translator = MagicMock()
        mock_translator.translate_batch.return_value = ["Uno", "Dos", "Tres"]
        mock_cls = MagicMock(return_value=mock_translator)
        mock_get_backend.return_value = mock_cls
        
        clear_cache()
        
        result = translate_batch(
            ["One", "Two", "Three"],
            tgt_lang="es",
            model="microsoft-translator"
        )
        
        assert result == ["Uno", "Dos", "Tres"]


class TestCaching:
    """Tests for cache behavior."""

    def test_clear_cache(self):
        """Test that clear_cache clears both caches."""
        stats_before = get_cache_stats()
        clear_cache()
        stats_after = get_cache_stats()
        
        assert stats_after["translator_cache"]["size"] == 0
        assert stats_after["backend_class_cache"]["size"] == 0

    @patch("byol.translation_backends.unified._get_backend_class")
    def test_different_languages_different_cache_keys_for_local(self, mock_get_backend):
        """Test that local models use language-specific cache keys."""
        mock_translator1 = MagicMock()
        mock_translator2 = MagicMock()
        call_count = 0
        
        def create_translator(**kwargs):
            nonlocal call_count
            call_count += 1
            return mock_translator1 if call_count == 1 else mock_translator2
        
        mock_cls = MagicMock(side_effect=create_translator)
        mock_get_backend.return_value = mock_cls
        
        clear_cache()
        
        # Translate with different language pairs for a local model
        translate("Hello", src_lang="eng_Latn", tgt_lang="swh_Latn", model="nllb-200-3.3B")
        translate("Hello", src_lang="eng_Latn", tgt_lang="fra_Latn", model="nllb-200-3.3B")
        
        # Should create two different translator instances
        assert mock_cls.call_count == 2
