# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Tests for translator factory and registry operations.
"""

import pytest
from unittest.mock import MagicMock, patch

from byol.common.exceptions import TranslatorNotFoundError
from byol.translation_backends.base import BaseTranslator
from byol.translation_backends.factory import get_translator, list_translators
from byol.translation_backends.registry import MODEL_REGISTRY


class TestGetTranslator:
    """Tests for get_translator function."""

    def test_unknown_translator_raises_error(self):
        """Test that requesting an unknown translator raises TranslatorNotFoundError."""
        with pytest.raises(TranslatorNotFoundError) as exc_info:
            get_translator("nonexistent-model-xyz", src_lang="en", tgt_lang="es")
        
        assert "nonexistent-model-xyz" in str(exc_info.value)
        assert exc_info.value.model_name == "nonexistent-model-xyz"
        # Should include available translators in error
        assert exc_info.value.available is not None
        assert isinstance(exc_info.value.available, list)

    def test_case_insensitive_lookup(self):
        """Test that translator names are case-insensitive."""
        # Use a real model from the registry for testing
        model_name = "microsoft-translator"
        if model_name not in MODEL_REGISTRY:
            pytest.skip(f"Model {model_name} not in registry")
        
        # Should work with different cases
        with patch("byol.translation_backends.factory.lazy_import_class") as mock_import:
            mock_cls = MagicMock()
            mock_import.return_value = mock_cls
            
            get_translator("MICROSOFT-TRANSLATOR", src_lang="en", tgt_lang="es")
            get_translator("microsoft-translator", src_lang="en", tgt_lang="es")
            get_translator("Microsoft-Translator", src_lang="en", tgt_lang="es")
            
            assert mock_import.call_count == 3

    def test_get_translator_passes_kwargs(self):
        """Test that kwargs are passed to translator constructor."""
        model_name = "gpt-4o"
        if model_name not in MODEL_REGISTRY:
            pytest.skip(f"Model {model_name} not in registry")
        
        with patch("byol.translation_backends.factory.lazy_import_class") as mock_import:
            mock_cls = MagicMock()
            mock_import.return_value = mock_cls
            
            get_translator(
                model_name,
                src_lang="en",
                tgt_lang="es",
                temperature=0.5
            )
            
            # Check that the class was called with the custom param
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["temperature"] == 0.5


class TestListTranslators:
    """Tests for list_translators function."""

    def test_list_translators_returns_list(self):
        """Test that list_translators returns a sorted list."""
        result = list_translators()
        
        assert isinstance(result, list)
        assert result == sorted(result)  # Should be sorted
        assert len(result) > 0  # Should have some models

    def test_list_translators_filter_by_type(self):
        """Test filtering translators by type."""
        api_translators = list_translators(translator_type="api")
        local_translators = list_translators(translator_type="local")
        all_translators = list_translators()
        
        # Should have both types
        assert len(api_translators) > 0
        assert len(local_translators) > 0
        
        # Combined should equal all (no overlap)
        assert set(api_translators) & set(local_translators) == set()
        assert set(api_translators) | set(local_translators) == set(all_translators)


class TestModelRegistry:
    """Tests for the MODEL_REGISTRY."""

    def test_registry_has_models(self):
        """Test that MODEL_REGISTRY contains models."""
        assert len(MODEL_REGISTRY) > 0
        
    def test_registry_entries_have_required_fields(self):
        """Test that all registry entries have required fields."""
        for name, config in MODEL_REGISTRY.items():
            assert config.backend, f"{name} missing backend"
            assert config.model_type in ("api", "local"), f"{name} has invalid model_type"
            assert config.provider, f"{name} missing provider"

    def test_known_models_exist(self):
        """Test that expected models are in the registry."""
        expected_models = [
            "microsoft-translator",
            "gpt-4o",
            "nllb-200-600m",
        ]
        for model in expected_models:
            assert model in MODEL_REGISTRY, f"Expected model {model} not in registry"
