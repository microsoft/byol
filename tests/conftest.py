# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Pytest configuration and fixtures for BYOL tests.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_translator():
    """Create a mock translator instance."""
    translator = MagicMock()
    translator.translate.return_value = "translated text"
    translator.translate_batch.return_value = ["translated 1", "translated 2"]
    return translator


@pytest.fixture
def mock_backend_class(mock_translator):
    """Create a mock backend class that returns mock translators."""
    mock_cls = MagicMock(return_value=mock_translator)
    return mock_cls


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear all caches before and after each test."""
    from byol.translation_backends.unified import clear_cache
    clear_cache()
    yield
    clear_cache()
