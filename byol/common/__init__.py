# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
BYOL Common Utilities

Shared utilities used across all BYOL packages (translation_backends, 
language_resource_assessment, training, evaluation, etc.)

Modules:
    - logging: Structured logging with configurable levels
    - caching: Thread-safe caching utilities
    - exceptions: Common exception hierarchy
    - imports: Dynamic import utilities
"""

from byol.common.logging import get_logger, setup_logging, LogLevel
from byol.common.caching import ThreadSafeCache, cached_with_key
from byol.common.exceptions import (
    BYOLError,
    ConfigurationError,
    TranslatorNotFoundError,
    LanguageNotSupportedError,
    BackendImportError,
)
from byol.common.imports import lazy_import_class
from byol.common.translator_support import (
    is_language_supported,
    get_supported_translators,
    validate_translator_for_language,
    resolve_language_codes,
)

__all__ = [
    # Logging
    "get_logger",
    "setup_logging",
    "LogLevel",
    # Caching
    "ThreadSafeCache",
    "cached_with_key",
    # Exceptions
    "BYOLError",
    "ConfigurationError",
    "TranslatorNotFoundError",
    "LanguageNotSupportedError",
    "BackendImportError",
    # Imports
    "lazy_import_class",
    # Translator support
    "is_language_supported",
    "get_supported_translators",
    "validate_translator_for_language",
    "resolve_language_codes",
]
