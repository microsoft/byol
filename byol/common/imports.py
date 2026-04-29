# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Dynamic import utilities for BYOL.

Provides lazy/dynamic importing of classes by string path, enabling
the registry pattern without hardcoded if-elif chains.

Usage:
    from byol.common.imports import lazy_import_class
    
    # Import a class by its full path
    cls = lazy_import_class("byol.translation_backends.api.azure_openai:AzureOpenAIGPT4Translator")
    translator = cls(tgt_lang="Spanish")
"""

from __future__ import annotations

import importlib
from typing import Any, Type

from byol.common.exceptions import BackendImportError
from byol.common.logging import get_logger

logger = get_logger(__name__)


def lazy_import_class(class_path: str) -> Type[Any]:
    """
    Dynamically import a class by its full path.
    
    The path format is: "module.path:ClassName"
    
    Args:
        class_path: Full import path in format "package.module:ClassName".
    
    Returns:
        The imported class.
    
    Raises:
        BackendImportError: If the module or class cannot be imported.
        ValueError: If the class_path format is invalid.
    
    Examples:
        >>> cls = lazy_import_class("byol.translation_backends.api.azure_openai:AzureOpenAIGPT4Translator")
        >>> translator = cls(tgt_lang="Spanish")
        
        >>> cls = lazy_import_class("byol.translation_backends.local.nllb:NLLBTranslator")
    """
    if ":" not in class_path:
        raise ValueError(
            f"Invalid class path format: '{class_path}'. "
            f"Expected format: 'module.path:ClassName'"
        )
    
    module_path, class_name = class_path.rsplit(":", 1)
    
    logger.debug(f"Lazy importing: {class_path}")
    
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise BackendImportError(
            class_path,
            original_error=e,
            hint=f"Make sure the module '{module_path}' exists and its dependencies are installed.",
        )
    
    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise BackendImportError(
            class_path,
            original_error=e,
            hint=f"Module '{module_path}' was imported but class '{class_name}' was not found.",
        )
    
    logger.debug(f"Successfully imported: {class_path}")
    return cls


def module_exists(module_path: str) -> bool:
    """
    Check if a module can be imported without actually importing it.
    
    Args:
        module_path: Full module path (e.g., "byol.translation_backends.api.google").
    
    Returns:
        True if the module exists and can be imported.
    """
    try:
        spec = importlib.util.find_spec(module_path)
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False


def safe_import(module_path: str, default: Any = None) -> Any:
    """
    Safely import a module, returning a default if it fails.
    
    Useful for optional dependencies.
    
    Args:
        module_path: Module to import.
        default: Value to return if import fails.
    
    Returns:
        The imported module or the default value.
    
    Example:
        torch = safe_import("torch", default=None)
        if torch is not None:
            device = torch.device("cuda")
    """
    try:
        return importlib.import_module(module_path)
    except ImportError:
        logger.debug(f"Optional module not available: {module_path}")
        return default
