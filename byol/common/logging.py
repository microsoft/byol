# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Structured logging with configurable levels for BYOL.

Provides a consistent logging interface across all BYOL packages with:
- Configurable log levels
- Structured output format
- Color support for terminals
- Easy per-module logger creation

Usage:
    from byol.common.logging import get_logger, setup_logging, LogLevel
    
    # Setup logging for the entire application
    setup_logging(level=LogLevel.INFO)
    
    # Get a logger for your module
    logger = get_logger(__name__)
    logger.info("Processing started", extra={"model": "gpt-5", "count": 100})
"""

from __future__ import annotations

import logging
import sys
from enum import IntEnum
from typing import Optional, TextIO


class LogLevel(IntEnum):
    """Log level enumeration with clear naming."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# ANSI color codes for terminal output
class _Colors:
    """ANSI color codes for log level colorization."""
    RESET = "\033[0m"
    DEBUG = "\033[36m"      # Cyan
    INFO = "\033[32m"       # Green
    WARNING = "\033[33m"    # Yellow
    ERROR = "\033[31m"      # Red
    CRITICAL = "\033[35m"   # Magenta
    TIMESTAMP = "\033[90m"  # Gray
    NAME = "\033[34m"       # Blue


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log output for terminals.
    
    Falls back to plain text when output is not a terminal.
    """
    
    LEVEL_COLORS = {
        logging.DEBUG: _Colors.DEBUG,
        logging.INFO: _Colors.INFO,
        logging.WARNING: _Colors.WARNING,
        logging.ERROR: _Colors.ERROR,
        logging.CRITICAL: _Colors.CRITICAL,
    }
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
    ):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stderr.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            # Colorize the level name
            level_color = self.LEVEL_COLORS.get(record.levelno, "")
            record.levelname = f"{level_color}{record.levelname:<8}{_Colors.RESET}"
            
            # Colorize the logger name
            record.name = f"{_Colors.NAME}{record.name}{_Colors.RESET}"
        else:
            record.levelname = f"{record.levelname:<8}"
        
        return super().format(record)


# Default format string
DEFAULT_FORMAT = "%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track if logging has been configured
_logging_configured = False


def setup_logging(
    level: LogLevel | int = LogLevel.INFO,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    stream: Optional[TextIO] = None,
    use_colors: bool = True,
    force: bool = False,
) -> None:
    """
    Configure logging for the BYOL package.
    
    This should be called once at application startup. Subsequent calls
    are ignored unless force=True.
    
    Args:
        level: Minimum log level to display.
        format_string: Custom format string. Uses DEFAULT_FORMAT if None.
        date_format: Custom date format. Uses DEFAULT_DATE_FORMAT if None.
        stream: Output stream. Defaults to sys.stderr.
        use_colors: Whether to use ANSI colors (auto-disabled if not a TTY).
        force: If True, reconfigure even if already configured.
    
    Example:
        # Basic setup
        setup_logging(level=LogLevel.DEBUG)
        
        # Production setup (no colors, custom format)
        setup_logging(
            level=LogLevel.WARNING,
            use_colors=False,
            format_string="%(asctime)s [%(levelname)s] %(message)s"
        )
    """
    global _logging_configured
    
    if _logging_configured and not force:
        return
    
    # Get the root BYOL logger
    root_logger = logging.getLogger("byol")
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create handler
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)
    
    # Create formatter
    formatter = ColoredFormatter(
        fmt=format_string or DEFAULT_FORMAT,
        datefmt=date_format or DEFAULT_DATE_FORMAT,
        use_colors=use_colors,
    )
    handler.setFormatter(formatter)
    
    root_logger.addHandler(handler)
    
    # Don't propagate to root logger
    root_logger.propagate = False
    
    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    The logger will be a child of the 'byol' root logger, ensuring
    consistent configuration across all BYOL modules.
    
    Args:
        name: Logger name, typically __name__ of the calling module.
    
    Returns:
        Configured Logger instance.
    
    Example:
        logger = get_logger(__name__)
        logger.info("Translation complete", extra={"model": "gpt-5"})
    """
    # Ensure logging is configured with defaults
    if not _logging_configured:
        setup_logging()
    
    # If name starts with 'byol.', use it directly; otherwise prefix it
    if name.startswith("byol."):
        return logging.getLogger(name)
    else:
        return logging.getLogger(f"byol.{name}")


def set_level(level: LogLevel | int, logger_name: Optional[str] = None) -> None:
    """
    Change the log level for a specific logger or all BYOL loggers.
    
    Args:
        level: New log level.
        logger_name: Specific logger to change. If None, changes root byol logger.
    
    Example:
        # Enable debug for all BYOL loggers
        set_level(LogLevel.DEBUG)
        
        # Enable debug only for translation backends
        set_level(LogLevel.DEBUG, "byol.translation_backends")
    """
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger("byol")
    
    logger.setLevel(level)
