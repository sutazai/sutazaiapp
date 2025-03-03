#!/usr/bin/env python3.11
"""
Advanced Logging Module

This module provides enhanced logging capabilities for the SutazAI system.
"""

import logging
import sys
from typing import Optional, Union
from typing import Union


def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
    log_format: str | None = None,
) -> logging.Logger:
    """
    Setup comprehensive logging configuration.

    Args:
    log_level: Logging level (default: "INFO")
    log_file: Optional file path to write logs
    log_format: Optional custom log format

    Returns:
    Configured logger instance
    """
    # Configure logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure log format
    default_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_formatter = logging.Formatter(log_format or default_format)

    # Create logger
    logger = logging.getLogger("SutazAILogger")
    logger.setLevel(numeric_level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file is provided)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(log_formatter)
            logger.addHandler(file_handler)
        except OSError as e:
            logger.error(f"Could not create log file: {e}")

    return logger


def log_info(message: str | object, logger: Optional[logging.Logger] = None) -> None:
    """
    Log an informational message.

    Args:
    message: Message to log
    logger: Optional logger instance. If not provided, uses root logger.
    """
    (logger or logging.getLogger()).info(str(message))


def log_warning(message: str | object, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a warning message.

    Args:
    message: Message to log
    logger: Optional logger instance. If not provided, uses root logger.
    """
    (logger or logging.getLogger()).warning(str(message))


def log_error(message: str | object, logger: Optional[logging.Logger] = None) -> None:
    """
    Log an error message.

    Args:
    message: Message to log
    logger: Optional logger instance. If not provided, uses root logger.
    """
    (logger or logging.getLogger()).error(str(message))


def log_critical(message: str | object, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a critical error message.

    Args:
    message: Message to log
    logger: Optional logger instance. If not provided, uses root logger.
    """
    (logger or logging.getLogger()).critical(str(message))


def main() -> None:
    """
    Demonstration of advanced logging capabilities.
    """
    # Setup logging with custom configuration
    logger = setup_logging(
        log_level="DEBUG",
        log_file="/tmp/sutazai_advanced_logger.log",
    )

    # Demonstrate logging levels
    log_info("This is an informational message", logger)
    log_warning("This is a warning message", logger)
    log_error("This is an error message", logger)
    log_critical("This is a critical error message", logger)


if __name__ == "__main__":
    main()