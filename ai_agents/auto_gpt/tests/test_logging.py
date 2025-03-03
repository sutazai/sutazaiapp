#!/usr/bin/env python3.11
"""Tests for the logging module."""

import pytest
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from ai_agents.auto_gpt.src.logging import (
    setup_logging,
    get_logger,
    LogLevel,
    LogFormat,
    LogHandler,
    LogFilter,
    LogFormatter,
    LogManager,
)


@pytest.fixture
def log_dir(tmp_path) -> Path:
    """Create a temporary directory for log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def log_config() -> Dict[str, Any]:
    """Create test logging configuration."""
    return {
        "level": LogLevel.INFO,
        "format": LogFormat.DEFAULT,
        "handlers": [
            {
                "type": LogHandler.CONSOLE,
                "level": LogLevel.DEBUG,
            },
            {
                "type": LogHandler.FILE,
                "level": LogLevel.INFO,
                "filename": "test.log",
                "max_bytes": 1024,
                "backup_count": 3,
            },
        ],
        "filters": [
            {
                "type": LogFilter.LEVEL,
                "level": LogLevel.WARNING,
            },
        ],
    }


def test_log_level_enum():
    """Test LogLevel enum values."""
    assert LogLevel.DEBUG.value == "DEBUG"
    assert LogLevel.INFO.value == "INFO"
    assert LogLevel.WARNING.value == "WARNING"
    assert LogLevel.ERROR.value == "ERROR"
    assert LogLevel.CRITICAL.value == "CRITICAL"


def test_log_format_enum():
    """Test LogFormat enum values."""
    assert LogFormat.DEFAULT.value == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    assert LogFormat.SIMPLE.value == "%(levelname)s - %(message)s"
    assert LogFormat.DETAILED.value == "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"


def test_log_handler_enum():
    """Test LogHandler enum values."""
    assert LogHandler.CONSOLE.value == "console"
    assert LogHandler.FILE.value == "file"
    assert LogHandler.SYSLOG.value == "syslog"
    assert LogHandler.HTTP.value == "http"


def test_log_filter_enum():
    """Test LogFilter enum values."""
    assert LogFilter.LEVEL.value == "level"
    assert LogFilter.REGEX.value == "regex"
    assert LogFilter.CUSTOM.value == "custom"


def test_log_formatter():
    """Test LogFormatter functionality."""
    formatter = LogFormatter(LogFormat.DEFAULT)
    
    # Test formatting log record
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert "Test message" in formatted
    assert "INFO" in formatted
    assert "test" in formatted


def test_log_manager_initialization(log_config):
    """Test LogManager initialization."""
    manager = LogManager(log_config)
    assert manager.config == log_config
    assert manager.loggers == {}
    assert manager.handlers == {}
    assert manager.filters == {}


def test_logger_creation(log_config):
    """Test logger creation and configuration."""
    manager = LogManager(log_config)
    logger = manager.get_logger("test_logger")
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    
    # Test logger handlers
    assert len(logger.handlers) == 2  # Console and file handlers
    
    # Test logger filters
    assert len(logger.filters) == 1  # Level filter


def test_log_file_rotation(log_dir, log_config):
    """Test log file rotation."""
    # Update config to use temporary directory
    log_config["handlers"][1]["filename"] = str(log_dir / "test.log")
    
    manager = LogManager(log_config)
    logger = manager.get_logger("test_logger")
    
    # Write enough logs to trigger rotation
    for i in range(100):
        logger.info(f"Test log message {i}")
    
    # Check for rotated files
    log_files = list(log_dir.glob("test.log*"))
    assert len(log_files) > 1  # Original + rotated files
    
    # Check file sizes
    for log_file in log_files:
        assert log_file.stat().st_size <= 1024  # max_bytes


def test_log_level_filtering(log_config):
    """Test log level filtering."""
    manager = LogManager(log_config)
    logger = manager.get_logger("test_logger")
    
    # Test different log levels
    logger.debug("Debug message")  # Should not be logged
    logger.info("Info message")    # Should be logged
    logger.warning("Warning message")  # Should be logged
    logger.error("Error message")  # Should be logged
    
    # Verify console handler level
    console_handler = next(
        h for h in logger.handlers
        if isinstance(h, logging.StreamHandler)
    )
    assert console_handler.level == logging.DEBUG
    
    # Verify file handler level
    file_handler = next(
        h for h in logger.handlers
        if isinstance(h, logging.FileHandler)
    )
    assert file_handler.level == logging.INFO


def test_log_formatting(log_config):
    """Test log message formatting."""
    manager = LogManager(log_config)
    logger = manager.get_logger("test_logger")
    
    # Test different log formats
    for format_type in LogFormat:
        formatter = LogFormatter(format_type)
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "Test message" in formatted
        assert "INFO" in formatted


def test_log_exception_handling(log_config):
    """Test logging of exceptions."""
    manager = LogManager(log_config)
    logger = manager.get_logger("test_logger")
    
    try:
        raise ValueError("Test exception")
    except ValueError as e:
        logger.exception("Exception occurred")
    
    # Verify exception was logged
    log_file = Path(log_config["handlers"][1]["filename"])
    log_content = log_file.read_text()
    assert "Test exception" in log_content
    assert "ValueError" in log_content


def test_log_context(log_config):
    """Test logging with context."""
    manager = LogManager(log_config)
    logger = manager.get_logger("test_logger")
    
    # Add context to log message
    context = {"user_id": "123", "action": "test"}
    logger.info("Action performed", extra=context)
    
    # Verify context was logged
    log_file = Path(log_config["handlers"][1]["filename"])
    log_content = log_file.read_text()
    assert "user_id" in log_content
    assert "action" in log_content


def test_log_manager_cleanup(log_dir, log_config):
    """Test LogManager cleanup."""
    # Update config to use temporary directory
    log_config["handlers"][1]["filename"] = str(log_dir / "test.log")
    
    manager = LogManager(log_config)
    logger = manager.get_logger("test_logger")
    
    # Write some logs
    logger.info("Test message")
    
    # Cleanup
    manager.cleanup()
    
    # Verify handlers were closed
    for handler in logger.handlers:
        assert not handler.stream or handler.stream.closed 