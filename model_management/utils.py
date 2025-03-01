#!/usr/bin/env python3.11
"""
Model Management Utilities
Provides file and configuration management utilities for the model system.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from .monitoring.advanced_logger import log_error, log_info
logger = logging.getLogger(__name__)
def read_file_content(    file_path: str) -> Optional[str]: """Read content from a file.    Args:    file_path: Path to the file to read    Returns:    Optional[str]: File content if successful, None if failed    """    try:        with open(file_path, encoding="utf-8") as f:            return f.read()    except Exception as e:    logger.error(f"Failed to read file {file_path}: {e!s}")    return None
    def write_file_content(        file_path: str,        content: str) -> bool: """Write content to a file.        Args:    file_path: Path to the file to write        content: Content to write to the file        Returns:    bool: True if successful, False if failed        """        try:        with open(file_path, "w", encoding="utf-8") as f:            f.write(content)
        return True
    except Exception as e:        logger.error(f"Failed to write to file {file_path}: {e!s}")
        return False
    def load_json_config(        config_path: str) -> Optional[Dict[str, Any]]: """Load JSON configuration from a file.        Args:    config_path: Path to the JSON config file        Returns:    Optional[Dict[str, Any]]: Config dictionary if successful, None if failed        """        try:        content = read_file_content(config_path)        if content is None:        return None        return json.loads(content)
        except Exception as e:        logger.error(
            f"Failed to load JSON config from {config_path}: {e!s}")
            return None
        def save_json_config(config_path: str,                        config: Dict[str,                Any]) -> bool: """Save configuration to a JSON file.        Args:    config_path: Path to save the JSON config        config: Configuration dictionary to save        Returns:    bool: True if successful, False if failed        """        try:        content = json.dumps(config, indent=2)
            return write_file_content(config_path, content)
        except Exception as e:        logger.error(
            f"Failed to save JSON config to {config_path}: {e!s}")
            return False
        def safe_file_op(        file_path: str,        mode: str,        operation: callable) -> Any: """Execute a file operation safely with proper error handling.        Args:    file_path: Path to the file        mode: File open mode        operation: Callable that performs the file operation        Returns:    Any: Result of the operation
            """
            try:        with open(file_path, mode, encoding="utf-8") as f:            return operation(f)
                except Exception as e:    logger.error(
                    f"File operation failed on {file_path}: {e!s}")
                    raise
                    def operation_with_file(operation: callable) -> callable:                        """Decorator for operations that require file handling.        Args:    operation: The operation to wrap        Returns:    callable: Wrapped operation with file handling        """    def wrapper(file_path: str) -> Any: try: with open(file_path, "rb") as file:    return operation(file)        except Exception as e:        logger.error(            f"Error performing file operation: {e}")        return None        return wrapper
