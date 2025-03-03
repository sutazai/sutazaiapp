#!/usr/bin/env python3.11
"""
Model Management Utilities

This module provides utility functions for model management tasks.
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional

import toml
import yaml

logger = logging.getLogger(__name__)


def read_file_content(file_path: str, encoding: str = "utf-8") -> str | None:
    """
    Read content from a file with robust error handling.

    Args:
        file_path: Path to the file to read
        encoding: File encoding (default: utf-8)

    Returns:
        File content if successful, None if failed
    """
    try:
        with open(file_path, encoding=encoding) as f:
            return f.read()
    except OSError as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading file {file_path}: {e}")
        return None


def write_file_content(
    file_path: str,
    content: str,
    encoding: str = "utf-8",
    mode: str = "w",
) -> bool:
    """
    Write content to a file with robust error handling.

    Args:
        file_path: Path to the file to write
        content: Content to write
        encoding: File encoding (default: utf-8)
        mode: File write mode (default: 'w')

    Returns:
        True if write was successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, mode, encoding=encoding) as f:
            f.write(content)

        logger.info(f"Successfully wrote to file {file_path}")
        return True
    except OSError as e:
        logger.error(f"Failed to write to file {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error writing to file {file_path}: {e}")
        return False


def parse_config_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Parse a configuration file (JSON, YAML, or TOML).

    Args:
        file_path: Path to the configuration file

    Returns:
        Parsed configuration dictionary if successful, None otherwise
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()

        content = read_file_content(file_path)
        if not content:
            return None

        if file_extension in (".json", ".jsonc"):
            return json.loads(content)
        if file_extension in (".yaml", ".yml"):
            return yaml.safe_load(content)
        if file_extension == ".toml":
            return toml.loads(content)

        logger.error(f"Unsupported file type: {file_extension}")
        return None

    except Exception as e:
        logger.error(f"Failed to parse configuration file {file_path}: {e}")
        return None


def main():
    """
    Demonstration of utility functions.
    """
    # Example usage with secure temporary file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as temp_file:
        sample_content = '{"model": "gpt-4", "temperature": 0.7}'

        # Write sample file
        write_file_content(temp_file.name, sample_content)

        # Read and parse file
        config = parse_config_file(temp_file.name)
        print("Parsed Configuration:", config)


if __name__ == "__main__":
    main()