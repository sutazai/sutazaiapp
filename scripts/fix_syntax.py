#!/usr/bin/env python3.11
"""
Python Syntax Fixer

This script fixes common syntax issues in Python files by using black for formatting
and performing additional syntax validation.
"""

import ast
import logging
import os
import sys
from typing import List, Optional, Tuple

from misc.utils.subprocess_utils import run_command, validate_path


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler("logs/syntax_fix.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                if validate_path(full_path):
                    python_files.append(full_path)
    return python_files


def validate_syntax(content: str) -> Tuple[bool, Optional[str]]:
    """Validate Python syntax."""
    try:
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def fix_file(file_path: str, logger: logging.Logger) -> bool:
    """Fix syntax issues in a Python file using black."""
    try:
        # First validate the current syntax
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        is_valid, error = validate_syntax(content)
        if not is_valid:
            logger.error(f"Invalid syntax in {file_path}: {error}")
            return False

        # Run black on the file using secure subprocess utility
        try:
            result = run_command(
                ["black", file_path],
                check=True,
            )
            logger.info(f"Successfully fixed {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to fix {file_path}: {e}")
            return False

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to fix Python syntax issues."""
    logger = setup_logging()

    # Ensure black is installed
    try:
        import black
    except ImportError:
        logger.error("black package is not installed. Please install it with: pip install black")
        sys.exit(1)

    # Get target directories
    directories = [
        "ai_agents",
        "model_management",
        "backend",
        "scripts",
        "misc",
    ]

    fixed_count = 0
    error_count = 0

    for directory in directories:
        if not validate_path(directory, must_exist=True):
            logger.warning(f"Directory not found or invalid: {directory}")
            continue

        python_files = find_python_files(directory)
        logger.info(f"Found {len(python_files)} Python files in {directory}")

        for file_path in python_files:
            if fix_file(file_path, logger):
                fixed_count += 1
            else:
                error_count += 1

    logger.info(f"Completed: Fixed {fixed_count} files, {error_count} errors")


if __name__ == "__main__":
    main()
