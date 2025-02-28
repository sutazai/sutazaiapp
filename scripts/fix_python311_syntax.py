#!/usr/bin/env python3.11
"""
Python 3.11 Syntax Fixer

This script fixes common syntax issues in Python files to make them compatible with Python 3.11.
"""

import os
import sys
import logging
from typing import List, Tuple
import ast
import tokenize
from io import StringIO


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
                python_files.append(os.path.join(root, file))
    return python_files


def fix_indentation(content: str) -> str:
    """Fix indentation issues in Python code."""
    lines = content.split("\n")
    fixed_lines = []
    current_indent = 0

    for line in lines:
        stripped = line.lstrip()
        if not stripped:  # Empty line
            fixed_lines.append("")
            continue

        # Check for indentation markers
        if stripped.startswith(
            ("def ", "class ", "if ", "elif ", "else:", "try:", "except ", "finally:", "with ", "for ", "while ")
        ):
            indent = " " * (current_indent * 4)
            fixed_lines.append(indent + stripped)
            if not stripped.endswith(":"):
                current_indent += 1
        else:
            indent = " " * (current_indent * 4)
            fixed_lines.append(indent + stripped)

        # Adjust indentation based on content
        if stripped.endswith(":"):
            current_indent += 1
        elif stripped.startswith(("return ", "break", "continue", "pass", "raise ")):
            current_indent = max(0, current_indent - 1)

    return "\n".join(fixed_lines)


def validate_syntax(content: str) -> Tuple[bool, str]:
    """Validate Python syntax."""
    try:
        ast.parse(content)
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def fix_file(file_path: str, logger: logging.Logger) -> bool:
    """Fix syntax issues in a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Fix indentation
        fixed_content = fix_indentation(content)

        # Validate syntax
        is_valid, error = validate_syntax(fixed_content)
        if not is_valid:
            logger.error(f"Failed to fix {file_path}: {error}")
            return False

        # Write fixed content back
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(fixed_content)

        logger.info(f"Successfully fixed {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to fix Python syntax issues."""
    logger = setup_logging()

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
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
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
