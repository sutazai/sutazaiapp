#!/usr/bin/env python3.11
import importlib.metadata
import sys
from typing import Any, Dict, List


def check_dependencies() -> bool:
    """
    Check and validate project dependencies.
    
    Returns:
        bool: True if all dependencies are satisfied, False otherwise
    """
    try:
        # Check Python version
        if sys.version_info < (3, 11):
            print(f"Unsupported Python version: {sys.version}")
            return False

        # Check key dependencies
        required_packages = ["langdetect", "pyperclip", "setuptools"]

        for package in required_packages:
            try:
                importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                print(f"Missing dependency: {package}")
                return False

        return True

    except Exception as e:
        print(f"Dependency check failed: {e}")
        return False
