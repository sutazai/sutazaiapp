#!/usr/bin/env python3
"""
Safe Import Utility

Provides robust and safe module importing mechanisms
to prevent circular imports and improve import reliability.
"""

import importlib
import logging
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

ALLOWED_MODULES = {
    "ai_agents": ["supreme_ai", "agent_factory"],
    "core_system": [
        "DamerauLevenshtein",
        "appearance",
        "dependency_management",
        "system_architecture_analyzer",
        "system_integrator",
    ],
}


def safe_import(module_name: str, package: Optional[str] = None) -> Any:
    """
    Safely import modules with strict whitelisting

    Args:
        module_name (str): Name of the module to import
        package (Optional[str]): Optional package name

    Returns:
        Imported module or raises ImportError
    """
    parts = module_name.split(".")
    root_module = parts[0]

    if root_module not in ALLOWED_MODULES:
        raise ImportError(f"Module {root_module} is not in the allowed list")

    if len(parts) > 1 and parts[1] not in ALLOWED_MODULES[root_module]:
        raise ImportError(f"Submodule {parts[1]} is not in the allowed list")

    try:
        return importlib.import_module(module_name, package)
    except ImportError as e:
        print(f"Import Error: {e}", file=sys.stderr)
        raise


def get_module_version(module: Any) -> str:
    """
    Retrieve module version safely

    Args:
        module (Any): Imported module

    Returns:
        Module version as string or 'Unknown'
    """
    try:
        return getattr(module, "__version__", "Unknown")
    except Exception:
        return "Unknown"
