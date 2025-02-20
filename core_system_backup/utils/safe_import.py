#!/usr/bin/env python3
"""
Safe Import Utility

Provides robust and safe module importing mechanisms
to prevent circular imports and improve import reliability.
"""

import importlib
import sys
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

def safe_import(module_name: str, package: Optional[str] = None) -> Any:
    """
    Safely import a module with comprehensive error handling
    
    Args:
        module_name (str): Name of the module to import
        package (Optional[str]): Optional package context for relative imports
    
    Returns:
        Imported module or None if import fails
    """
    try:
        # Attempt standard import
        module = importlib.import_module(module_name, package)
        return module
    except ImportError as e:
        logger.warning(f"Import failed for {module_name}: {e}")
        
        # Attempt alternative import strategies
        try:
            # Try importing from sys.path
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
        except Exception as alt_error:
            logger.error(f"Alternative import failed for {module_name}: {alt_error}")
        
        return None

def get_module_version(module: Any) -> str:
    """
    Retrieve module version safely
    
    Args:
        module (Any): Imported module
    
    Returns:
        Module version as string or 'Unknown'
    """
    try:
        return getattr(module, '__version__', 'Unknown')
    except Exception:
        return 'Unknown'