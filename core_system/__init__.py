#!/usr/bin/env python3
"""
SutazAI Core System Initialization

Provides safe and robust module initialization
with comprehensive import management.
"""

import logging
import os
import sys
from typing import Any, List, Optional, Set

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "core_system.log")),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# Import the UltraImportResolver
try:
    from core_system.utils.import_resolver import UltraImportResolver

    resolver = UltraImportResolver()
except ImportError as e:
    logger.error(f"Failed to import UltraImportResolver: {e}")
    resolver = None


def safe_import(module_name: str) -> Optional[Any]:
    """
    Safely import a module with comprehensive error handling and logging

    Args:
        module_name (str): Name of the module to import

    Returns:
        Optional[Any]: Imported module or None if import fails
    """
    try:
        if resolver:
            # Check if module needs to be installed
            if not resolver._check_import(module_name):
                package = resolver.suggest_package(module_name)
                if package:
                    logger.info(f"Attempting to install package for {module_name}")
                    resolver.install_missing_packages([package])

        return __import__(module_name)
    except ImportError as e:
        logger.warning(f"Optional module {module_name} not found: {e}")
        if resolver:
            resolver._log_unresolved_import(module_name, str(e))
        return None


def initialize_core_system():
    """Initialize core system components with import validation"""
    logger.info("SutazAI Core System Initializing...")

    if resolver:
        # Scan for import issues
        issues = resolver.scan_project_imports()
        if issues:
            logger.warning("Found import issues in the following files:")
            for file_path, missing in issues.items():
                logger.warning(f"{file_path}: Missing imports: {', '.join(missing)}")

            # Attempt to resolve missing packages
            all_packages: Set[str] = set()
            for imports in issues.values():
                for imp in imports:
                    package = resolver.suggest_package(imp)
                    if package is not None:
                        all_packages.add(package)

            if all_packages:
                logger.info(
                    f"Attempting to install missing packages: {', '.join(all_packages)}"
                )
                resolver.install_missing_packages(list(all_packages))


# Run initialization on import
initialize_core_system()

# Safely import optional dependencies
fitz = safe_import("fitz")  # For PDF processing
networkx = safe_import("networkx")  # For graph operations
yaml = safe_import("yaml")  # For YAML processing
numpy = safe_import("numpy")  # For numerical operations
pandas = safe_import("pandas")  # For data processing
