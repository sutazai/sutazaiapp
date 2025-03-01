#!/usr/bin/env python3.11
"""
Python Compatibility Management for SutazAI Project
This script ensures compatibility across different Python versions
and manages version-specific features and dependencies.
"""
import importlib
import logging
import sys
from typing import Any, Dict, List, Optional

import pkg_resources

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PythonCompatibilityManager:
    """
    Manages Python version compatibility and dependency tracking.
    """

    def __init__(self, python_version: str = "3.11"):
        """
        Initialize the compatibility manager.

        Args:
            python_version: Target Python version for compatibility
        """
        self.target_version = python_version
        self.current_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    def check_version_compatibility(self) -> bool:
        """
        Check if the current Python version meets the target version.

        Returns:
            Boolean indicating version compatibility
        """
        current_major, current_minor = map(
            int, self.current_version.split("."),
        )
        target_major, target_minor = map(
            int, self.target_version.split("."),
        )

        return (current_major > target_major) or (
            current_major == target_major and current_minor >= target_minor
        )

    def get_installed_packages(self) -> List[Dict[str, str]]:
        """
        Retrieve a list of installed Python packages.

        Returns:
            List of dictionaries containing package information
        """
        try:
            return [
                {
                    "name": pkg.key,
                    "version": pkg.version,
                    "location": pkg.location,
                }
                for pkg in pkg_resources.working_set
            ]
        except Exception as e:
            logger.error(f"Error retrieving installed packages: {e}")
            return []

    def check_package_compatibility(
        self, package_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Check compatibility of a specific package.

        Args:
            package_name: Name of the package to check

        Returns:
            Dictionary with package compatibility information
        """
        try:
            package = importlib.import_module(package_name)
            return {
                "name": package_name,
                "version": getattr(package, "__version__", "Unknown"),
                "compatible": True,
            }
        except ImportError:
            logger.warning(f"Package {package_name} not found")
            return None
        except Exception as e:
            logger.error(f"Error checking package {package_name}: {e}")
            return None

    def generate_compatibility_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive compatibility report.

        Returns:
            Dictionary containing compatibility information
        """
        report = {
            "python_version": {
                "current": self.current_version,
                "target": self.target_version,
                "compatible": self.check_version_compatibility(),
            },
            "packages": self.get_installed_packages(),
        }
        return report


def main() -> None:
    """Main function to run the compatibility manager."""
    compatibility_manager = PythonCompatibilityManager()
    report = compatibility_manager.generate_compatibility_report()

    print("Python Compatibility Report:")
    print(f"Current Version: {report['python_version']['current']}")
    print(f"Target Version: {report['python_version']['target']}")
    print(f"Version Compatible: {report['python_version']['compatible']}")
    print("\nInstalled Packages:")
    for pkg in report["packages"]:
        print(f"- {pkg['name']} (v{pkg['version']})")


if __name__ == "__main__":
    main()
