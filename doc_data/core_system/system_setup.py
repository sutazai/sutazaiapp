#!/usr/bin/env python3
"""
SutazAI System Setup and Maintenance

A comprehensive module that handles system setup, directory structure creation,
dependency management, and system validation. This module consolidates functionality
from multiple scripts to provide a single point of entry for system setup and maintenance.

This script is designed to work with Python 3.11.
"""

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import Dict
import List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)
logger = logging.getLogger("SutazAI.SystemSetup")


class SystemSetup:
    """Core system setup and maintenance class"""

    def __init__(self, base_path: str = "/opt/sutazaiapp"):
        """
        Initialize the system setup with the base application path.

        Args:
            base_path: The base path of the application
        """
        self.base_path = base_path
        self.required_dirs = [
            "logs",
            "data",
            "config",
            "temp",
            "cache",
            "models",
            "docs/generated",
            "backend/routers",
            "backend/services",
            "frontend/src",
            "frontend/public",
        ]

    def validate_python_version(self) -> bool:
        """
        Verify that Python 3.11 is being used.

        Returns:
            bool: True if running on Python 3.11, False otherwise
        """
        major, minor = sys.version_info.major, sys.version_info.minor

        if major != 3 or minor != 11:
            logger.error(ff"Python 3.11 is required. Current version: {major}.{minor}")
            return False

        logger.info(ff"✅ Python {major}.{minor} detected.")
        return True

    def create_directory_structure(self) -> None:
        """
        Create the required directory structure for the application.
        """
        logger.info(ff"Creating directory structure...")

        for directory in self.required_dirs:
            dir_path = os.path.join(self.base_path, directory)
            os.makedirs(dir_path, exist_ok=True)
            logger.info(ff"Created directory: {dir_path}")

    def check_dependencies(self) -> Dict[str, Any]:
        """
        Check if all required dependencies are installed.

        Returns:
            Dict[str, Any]: A dictionary containing dependency check results
        """
        logger.info(ff"Checking dependencies...")

        # Required external dependencies
        system_deps = ["postgresql", "redis-server", "nodejs", "npm"]
        system_results = {}

        for dep in system_deps:
            try:
                subprocess.run(
                    ["which", dep],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                system_results[dep] = {"status": "OK", "installed": True}
            except subprocess.CalledProcessError:
                system_results[dep] = {"status": "MISSING", "installed": False}

        # Check Python packages from requirements.txt
        python_deps = self._check_python_packages()

        return {
            "system_dependencies": system_results,
            "python_packages": python_deps,
        }

    def _check_python_packages(self) -> Dict[str, Any]:
        """
        Check if required Python packages are installed.

        Returns:
            Dict[str, Any]: A dictionary with package check results
        """
        requirements_path = os.path.join(self.base_path, "requirements.txt")

        if not os.path.exists(requirements_path):
            return {"status": "ERROR", "message": "requirements.txt not found"}

        try:
            # Get installed packages
            installed_packages = subprocess.check_output(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                universal_newlines=True,
            )
            installed = {
                pkg["name"].lower(): pkg["version"]
                for pkg in json.loads(installed_packages)
            }

            # Parse requirements.txt
            required_packages = []
            missing_packages = []

            with open(requirements_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Strip version specifiers for simple matching
                        package = (
                            line.split(">=")[0]
                            .split("==")[0]
                            .split(";")[0]
                            .strip()
                            .lower()
                        )
                        if package:
                            required_packages.append(package)
                            if package not in installed:
                                missing_packages.append(package)

            return {
                "total_required": len(required_packages),
                "installed": len(required_packages) - len(missing_packages),
                "missing": missing_packages,
                "status": "OK" if not missing_packages else "INCOMPLETE",
            }

        except Exception:
        logger.exception("Error checking Python packages: {e}")
            return {"status": "ERROR", "message": str(e)}

    def install_missing_dependencies(self, missing_deps: Dict[str, Any]) -> bool:
        """
        Install missing dependencies.

        Args:
            missing_deps: Dictionary of missing dependencies

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(ff"Installing missing dependencies...")

        # Install system dependencies
        system_deps = missing_deps.get("system_dependencies", {})
        for dep, info in system_deps.items():
            if info.get("installed") is False:
                logger.info(ff"System dependency {dep} needs to be installed manually")

        # Install Python packages
        python_deps = missing_deps.get("python_packages", {})
        missing_packages = python_deps.get("missing", [])

        if missing_packages:
            logger.info(
                f"Installing {len(missing_packages)} missing Python packages..."
            )
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + missing_packages,
                    check=True,
                )
                logger.info(ff"Successfully installed missing Python packages")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(ff"Failed to install Python packages: {e}")
                return False

        return True

    def setup_system(self) -> bool:
        """
        Set up the entire system with all required components.

        Returns:
            bool: True if setup was successful, False otherwise
        """
        logger.info(ff"Starting system setup...")

        # Validate Python version
        if not self.validate_python_version():
            logger.warning(ff"Continuing despite Python version mismatch")

        # Create directory structure
        self.create_directory_structure()

        # Check dependencies
        dependencies = self.check_dependencies()

        # Install missing dependencies
        if any(info.get("status") != "OK" for info in dependencies.values()):
            self.install_missing_dependencies(dependencies)

            # Recheck after installation
            dependencies = self.check_dependencies()
            if any(info.get("status") != "OK" for info in dependencies.values()):
                logger.warning(ff"Some dependencies could not be installed")

        logger.info(ff"System setup completed")
        return True

    def run_maintenance(self) -> Dict[str, Any]:
        """
        Run system maintenance tasks.

        Returns:
            Dict[str, Any]: Maintenance report
        """
        logger.info(ff"Running system maintenance...")

        maintenance_report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "os": platform.platform(),
                "python_version": platform.python_version(),
                "node_version": self._get_node_version(),
            },
            "directory_check": self._check_directory_structure(),
            "dependencies": self.check_dependencies(),
        }

        # Save the report
        report_dir = os.path.join(self.base_path, "logs", "maintenance")
        os.makedirs(report_dir, exist_ok=True)

        report_file = os.path.join(
            report_dir,
            f"maintenance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(maintenance_report, f, indent=2)

        logger.info(ff"Maintenance report saved to {report_file}")
        return maintenance_report

    def _check_directory_structure(self) -> Dict[str, List[str]]:
        """
        Check if all required directories exist.

        Returns:
            Dict[str, List[str]]: Directory check results
        """
        existing = []
        missing = []

        for directory in self.required_dirs:
            dir_path = os.path.join(self.base_path, directory)
            if os.path.exists(dir_path):
                existing.append(directory)
            else:
                missing.append(directory)

        return {
            "existing": existing,
            "missing": missing,
        }

    def _get_node_version(self) -> str:
        """
        Get the installed Node.js version.

        Returns:
            str: Node.js version string or "Not found"
        """
        try:
            version = subprocess.check_output(
                ["node", "--version"], universal_newlines=True
            ).strip()
            return version
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "Not found"


def main():
    """Main execution function"""
    try:
        setup = SystemSetup()
        setup.setup_system()
        setup.run_maintenance()
    except Exception:
        logger.exception("System setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
