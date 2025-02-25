#!/usr/bin/env python3
"""
SutazAI Dependency Management and Tracking Utility

Provides advanced capabilities for:
- Dependency discovery
- Version tracking
- Compatibility checking
- Automated updates
- Dependency graph generation
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from packaging import version

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            "/opt/sutazai_project/SutazAI/logs/dependency_management.log"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("SutazAI.DependencyManager")


class AdvancedDependencyManager:
    """
    Comprehensive Dependency Management Framework

    Provides intelligent dependency tracking,
    compatibility analysis, and automated management
    """

    def __init__(
        self,
        base_dir: str = "/opt/sutazai_project/SutazAI",
        requirements_path: Optional[str] = None,
    ):
        """
        Initialize dependency manager

        Args:
            base_dir (str): Base project directory
            requirements_path (Optional[str]): Path to requirements file
        """
        self.base_dir = base_dir
        self.requirements_path = requirements_path or os.path.join(
            base_dir, "requirements.txt"
        )
        self.dependency_cache_path = os.path.join(base_dir, "dependency_cache.json")

    def discover_dependencies(self) -> Dict[str, Any]:
        """
        Discover and analyze project dependencies

        Returns:
            Dictionary of dependency details
        """
        dependencies = {}

        try:
            with open(self.requirements_path, "r") as f:
                requirements = f.readlines()

            for req in requirements:
                req = req.strip()
                if req and not req.startswith("#"):
                    try:
                        # Parse dependency details
                        name, current_version = req.split("==")

                        # Get latest version
                        latest_version = self._get_latest_version(name)

                        dependencies[name] = {
                            "current_version": current_version,
                            "latest_version": latest_version,
                            "needs_update": version.parse(latest_version)
                            > version.parse(current_version),
                        }
                    except Exception as e:
                        logger.warning(f"Could not process dependency {req}: {e}")

        except FileNotFoundError:
            logger.error(f"Requirements file not found: {self.requirements_path}")

        return dependencies

    def _get_latest_version(self, package_name: str) -> str:
        """
        Retrieve the latest version of a package

        Args:
            package_name (str): Name of the package

        Returns:
            Latest version string
        """
        try:
            result = subprocess.run(
                ["pip", "index", "versions", package_name],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse output to get latest version
            versions = result.stdout.split("\n")
            latest = versions[0].split()[0] if versions else "Unknown"

            return latest

        except subprocess.CalledProcessError:
            logger.warning(f"Could not retrieve latest version for {package_name}")
            return "Unknown"

    def update_dependencies(self, force: bool = False) -> Dict[str, Any]:
        """
        Update project dependencies

        Args:
            force (bool): Force update all dependencies

        Returns:
            Dictionary of update results
        """
        update_results = {}
        dependencies = self.discover_dependencies()

        for name, details in dependencies.items():
            if force or details["needs_update"]:
                try:
                    # Perform dependency update
                    subprocess.run(
                        [
                            "pip",
                            "install",
                            "--upgrade",
                            f'{name}=={details["latest_version"]}',
                        ],
                        check=True,
                        capture_output=True,
                    )

                    update_results[name] = {
                        "old_version": details["current_version"],
                        "new_version": details["latest_version"],
                        "status": "Updated Successfully",
                    }

                    # Update requirements file
                    self._update_requirements_file(name, details["latest_version"])

                except subprocess.CalledProcessError as e:
                    update_results[name] = {
                        "old_version": details["current_version"],
                        "new_version": details["latest_version"],
                        "status": f"Update Failed: {e}",
                    }

        # Persist update results
        self._log_update_results(update_results)

        return update_results

    def _update_requirements_file(self, package_name: str, new_version: str):
        """
        Update requirements file with new package version

        Args:
            package_name (str): Name of the package
            new_version (str): New version to update
        """
        try:
            with open(self.requirements_path, "r") as f:
                requirements = f.readlines()

            updated_requirements = [
                (f"{package_name}=={new_version}" if package_name in line else line)
                for line in requirements
            ]

            with open(self.requirements_path, "w") as f:
                f.writelines(updated_requirements)

        except Exception as e:
            logger.error(f"Could not update requirements file: {e}")

    def _log_update_results(self, update_results: Dict[str, Any]):
        """
        Log dependency update results

        Args:
            update_results (Dict): Results of dependency updates
        """
        log_path = os.path.join(
            self.base_dir,
            "logs",
            f'dependency_update_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
        )

        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(update_results, f, indent=2)

            logger.info(f"Dependency update results logged: {log_path}")

        except Exception as e:
            logger.error(f"Could not log update results: {e}")


def main():
    """
    Main execution for dependency management
    """
    try:
        dependency_manager = AdvancedDependencyManager()

        # Discover dependencies
        dependencies = dependency_manager.discover_dependencies()
        print("Current Dependencies:")
        for name, details in dependencies.items():
            print(f"{name}: {details}")

        # Update dependencies
        update_results = dependency_manager.update_dependencies()
        print("\nDependency Update Results:")
        for name, result in update_results.items():
            print(f"{name}: {result}")

    except Exception as e:
        logger.error(f"Dependency management failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
