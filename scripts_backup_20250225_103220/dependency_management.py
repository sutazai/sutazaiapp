import logging
import subprocess
import sys
from typing import Dict, List

import pkg_resources


class AdvancedDependencyManager:
    def __init__(self):
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
        )
        return logging.getLogger(__name__)

    def get_installed_packages(self) -> Dict[str, str]:
        """Get all installed packages with versions"""
        return {pkg.key: pkg.version for pkg in pkg_resources.working_set}

    def check_package_compatibility(self, package_name: str) -> bool:
        """Check if a specific package is compatible"""
        try:
            __import__(package_name)
            return True
        except ImportError:
            self.logger.warning(f"Package {package_name} not importable")
            return False

    def upgrade_packages(self, packages: List[str] = None) -> bool:
        """Upgrade specified or all packages"""
        packages = packages or list(self.get_installed_packages().keys())

        for package in packages:
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        package,
                    ]
                )
                self.logger.info(f"Upgraded {package}")
            except subprocess.CalledProcessError:
                self.logger.error(f"Failed to upgrade {package}")
                return False

        return True


def main():
    dependency_manager = AdvancedDependencyManager()

    # Check critical packages
    critical_packages = ["pydantic", "safety", "fastapi", "sqlalchemy"]
    compatibility_check = all(
        dependency_manager.check_package_compatibility(pkg) for pkg in critical_packages
    )

    if not compatibility_check:
        print("❌ Package compatibility check failed")
        sys.exit(1)

    # Upgrade packages
    upgrade_success = dependency_manager.upgrade_packages()

    if upgrade_success:
        print("✅ Dependency management completed successfully")
        sys.exit(0)
    else:
        print("❌ Dependency upgrade failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
