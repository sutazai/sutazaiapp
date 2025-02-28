#!/usr/bin/env python3
"""
SutazAI System Health Check and Diagnostic Tool

Provides comprehensive system health assessment,
dependency verification, and performance analysis.
"""

import json
import logging
import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict

from misc.utils.subprocess_utils import run_command, run_python_module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    filename="/opt/sutazaiapp/logs/system_health.log",
)
logger = logging.getLogger(__name__)


class SystemHealthChecker:
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        """
        Initialize system health checker.

        Args:
            project_root (str): Root directory of the project
        """
        self.project_root = project_root
        self.health_report_dir = "/opt/sutazaiapp/logs/health_reports"
        os.makedirs(self.health_report_dir, exist_ok=True)

    def check_system_dependencies(self) -> Dict[str, Any]:
        """
        Check critical system dependencies.

        Returns:
            Dict of dependency check results
        """
        dependencies = {
            "python_version": self._check_python_version(),
            "pip_version": self._check_pip_version(),
            "venv_status": self._check_virtual_environment(),
            "required_packages": self._check_required_packages(),
        }
        return dependencies

    def _check_python_version(self) -> Dict[str, Any]:
        """
        Check Python version compatibility.

        Returns:
            Dict with Python version details
        """
        try:
            result = run_python_module("sys", ["--version"], check=False)
            version_output = result.stdout.strip()

            version_parts = version_output.split()[1].split(".")
            major, minor = int(version_parts[0]), int(version_parts[1])

            return {
                "version": version_output,
                "is_compatible": (major == 3 and minor >= 8),
                "status": "OK" if (major == 3 and minor >= 8) else "WARNING",
            }
        except Exception as e:
            logger.exception(f"Python version check failed: {e}")
            return {"error": str(e), "status": "ERROR"}

    def _check_pip_version(self) -> Dict[str, Any]:
        """
        Check pip version and installation.

        Returns:
            Dict with pip version details
        """
        try:
            result = run_python_module("pip", ["--version"], check=False)
            version_output = result.stdout.strip()

            return {"version": version_output, "status": "OK"}
        except Exception as e:
            logger.exception(f"Pip version check failed: {e}")
            return {"error": str(e), "status": "ERROR"}

    def _check_virtual_environment(self) -> Dict[str, Any]:
        """
        Check virtual environment status.

        Returns:
            Dict with virtual environment details
        """
        try:
            result = run_python_module("venv", ["--version"], check=False)
            version_output = result.stdout.strip()

            venv_path = os.environ.get("VIRTUAL_ENV")
            return {
                "venv_available": bool(version_output),
                "active_venv": venv_path if venv_path else None,
                "status": "OK" if venv_path else "WARNING",
            }
        except Exception as e:
            logger.exception(f"Virtual environment check failed: {e}")
            return {"error": str(e), "status": "ERROR"}

    def _check_required_packages(self) -> Dict[str, Any]:
        """
        Check required package installations.

        Returns:
            Dict with package check results
        """
        try:
            result = run_python_module("pip", ["list", "--format=json"], check=False)
            installed_packages = json.loads(result.stdout)

            required_packages = {
                "pylint": ">=2.17.0",
                "black": ">=23.0.0",
                "mypy": ">=1.0.0",
                "ruff": ">=0.1.0",
                "isort": ">=5.12.0",
            }

            package_status = {}
            for package, version_req in required_packages.items():
                package_info = next(
                    (p for p in installed_packages if p["name"].lower() == package.lower()),
                    None,
                )
                if package_info:
                    package_status[package] = {
                        "installed": True,
                        "version": package_info["version"],
                        "status": "OK",
                    }
                else:
                    package_status[package] = {
                        "installed": False,
                        "version": None,
                        "status": "ERROR",
                    }

            return package_status

        except Exception as e:
            logger.exception(f"Package check failed: {e}")
            return {"error": str(e)}

    def generate_health_report(self) -> str:
        """
        Generate comprehensive system health report.

        Returns:
            Path to generated health report
        """
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "os": platform.platform(),
                "python_version": platform.python_version(),
            },
            "dependencies": self.check_system_dependencies(),
        }

        # Generate report filename
        report_filename = f'health_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_path = os.path.join(self.health_report_dir, report_filename)

        # Write report
        with open(report_path, "w") as f:
            json.dump(health_report, f, indent=4)

        logger.info(f"Health report generated: {report_path}")
        return report_path


def main():
    """
    Main execution function for system health check.
    """
    try:
        health_checker = SystemHealthChecker()
        report_path = health_checker.generate_health_report()
        print(f"Health report generated: {report_path}")

        # Check for any critical issues
        with open(report_path) as f:
            report = json.load(f)

        # Example of checking for critical issues
        if any(dep.get("status") == "ERROR" for dep in report["dependencies"].values()):
            sys.exit(1)

    except Exception:
        logger.exception("System health check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
