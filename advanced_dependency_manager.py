#!/usr/bin/env python3
"""
🌐 SutazAI Advanced Dependency and Environment Manager 🌐

Comprehensive dependency management, virtual environment configuration,
and system compatibility toolkit designed to:
- Manage project dependencies
- Create reproducible development environments
- Ensure cross-platform compatibility
- Optimize package installations
- Provide detailed dependency insights
"""

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] 🔍 %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/opt/sutazai/logs/dependency_manager.log"),
    ],
)
logger = logging.getLogger("DependencyManager")


class SutazAIDependencyManager:
    def __init__(self, project_root: str = "/opt/sutazai_project/SutazAI"):
        self.project_root = project_root
        self.dependency_report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "virtual_environments": {},
            "dependencies": {
                "installed": {},
                "outdated": [],
                "compatibility_issues": [],
            },
            "recommendations": [],
        }

    def _get_system_info(self) -> Dict[str, str]:
        """Collect comprehensive system information."""
        return {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
        }

    def create_virtual_environment(self, env_name: str = "sutazai_env") -> str:
        """Create a new virtual environment with advanced configuration."""
        logger.info(f"🌱 Creating Virtual Environment: {env_name}")

        venv_path = os.path.join(self.project_root, env_name)

        try:
            # Create virtual environment
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "venv",
                    "--clear",  # Clear existing environment
                    "--upgrade-deps",  # Upgrade pip, setuptools, wheel
                    venv_path,
                ],
                check=True,
            )

            # Activate virtual environment and upgrade core packages
            activate_script = os.path.join(venv_path, "bin", "activate")
            upgrade_cmd = f"""
            source {activate_script} && 
            pip install --upgrade pip setuptools wheel && 
            pip install --upgrade \
                cython numpy numba psutil py-spy memory_profiler \
                pylint black isort flake8 mypy bandit safety \
                pipdeptree
            """

            subprocess.run(upgrade_cmd, shell=True, executable="/bin/bash", check=True)

            self.dependency_report["virtual_environments"][env_name] = {
                "path": venv_path,
                "created_at": datetime.now().isoformat(),
            }

            logger.info(f"✅ Virtual Environment {env_name} Created Successfully")
            return venv_path

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Virtual Environment Creation Failed: {e}")
            return ""

    def analyze_dependencies(self) -> Dict[str, Any]:
        """Perform comprehensive dependency analysis."""
        logger.info("🔍 Analyzing Project Dependencies...")

        try:
            # Get installed packages
            pip_list = subprocess.run(
                ["pip", "list", "--format=json"], capture_output=True, text=True
            )
            installed_packages = json.loads(pip_list.stdout)

            self.dependency_report["dependencies"]["installed"] = {
                pkg["name"]: pkg["version"] for pkg in installed_packages
            }

            # Check for outdated packages
            outdated_list = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
            )
            outdated_packages = json.loads(outdated_list.stdout)

            self.dependency_report["dependencies"]["outdated"] = [
                {
                    "name": pkg["name"],
                    "current": pkg["version"],
                    "latest": pkg["latest_version"],
                }
                for pkg in outdated_packages
            ]

            # Run safety check for vulnerabilities
            safety_check = subprocess.run(
                ["safety", "check"], capture_output=True, text=True
            )

            if safety_check.returncode != 0:
                self.dependency_report["dependencies"][
                    "compatibility_issues"
                ] = safety_check.stdout.splitlines()

            logger.info("✅ Dependency Analysis Completed")
            return self.dependency_report["dependencies"]

        except Exception as e:
            logger.error(f"❌ Dependency Analysis Failed: {e}")
            return {}

    def generate_requirements_file(self, output_path: str = "") -> str:
        """Generate a comprehensive requirements file with version pinning."""
        logger.info("📦 Generating Requirements File...")

        if not output_path:
            output_path = os.path.join(
                self.project_root, "requirements_comprehensive.txt"
            )

        try:
            # Generate requirements with specific versions
            subprocess.run(["pip", "freeze", ">", output_path], shell=True, check=True)

            # Add some recommended configurations
            with open(output_path, "a") as f:
                f.write("\n# Recommended Configurations\n")
                f.write("--upgrade\n")
                f.write("--no-cache-dir\n")

            logger.info(f"✅ Requirements File Generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"❌ Requirements File Generation Failed: {e}")
            return ""

    def generate_dependency_report(self) -> str:
        """Generate a comprehensive dependency management report."""
        report_path = os.path.join(
            self.project_root,
            f"dependency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        with open(report_path, "w") as f:
            json.dump(self.dependency_report, f, indent=2)

        # Generate human-readable summary
        summary_path = report_path.replace(".json", "_summary.txt")
        with open(summary_path, "w") as f:
            f.write("🚀 SutazAI Dependency Management Report 🚀\n\n")
            f.write(f"Timestamp: {self.dependency_report['timestamp']}\n\n")

            f.write("💻 System Information:\n")
            for key, value in self.dependency_report["system_info"].items():
                f.write(f"- {key.replace('_', ' ').title()}: {value}\n")

            f.write("\n📦 Dependency Overview:\n")
            f.write(
                f"- Total Installed Packages: {len(self.dependency_report['dependencies']['installed'])}\n"
            )
            f.write(
                f"- Outdated Packages: {len(self.dependency_report['dependencies']['outdated'])}\n"
            )

        logger.info(f"📄 Dependency Report Generated: {report_path}")
        logger.info(f"📝 Dependency Summary Generated: {summary_path}")
        return report_path

    def run_dependency_management(self):
        """Execute comprehensive dependency management process."""
        logger.info("🚀 Starting SutazAI Dependency Management 🚀")

        try:
            # Create virtual environment
            self.create_virtual_environment()

            # Analyze dependencies
            self.analyze_dependencies()

            # Generate requirements file
            self.generate_requirements_file()

            # Generate dependency report
            self.generate_dependency_report()

            logger.info("🎉 Dependency Management Completed Successfully 🎉")

        except Exception as e:
            logger.error(f"❌ Dependency Management Process Failed: {e}")


def main():
    dependency_manager = SutazAIDependencyManager()
    dependency_manager.run_dependency_management()


if __name__ == "__main__":
    main()
