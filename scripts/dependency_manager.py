#!/usr/bin/env python3
"""
SutazAI Advanced Dependency Management System

Comprehensive dependency tracking, vulnerability assessment,
and intelligent management capabilities.
"""

import json
import logging
import subprocess
import sys
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List

import pkg_resources
import requests
import safety
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="/opt/sutazaiapp/logs/dependency_management.log",
    filemode="a",
)
logger = logging.getLogger("SutazAI.DependencyManager")


@dataclass
class DependencyReport:
    """
    Comprehensive dependency management report
    """

    timestamp: str
    total_dependencies: int
    outdated_dependencies: List[str]
    vulnerable_dependencies: List[str]
    dependency_details: Dict[str, Any]
    optimization_recommendations: List[str]


class AdvancedDependencyManager:
    """
    Sophisticated dependency management framework
    """

    def __init__(
        self,
        requirements_path: str = "/opt/sutazaiapp/requirements.txt",
        policy_path: str = "/opt/sutazaiapp/config/dependency_policy.yml",
    ):
        """
        Initialize advanced dependency management system

        Args:
            requirements_path (str): Path to requirements file
            policy_path (str): Path to dependency management policy
        """
        self.requirements_path = requirements_path
        self.policy_path = policy_path

        # Load dependency management policy
        with open(policy_path, "r") as f:
            self.policy = yaml.safe_load(f)

    def analyze_dependencies(self) -> Dict[str, Any]:
        """
        Perform comprehensive dependency analysis

        Returns:
            Detailed dependency information
        """
        dependencies = {}

        try:
            with open(self.requirements_path, "r") as f:
                for line in f:
                    if "==" in line and not line.startswith("#"):
                        package, version = line.strip().split("==")

                        try:
                            # Current installed version
                            current_version = pkg_resources.get_distribution(
                                package
                            ).version

                            # Latest version from PyPI
                            latest_version = self._get_latest_version(package)

                            # Vulnerability check
                            vulnerability_check = safety.check(
                                [f"{package}=={version}"]
                            )

                            dependencies[package] = {
                                "current_version": current_version,
                                "required_version": version,
                                "latest_version": latest_version,
                                "is_outdated": current_version
                                != latest_version,
                                "vulnerabilities": vulnerability_check,
                                "potential_replacements": 
                                self._find_alternative_packages(package),
                            }

                        except Exception as e:
                            dependencies[package] = {"error": str(e)}

        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")

        return dependencies

    def _get_latest_version(self, package: str) -> str:
        """
        Retrieve the latest version of a package from PyPI

        Args:
            package (str): Package name

        Returns:
            Latest package version
        """
        try:
            response = requests.get(f"https://pypi.org/pypi/{package}/json")
            return response.json()["info"]["version"]
        except Exception:
            return "Unknown"

    def _find_alternative_packages(self, package: str) -> List[str]:
        """
        Find potential alternative packages

        Args:
            package (str): Original package name

        Returns:
            List of alternative package suggestions
        """
        alternatives = {
            "requests": ["httpx", "aiohttp"],
            "flask": ["fastapi", "starlette"],
            "numpy": ["cupy", "tensorflow"],
        }

        return alternatives.get(package, [])

    def generate_optimization_recommendations(
        self, dependencies: Dict[str, Any]
    ) -> List[str]:
        """
        Generate intelligent dependency optimization recommendations

        Args:
            dependencies (Dict): Comprehensive dependency analysis

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Outdated dependencies
        outdated_deps = [
            pkg
            for pkg, details in dependencies.items()
            if details.get("is_outdated", False)
        ]
        if outdated_deps:
            recommendations.append(
                f"Update outdated dependencies: {', '.join(outdated_deps)}"
            )

        # Vulnerability recommendations
        vulnerable_deps = [
            pkg
            for pkg, details in dependencies.items()
            if details.get("vulnerabilities")
        ]
        if vulnerable_deps:
            recommendations.append(
                f"Address vulnerabilities in: {', '.join(vulnerable_deps)}"
            )

        # Alternative package suggestions
        for pkg, details in dependencies.items():
            if details.get("potential_replacements"):
                recommendations.append(
                    f"Consider alternatives for {pkg}: "
                    f"{', '.join(details['potential_replacements'])}"
                )

        return recommendations

    def generate_dependency_report(self) -> DependencyReport:
        """
        Generate comprehensive dependency report

        Returns:
            Detailed dependency analysis report
        """
        # Analyze dependencies
        dependencies = self.analyze_dependencies()

        # Generate optimization recommendations
        optimization_recommendations = (
            self.generate_optimization_recommendations(dependencies)
        )

        # Create dependency report
        dependency_report = DependencyReport(
            timestamp=datetime.now().isoformat(),
            total_dependencies=len(dependencies),
            outdated_dependencies=[
                pkg
                for pkg, details in dependencies.items()
                if details.get("is_outdated", False)
            ],
            vulnerable_dependencies=[
                pkg
                for pkg, details in dependencies.items()
                if details.get("vulnerabilities")
            ],
            dependency_details=dependencies,
            optimization_recommendations=optimization_recommendations,
        )

        # Persist dependency report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            self.base_dir,
            f'logs/ultra_dependency_report_{timestamp}.json',
        )

        with open(report_path, "w") as f:
            json.dump(asdict(dependency_report), f, indent=2)

        logger.info(f"Dependency report generated: {report_path}")

        return dependency_report

    def update_dependencies(self, force: bool = False):
        """
        Automated dependency update mechanism

        Args:
            force (bool): Force update regardless of policy
        """
        update_candidates = [
            dep
            for dep, details in self.analyze_dependencies().items()
            if details.get("is_outdated") or force
        ]

        if update_candidates:
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        *update_candidates,
                    ],
                    check=True,
                )
                logger.info(f"Updated dependencies: {update_candidates}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Dependency update failed: {e}")


# Verify Python version
def verify_python_version():
    """
    Verify that Python 3.11 or higher is being used.
    """
    major, minor = sys.version_info.major, sys.version_info.minor
    if major < 3 or (major == 3 and minor < 11):
        print("❌ Error: Python 3.11 or higher is required.")
        print(f"Current Python version: {sys.version}")
        print("Please install Python 3.11 and try again.")
        sys.exit(1)
    print(f"✅ Python {major}.{minor} detected.")


def main():
    """
    Main execution for dependency management
    """
    try:
        # Verify Python version
        verify_python_version()
        
        dependency_manager = AdvancedDependencyManager()
        report = dependency_manager.generate_dependency_report()

        print("Dependency Management Recommendations:")
        for recommendation in report.optimization_recommendations:
            print(f"- {recommendation}")

    except Exception as e:
        print(f"Dependency management failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
