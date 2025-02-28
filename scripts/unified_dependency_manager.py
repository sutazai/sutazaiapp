from typing import Dict, List, Optional

#!/usr/bin/env python3.11
"""
SutazAI Unified Dependency Manager

Advanced dependency management system providing:
- Comprehensive dependency tracking and analysis
- Vulnerability detection and security assessment
- Dependency graph visualization
- Automated dependency updates and conflict resolution
- Package optimization recommendations
"""

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional, dict, list

import networkx as nx
import pkg_resources
from misc.utils.subprocess_utils import run_command, run_python_module
from packaging import version
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to Python path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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


        @dataclass
        class PackageInfo:
            """Comprehensive package information"""

            name: str
            current_version: str
            latest_version: Optional[str] = None
            is_outdated: bool = False
            is_vulnerable: bool = False
            vulnerability_details: List[Dict[str, Any]] = field(
                default_factory=list)
            dependencies: List[str] = field(default_factory=list)
            used_by: List[str] = field(default_factory=list)
            update_priority: int = 0  # 0-10, 10 being highest priority


            @dataclass
            class DependencyReport:
                """Comprehensive dependency management report"""

                timestamp: str
                total_dependencies: int
                outdated_dependencies: List[PackageInfo]
                vulnerable_dependencies: List[PackageInfo]
                dependency_graph: Dict[str, Any]
                optimization_recommendations: List[str]


                class UnifiedDependencyManager:
                    """Advanced dependency management system with comprehensive capabilities"""

                    def __init__(
                        self,
                        requirements_path: str = "/opt/sutazaiapp/requirements.txt",
                        policy_path: str = "/opt/sutazaiapp/config/dependency_policy.yml",
                        log_dir: str = "/opt/sutazaiapp/logs",
                        ):
                        """Initialize unified dependency manager

                        Args:
                        requirements_path (str): Path to requirements.txt file
                        policy_path (
                            str): Path to dependency policy configuration
                        log_dir (str): Directory for logging
                        """
                        self.requirements_path = requirements_path
                        self.policy_path = policy_path
                        self.log_dir = log_dir

                        # Rich console for visualization
                        self.console = Console()

                        # Ensure log directory exists
                        os.makedirs(log_dir, exist_ok=True)

                        # Configure logging
                        log_file = os.path.join(
                            log_dir,
                            "dependency_manager.log")
                        logging.basicConfig(
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s: %(message)s",
                        handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler(),
                        ],
                        )
                        self.logger = logging.getLogger(
                            "SutazAI.DependencyManager")

                        # Package registry for API access
                        self.pypi_url = "https://pypi.org/pypi"

                        # Initialize dependency graph
                        self.dependency_graph = nx.DiGraph()

                        def analyze_dependencies(self) -> Dict[str, Any]:
                            """Perform comprehensive dependency analysis

                            Returns:
                            Detailed dependency analysis report
                            """
                            try:
                                # Get currently installed packages
                                installed_packages = self.get_installed_packages()

                                # Build dependency graph
                                self._build_dependency_graph(
                                    installed_packages)

                                # Check for outdated packages
                                outdated_packages = self._check_outdated_packages(
                                    installed_packages)

                                # Check for vulnerabilities
                                vulnerable_packages = self.check_vulnerabilities()

                                # Analyze for optimization
                                optimization_recommendations = self.generate_optimization_recommendations(
                                installed_packages,
                                outdated_packages,
                                vulnerable_packages,
                                )

                                # Create analysis report
                                analysis_report = {
                                "timestamp": datetime.now().isoformat(),
                                "total_packages": len(installed_packages),
                                "outdated_packages": [asdict(
                                    pkg) for pkg in outdated_packages],
                                "vulnerable_packages": [asdict(
                                    pkg) for pkg in vulnerable_packages],
                                "outdated_count": len(outdated_packages),
                                "vulnerable_count": len(vulnerable_packages),
                                "recommendations": optimization_recommendations,
                                }

                                self.logger.info(
                                f"Dependency analysis completed: "
                                f"{analysis_report['total_packages']} packages analyzed, "
                                f"{analysis_report['outdated_count']} outdated, "
                                f"{analysis_report['vulnerable_count']} vulnerable.",
                                )

                            return analysis_report

                            except Exception as e:
                                self.logger.error(
                                    "Dependency analysis failed: %s",
                                    e)
                            return {
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                            }

                            def get_installed_packages(
                                self) -> List[PackageInfo]:
                                """Get comprehensive information about installed packages

                                Returns:
                                List of PackageInfo objects
                                """
                                try:
                                    # Get list of installed packages
                                    result = run_python_module(
                                        "pip",
                                        ["list", "--format=json"],
                                        check=False)
                                    installed_packages = json.loads(
                                        result.stdout)

                                    package_info_list = []
                                    for pkg in installed_packages:
                                        name = pkg["name"]
                                        current_version = pkg["version"]

                                        # Get latest version from PyPI
                                        try:
                                            pypi_result = run_command(
                                            ["pip", "index", "versions", name],
                                            check=False,
                                            )
                                            latest_version = pypi_result.stdout.split(
                                                "\n")[0].split()[-1]
                                            except Exception:
                                                latest_version = current_version

                                                # Check for vulnerabilities using safety
                                                try:
                                                    safety_result = run_command(
                                                    ["safety", "check", f"{name}=={current_version}", "--json"],
                                                    check=False,
                                                    )
                                                    vulnerabilities = json.loads(
                                                        safety_result.stdout) if safety_result.returncode != 0 else []
                                                    except Exception:
                                                        vulnerabilities = []

                                                        # Get package dependencies
                                                        try:
                                                            deps_result = run_python_module(
                                                            "pip",
                                                            ["show", name],
                                                            check=False,
                                                            )
                                                            deps_lines = [
                                                            line.split(
                                                                ": ")[1].strip()
                                                            for line in deps_result.stdout.split(
                                                                "\n")
                                                                if line.startswith(
                                                                    "Requires: ")
                                                                    ]
                                                                    dependencies = deps_lines[0].split(
                                                                        ",
                                                                        ") if deps_lines else []
                                                                    except Exception:
                                                                        dependencies = []

                                                                        # Create PackageInfo object
                                                                        pkg_info = PackageInfo(
                                                                        name=name,
                                                                        current_version=current_version,
                                                                        latest_version=latest_version,
                                                                        is_outdated=version.parse(
                                                                            latest_version) > version.parse(current_version),
                                                                        is_vulnerable=bool(
                                                                            vulnerabilities),
                                                                        vulnerability_details=vulnerabilities,
                                                                        dependencies=dependencies,
                                                                        )

                                                                        package_info_list.append(
                                                                            pkg_info)

                                                                    return package_info_list

                                                                    except Exception as e:
                                                                        logging.exception(
                                                                            f"Failed to get installed packages: {e}")
                                                                    return []

                                                                    def update_package(
                                                                        self,
                                                                        package_name: str,
                                                                        target_version: Optional[str] = None) -> bool:
                                                                                                                                                """Update a specific package to latest or \
                                                                            target version

                                                                        Args:
                                                                        package_name (
                                                                            str): Name of package to update
                                                                        target_version (
                                                                            str,
                                                                            optional): Target version, latest if not specified

                                                                        Returns:
                                                                        bool: True if update successful
                                                                        """
                                                                        try:
                                                                            cmd = ["pip", "install", "--upgrade"]
                                                                            if target_version:
                                                                                cmd.append(
                                                                                    f"{package_name}=={target_version}")
                                                                                else:
                                                                                cmd.append(
                                                                                    package_name)

                                                                                result = run_command(
                                                                                    cmd,
                                                                                    check=True)
                                                                            return result.returncode == 0

                                                                            except Exception as e:
                                                                                logging.exception(
                                                                                    f"Failed to update package {package_name}: {e}")
                                                                            return False

                                                                            def check_vulnerabilities(
                                                                                self) -> List[Dict[str, Any]]:
                                                                                                                                                                """Check for security vulnerabilities in \
                                                                                    dependencies

                                                                                Returns:
                                                                                List of vulnerability reports
                                                                                """
                                                                                try:
                                                                                    result = run_command(
                                                                                    ["safety", "check", "--json", "-r", self.requirements_path],
                                                                                    check=False,
                                                                                    )
                                                                                    if result.returncode != 0:
                                                                                    return json.loads(
                                                                                        result.stdout)
                                                                                return []

                                                                                except Exception as e:
                                                                                    logging.exception(
                                                                                        f"Failed to check vulnerabilities: {e}")
                                                                                return []

                                                                                def _build_dependency_graph(
                                                                                    self,
                                                                                    packages: Dict[str, PackageInfo]) -> None:
                                                                                    """Build a comprehensive dependency graph

                                                                                    Args:
                                                                                    packages (
                                                                                        Dict): Dictionary of package information
                                                                                    """
                                                                                    try:
                                                                                        # Clear existing graph
                                                                                        self.dependency_graph.clear()

                                                                                        # Add all packages as nodes
                                                                                        for name, pkg_info in packages.items():
                                                                                            self.dependency_graph.add_node(
                                                                                            name,
                                                                                            version=pkg_info.current_version,
                                                                                            is_outdated=pkg_info.is_outdated,
                                                                                            is_vulnerable=pkg_info.is_vulnerable,
                                                                                            )

                                                                                            # Add dependency relationships as edges
                                                                                            for name, pkg_info in packages.items():
                                                                                                                                                                                                for dep in \
                                                                                                    pkg_info.dependencies:
                                                                                                                                                                                                        if dep in \
                                                                                                        packages:
                                                                                                        self.dependency_graph.add_edge(
                                                                                                            name,
                                                                                                            dep)

                                                                                                        self.logger.info(
                                                                                                        f"Dependency graph built: {len(
                                                                                                            self.dependency_graph.nodes())} nodes, "
                                                                                                        f"{len(
                                                                                                            self.dependency_graph.edges())} edges.",
                                                                                                        )

                                                                                                        except Exception as e:
                                                                                                            self.logger.error(
                                                                                                                "Failed to build dependency graph: %s",
                                                                                                                e)
                                                                                                            # Re-raise to allow proper error handling by caller
                                                                                                        raise

                                                                                                        def _check_outdated_packages(
                                                                                                            self,
                                                                                                            packages: Dict[str, PackageInfo],
                                                                                                            ) -> List[PackageInfo]:
                                                                                                            """Check for outdated packages

                                                                                                            Args:
                                                                                                            packages (
                                                                                                                Dict): Dictionary of package information

                                                                                                            Returns:
                                                                                                            List of outdated packages
                                                                                                            """
                                                                                                            outdated_packages = []

                                                                                                            try:
                                                                                                                # Call pip list --outdated to get outdated packages
                                                                                                                process = subprocess.run(
                                                                                                                [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                                                                                                                capture_output=True,
                                                                                                                text=True,
                                                                                                                check=False,
                                                                                                                )

                                                                                                                                                                                                                                if process.returncode == 0 and \
                                                                                                                    process.stdout:
                                                                                                                    try:
                                                                                                                        outdated_data = json.loads(
                                                                                                                            process.stdout)
                                                                                                                        except json.JSONDecodeError as e:
                                                                                                                            self.logger.error(
                                                                                                                                "Failed to parse pip output: %s",
                                                                                                                                e)
                                                                                                                        return []

                                                                                                                                                                                                                                                for pkg_data in \
                                                                                                                            outdated_data:
                                                                                                                            name = pkg_data["name"].lower()
                                                                                                                            latest_version = pkg_data.get(
                                                                                                                                "latest_version")

                                                                                                                                                                                                                                                        if name in \
                                                                                                                                packages:
                                                                                                                                # Update package info
                                                                                                                                packages[name].latest_version = latest_version
                                                                                                                                packages[name].is_outdated = True

                                                                                                                                # Calculate update priority
                                                                                                                                                                                                                                                                if latest_version and \
                                                                                                                                    packages[name].current_version:
                                                                                                                                    try:
                                                                                                                                        current = version.parse(
                                                                                                                                            packages[name].current_version)
                                                                                                                                        latest = version.parse(
                                                                                                                                            latest_version)

                                                                                                                                        # Higher priority for major updates and security fixes
                                                                                                                                        if latest.major > current.major:
                                                                                                                                            packages[name].update_priority = 8
                                                                                                                                            elif latest.minor > current.minor:
                                                                                                                                            packages[name].update_priority = 5
                                                                                                                                            else:
                                                                                                                                            packages[name].update_priority = 3
                                                                                                                                            except version.InvalidVersion:
                                                                                                                                                self.logger.warning(
                                                                                                                                                    "Invalid version format for %s: {e}",
                                                                                                                                                    name)
                                                                                                                                                packages[name].update_priority = 4

                                                                                                                                                outdated_packages.append(
                                                                                                                                                    packages[name])

                                                                                                                                                # Sort by update priority
                                                                                                                                                outdated_packages.sort(
                                                                                                                                                    key=lambda x: x.update_priority,
                                                                                                                                                    reverse=True)

                                                                                                                                                self.logger.info(
                                                                                                                                                    "Found %s outdated packages.",
                                                                                                                                                    len(outdated_packages))
                                                                                                                                            return outdated_packages

                                                                                                                                            except Exception as e:
                                                                                                                                                self.logger.error(
                                                                                                                                                    "Failed to check outdated packages: %s",
                                                                                                                                                    e)
                                                                                                                                            return []

                                                                                                                                            def generate_optimization_recommendations(
                                                                                                                                                self,
                                                                                                                                                packages: Dict[str, PackageInfo],
                                                                                                                                                outdated_packages: List[PackageInfo],
                                                                                                                                                vulnerable_packages: List[PackageInfo],
                                                                                                                                                ) -> List[str]:
                                                                                                                                                """Generate intelligent optimization recommendations

                                                                                                                                                Args:
                                                                                                                                                packages (
                                                                                                                                                    Dict): Dictionary of package information
                                                                                                                                                outdated_packages (
                                                                                                                                                    List): List of outdated packages
                                                                                                                                                vulnerable_packages (
                                                                                                                                                    List): List of vulnerable packages

                                                                                                                                                Returns:
                                                                                                                                                List of optimization recommendations
                                                                                                                                                """
                                                                                                                                                recommendations = []

                                                                                                                                                try:
                                                                                                                                                    # Security recommendations (highest priority)
                                                                                                                                                    if vulnerable_packages:
                                                                                                                                                        recommendations.append(
                                                                                                                                                        f"CRITICAL: Update {len(
                                                                                                                                                            vulnerable_packages)} packages with security vulnerabilities immediately",
                                                                                                                                                        )

                                                                                                                                                        # Add specific recommendations for top vulnerabilities
                                                                                                                                                                                                                                                                                                                for pkg in \
                                                                                                                                                            vulnerable_packages[:3]:
                                                                                                                                                            recommendations.append(
                                                                                                                                                            f"Update vulnerable package {pkg.name} from {pkg.current_version} "
                                                                                                                                                                                                                                                                                                                        f"to {pkg.latest_version or \
                                                                                                                                                                'latest version'}",
                                                                                                                                                            )

                                                                                                                                                            # Major outdated package recommendations
                                                                                                                                                            major_outdated = [
                                                                                                                                                                                                                                                                                                                        pkg for pkg in outdated_packages if pkg.update_priority >= 8 and \
                                                                                                                                                                pkg not in vulnerable_packages
                                                                                                                                                            ]
                                                                                                                                                            if major_outdated:
                                                                                                                                                                recommendations.append(
                                                                                                                                                                f"Update {len(
                                                                                                                                                                    major_outdated)} packages with major version updates available",
                                                                                                                                                                )

                                                                                                                                                                # Dependency structure recommendations
                                                                                                                                                                try:
                                                                                                                                                                    # Find highly connected packages
                                                                                                                                                                    centrality = nx.degree_centrality(
                                                                                                                                                                        self.dependency_graph)
                                                                                                                                                                    central_packages = sorted(
                                                                                                                                                                    centrality.items(),
                                                                                                                                                                    key=lambda x: x[1],
                                                                                                                                                                    reverse=True,
                                                                                                                                                                    )[:5]

                                                                                                                                                                                                                                                                                                                                        for name, score in \
                                                                                                                                                                        central_packages:
                                                                                                                                                                                                                                                                                                                                                if score > 0.5 and \
                                                                                                                                                                            name in packages:
                                                                                                                                                                            pkg = packages[name]
                                                                                                                                                                                                                                                                                                                                                        if pkg.is_outdated or \
                                                                                                                                                                                pkg.is_vulnerable:
                                                                                                                                                                                recommendations.append(
                                                                                                                                                                                f"Prioritize update of {name} as it has high centrality "
                                                                                                                                                                                f"(
                                                                                                                                                                                    {len(pkg.used_by)} dependent packages)",
                                                                                                                                                                                )
                                                                                                                                                                                except nx.NetworkXError as e:
                                                                                                                                                                                    self.logger.warning(
                                                                                                                                                                                        "Could not analyze package centrality: %s",
                                                                                                                                                                                        e)
                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                        self.logger.error(
                                                                                                                                                                                            "Unexpected error analyzing package centrality: %s",
                                                                                                                                                                                            e)

                                                                                                                                                                                        # Development workflow recommendations
                                                                                                                                                                                        if len(
                                                                                                                                                                                            packages) > 100:
                                                                                                                                                                                            recommendations.append(
                                                                                                                                                                                                                                                                                                                                                                                        "Consider using a dependency management tool like Poetry or \
                                                                                                                                                                                                pipenv "
                                                                                                                                                                                            "to better manage this large number of dependencies",
                                                                                                                                                                                            )

                                                                                                                                                                                        return recommendations

                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                            self.logger.error(
                                                                                                                                                                                                "Failed to generate recommendations: %s",
                                                                                                                                                                                                e)
                                                                                                                                                                                        return ["Error generating recommendations: " + str(
                                                                                                                                                                                            e)]

                                                                                                                                                                                        def update_dependencies(
                                                                                                                                                                                            self,
                                                                                                                                                                                            interactive: bool = True,
                                                                                                                                                                                            only_vulnerable: bool = False,
                                                                                                                                                                                            ) -> Dict[str, Any]:
                                                                                                                                                                                            """Update dependencies based on analysis

                                                                                                                                                                                            Args:
                                                                                                                                                                                            interactive (
                                                                                                                                                                                                bool): Whether to prompt for updates
                                                                                                                                                                                            only_vulnerable (
                                                                                                                                                                                                bool): Whether to only update vulnerable packages

                                                                                                                                                                                            Returns:
                                                                                                                                                                                            Update results
                                                                                                                                                                                            """
                                                                                                                                                                                            try:
                                                                                                                                                                                                # Analyze dependencies first
                                                                                                                                                                                                analysis = self.analyze_dependencies()

                                                                                                                                                                                                # Extract packages to update
                                                                                                                                                                                                if only_vulnerable:
                                                                                                                                                                                                                                                                                                                                                                                                        packages_to_update = [pkg["name"] for pkg in \
                                                                                                                                                                                                        analysis["vulnerable_packages"]]
                                                                                                                                                                                                    else:
                                                                                                                                                                                                                                                                                                                                                                                                        packages_to_update = [pkg["name"] for pkg in \
                                                                                                                                                                                                        analysis["outdated_packages"]]

                                                                                                                                                                                                    if not packages_to_update:
                                                                                                                                                                                                    return {
                                                                                                                                                                                                    "status": "success",
                                                                                                                                                                                                    "updated_packages": [],
                                                                                                                                                                                                    "message": "No packages need updating",
                                                                                                                                                                                                    }

                                                                                                                                                                                                    # Interactive mode
                                                                                                                                                                                                    if interactive:
                                                                                                                                                                                                        self._display_update_prompt(
                                                                                                                                                                                                            analysis,
                                                                                                                                                                                                            packages_to_update)

                                                                                                                                                                                                        # Get user confirmation
                                                                                                                                                                                                        response = input(
                                                                                                                                                                                                            "\nProceed with updates? (y/n): ").strip().lower()
                                                                                                                                                                                                        if response != "y":
                                                                                                                                                                                                        return {
                                                                                                                                                                                                        "status": "cancelled",
                                                                                                                                                                                                        "message": "Update cancelled by user",
                                                                                                                                                                                                        }

                                                                                                                                                                                                        # Perform updates
                                                                                                                                                                                                        updated_packages = []
                                                                                                                                                                                                                                                                                                                                                                                                                for package_name in \
                                                                                                                                                                                                            packages_to_update:
                                                                                                                                                                                                            try:
                                                                                                                                                                                                                self.logger.info(
                                                                                                                                                                                                                    "Updating package: %s",
                                                                                                                                                                                                                    package_name)
                                                                                                                                                                                                                process = subprocess.run(
                                                                                                                                                                                                                [
                                                                                                                                                                                                                sys.executable,
                                                                                                                                                                                                                "-m",
                                                                                                                                                                                                                "pip",
                                                                                                                                                                                                                "install",
                                                                                                                                                                                                                "--upgrade",
                                                                                                                                                                                                                package_name,
                                                                                                                                                                                                                ],
                                                                                                                                                                                                                capture_output=True,
                                                                                                                                                                                                                text=True,
                                                                                                                                                                                                                check=False,
                                                                                                                                                                                                                )

                                                                                                                                                                                                                if process.returncode == 0:
                                                                                                                                                                                                                    updated_packages.append(
                                                                                                                                                                                                                        package_name)
                                                                                                                                                                                                                    else:
                                                                                                                                                                                                                    self.logger.error(
                                                                                                                                                                                                                    f"Failed to update {package_name}: {process.stderr}",
                                                                                                                                                                                                                    )
                                                                                                                                                                                                                    except Exception:
                                                                                                                                                                                                                        self.logger.error(
                                                                                                                                                                                                                            "Error updating %s: {e}",
                                                                                                                                                                                                                            package_name)

                                                                                                                                                                                                                        # Return results
                                                                                                                                                                                                                    return {
                                                                                                                                                                                                                    "status": "success",
                                                                                                                                                                                                                    "updated_packages": updated_packages,
                                                                                                                                                                                                                    "total_updated": len(
                                                                                                                                                                                                                        updated_packages),
                                                                                                                                                                                                                    "total_attempted": len(
                                                                                                                                                                                                                        packages_to_update),
                                                                                                                                                                                                                    }

                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                        self.logger.error(
                                                                                                                                                                                                                            "Dependency update failed: %s",
                                                                                                                                                                                                                            e)
                                                                                                                                                                                                                    return {
                                                                                                                                                                                                                    "status": "error",
                                                                                                                                                                                                                    "error": str(
                                                                                                                                                                                                                        e),
                                                                                                                                                                                                                    }

                                                                                                                                                                                                                    def _display_update_prompt(
                                                                                                                                                                                                                        self,
                                                                                                                                                                                                                        analysis: Dict[str, Any],
                                                                                                                                                                                                                        packages_to_update: List[str],
                                                                                                                                                                                                                        ) -> None:
                                                                                                                                                                                                                        """Display an interactive update prompt

                                                                                                                                                                                                                        Args:
                                                                                                                                                                                                                        analysis (
                                                                                                                                                                                                                            Dict): Dependency analysis results
                                                                                                                                                                                                                        packages_to_update (
                                                                                                                                                                                                                            List): List of packages to update
                                                                                                                                                                                                                        """
                                                                                                                                                                                                                        # Create a nice UI with rich
                                                                                                                                                                                                                        self.console.print()
                                                                                                                                                                                                                        self.console.rule(
                                                                                                                                                                                                                            "[bold blue]SutazAI Dependency Update[/bold blue]")

                                                                                                                                                                                                                        # Summary panel
                                                                                                                                                                                                                        summary = Panel(
                                                                                                                                                                                                                        f"Found [bold]{len(
                                                                                                                                                                                                                            packages_to_update)}[/bold] packages to update\n"
                                                                                                                                                                                                                        f"[bold red]{analysis['vulnerable_count']}[/bold red] with security vulnerabilities\n"
                                                                                                                                                                                                                        f"[bold yellow]{analysis['outdated_count']}[/bold yellow] outdated packages",
                                                                                                                                                                                                                        title="Update Summary",
                                                                                                                                                                                                                        expand=False,
                                                                                                                                                                                                                        )
                                                                                                                                                                                                                        self.console.print(
                                                                                                                                                                                                                            summary)

                                                                                                                                                                                                                        # Create table of packages to update
                                                                                                                                                                                                                        table = Table(
                                                                                                                                                                                                                            title="Packages to Update")
                                                                                                                                                                                                                        table.add_column(
                                                                                                                                                                                                                            "Package",
                                                                                                                                                                                                                            style="cyan")
                                                                                                                                                                                                                        table.add_column(
                                                                                                                                                                                                                            "Current Version",
                                                                                                                                                                                                                            style="yellow")
                                                                                                                                                                                                                        table.add_column(
                                                                                                                                                                                                                            "Latest Version",
                                                                                                                                                                                                                            style="green")
                                                                                                                                                                                                                        table.add_column(
                                                                                                                                                                                                                            "Status",
                                                                                                                                                                                                                            style="bold")

                                                                                                                                                                                                                                                                                                                                                                                                                                                for pkg_name in \
                                                                                                                                                                                                                            packages_to_update:
                                                                                                                                                                                                                            # Find package in analysis
                                                                                                                                                                                                                            pkg_data = None
                                                                                                                                                                                                                                                                                                                                                                                                                                                        for pkg in \
                                                                                                                                                                                                                                analysis["vulnerable_packages"]:
                                                                                                                                                                                                                                if pkg["name"] == pkg_name:
                                                                                                                                                                                                                                    pkg_data = pkg
                                                                                                                                                                                                                                    status = "[bold red]Vulnerable[/bold red]"
                                                                                                                                                                                                                                break

                                                                                                                                                                                                                                if not pkg_data:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        for pkg in \
                                                                                                                                                                                                                                        analysis["outdated_packages"]:
                                                                                                                                                                                                                                        if pkg["name"] == pkg_name:
                                                                                                                                                                                                                                            pkg_data = pkg
                                                                                                                                                                                                                                            status = "[yellow]Outdated[/yellow]"
                                                                                                                                                                                                                                        break

                                                                                                                                                                                                                                        if pkg_data:
                                                                                                                                                                                                                                            table.add_row(
                                                                                                                                                                                                                                            pkg_data["name"],
                                                                                                                                                                                                                                            pkg_data["current_version"],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        pkg_data["latest_version"] or \
                                                                                                                                                                                                                                                "Unknown",
                                                                                                                                                                                                                                            status,
                                                                                                                                                                                                                                            )

                                                                                                                                                                                                                                            self.console.print(
                                                                                                                                                                                                                                                table)

                                                                                                                                                                                                                                            def generate_dependency_report(
                                                                                                                                                                                                                                                self) -> DependencyReport:
                                                                                                                                                                                                                                                """Generate comprehensive dependency report

                                                                                                                                                                                                                                                Returns:
                                                                                                                                                                                                                                                Detailed dependency report
                                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                                # Run dependency analysis
                                                                                                                                                                                                                                                analysis = self.analyze_dependencies()

                                                                                                                                                                                                                                                # Convert to PackageInfo objects
                                                                                                                                                                                                                                                outdated_dependencies = []
                                                                                                                                                                                                                                                for pkg_data in analysis.get(
                                                                                                                                                                                                                                                    "outdated_packages",
                                                                                                                                                                                                                                                    []):
                                                                                                                                                                                                                                                    outdated_dependencies.append(
                                                                                                                                                                                                                                                    PackageInfo(
                                                                                                                                                                                                                                                    name=pkg_data["name"],
                                                                                                                                                                                                                                                    current_version=pkg_data["current_version"],
                                                                                                                                                                                                                                                    latest_version=pkg_data["latest_version"],
                                                                                                                                                                                                                                                    is_outdated=True,
                                                                                                                                                                                                                                                    update_priority=pkg_data["update_priority"],
                                                                                                                                                                                                                                                    ),
                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                    vulnerable_dependencies = []
                                                                                                                                                                                                                                                    for pkg_data in analysis.get(
                                                                                                                                                                                                                                                        "vulnerable_packages",
                                                                                                                                                                                                                                                        []):
                                                                                                                                                                                                                                                        vulnerable_dependencies.append(
                                                                                                                                                                                                                                                        PackageInfo(
                                                                                                                                                                                                                                                        name=pkg_data["name"],
                                                                                                                                                                                                                                                        current_version=pkg_data["current_version"],
                                                                                                                                                                                                                                                        latest_version=pkg_data["latest_version"],
                                                                                                                                                                                                                                                        is_vulnerable=True,
                                                                                                                                                                                                                                                        vulnerability_details=pkg_data["vulnerability_details"],
                                                                                                                                                                                                                                                        update_priority=pkg_data["update_priority"],
                                                                                                                                                                                                                                                        ),
                                                                                                                                                                                                                                                        )

                                                                                                                                                                                                                                                        # Create report
                                                                                                                                                                                                                                                        report = DependencyReport(
                                                                                                                                                                                                                                                        timestamp=datetime.now(
                                                                                                                                                                                                                                                            ).isoformat(),
                                                                                                                                                                                                                                                        total_dependencies=analysis.get(
                                                                                                                                                                                                                                                            "total_packages",
                                                                                                                                                                                                                                                            0),
                                                                                                                                                                                                                                                        outdated_dependencies=outdated_dependencies,
                                                                                                                                                                                                                                                        vulnerable_dependencies=vulnerable_dependencies,
                                                                                                                                                                                                                                                        dependency_graph=nx.to_dict_of_lists(
                                                                                                                                                                                                                                                            self.dependency_graph),
                                                                                                                                                                                                                                                        optimization_recommendations=analysis.get(
                                                                                                                                                                                                                                                            "recommendations",
                                                                                                                                                                                                                                                            []),
                                                                                                                                                                                                                                                        )

                                                                                                                                                                                                                                                        # Save report to file
                                                                                                                                                                                                                                                        self._save_dependency_report(
                                                                                                                                                                                                                                                            report)

                                                                                                                                                                                                                                                    return report

                                                                                                                                                                                                                                                    def _save_dependency_report(
                                                                                                                                                                                                                                                        self,
                                                                                                                                                                                                                                                        report: DependencyReport) -> None:
                                                                                                                                                                                                                                                        """Save dependency report to file

                                                                                                                                                                                                                                                        Args:
                                                                                                                                                                                                                                                        report (
                                                                                                                                                                                                                                                            DependencyReport): Dependency report to save
                                                                                                                                                                                                                                                        """
                                                                                                                                                                                                                                                        try:
                                                                                                                                                                                                                                                            # Ensure log directory exists
                                                                                                                                                                                                                                                            os.makedirs(
                                                                                                                                                                                                                                                                self.log_dir,
                                                                                                                                                                                                                                                                exist_ok=True)

                                                                                                                                                                                                                                                            # Generate filename
                                                                                                                                                                                                                                                            timestamp = datetime.now(
                                                                                                                                                                                                                                                                ).strftime("%Y%m%d_%H%M%S")
                                                                                                                                                                                                                                                            filename = os.path.join(
                                                                                                                                                                                                                                                                self.log_dir,
                                                                                                                                                                                                                                                                f"dependency_report_{timestamp}.json")

                                                                                                                                                                                                                                                            # Convert to dictionary and save
                                                                                                                                                                                                                                                            with open(
                                                                                                                                                                                                                                                                filename,
                                                                                                                                                                                                                                                                "w") as f:
                                                                                                                                                                                                                                                            json.dump(
                                                                                                                                                                                                                                                                asdict(report),
                                                                                                                                                                                                                                                                f,
                                                                                                                                                                                                                                                                indent=2)

                                                                                                                                                                                                                                                            self.logger.info(
                                                                                                                                                                                                                                                                "Dependency report saved to %s",
                                                                                                                                                                                                                                                                filename)

                                                                                                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                                                                                                self.logger.error(
                                                                                                                                                                                                                                                                    "Failed to save dependency report: %s",
                                                                                                                                                                                                                                                                    e)

                                                                                                                                                                                                                                                                def visualize_dependency_graph(
                                                                                                                                                                                                                                                                    self) -> None:
                                                                                                                                                                                                                                                                    """Visualize dependency graph using rich"""
                                                                                                                                                                                                                                                                    try:
                                                                                                                                                                                                                                                                        if not self.dependency_graph.nodes():
                                                                                                                                                                                                                                                                            self.console.print(
                                                                                                                                                                                                                                                                            "[yellow]No dependency graph available. Run analyze_dependencies() first.[/yellow]",
                                                                                                                                                                                                                                                                            )
                                                                                                                                                                                                                                                                        return

                                                                                                                                                                                                                                                                        # Generate a simplified text representation of the graph
                                                                                                                                                                                                                                                                        self.console.print()
                                                                                                                                                                                                                                                                        self.console.rule(
                                                                                                                                                                                                                                                                            "[bold blue]SutazAI Dependency Graph[/bold blue]")

                                                                                                                                                                                                                                                                        # Find central packages
                                                                                                                                                                                                                                                                        centrality = nx.degree_centrality(
                                                                                                                                                                                                                                                                            self.dependency_graph)
                                                                                                                                                                                                                                                                        central_packages = sorted(
                                                                                                                                                                                                                                                                        centrality.items(),
                                                                                                                                                                                                                                                                        key=lambda x: x[1],
                                                                                                                                                                                                                                                                        reverse=True,
                                                                                                                                                                                                                                                                        )[:10]

                                                                                                                                                                                                                                                                        # Create table of central packages
                                                                                                                                                                                                                                                                        table = Table(
                                                                                                                                                                                                                                                                            title="Most Central Packages")
                                                                                                                                                                                                                                                                        table.add_column(
                                                                                                                                                                                                                                                                            "Package",
                                                                                                                                                                                                                                                                            style="cyan")
                                                                                                                                                                                                                                                                        table.add_column(
                                                                                                                                                                                                                                                                            "Centrality Score",
                                                                                                                                                                                                                                                                            style="magenta")
                                                                                                                                                                                                                                                                        table.add_column(
                                                                                                                                                                                                                                                                            "Dependencies",
                                                                                                                                                                                                                                                                            style="green")
                                                                                                                                                                                                                                                                        table.add_column(
                                                                                                                                                                                                                                                                            "Dependents",
                                                                                                                                                                                                                                                                            style="yellow")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                for pkg_name, score in \
                                                                                                                                                                                                                                                                            central_packages:
                                                                                                                                                                                                                                                                            dependencies = list(
                                                                                                                                                                                                                                                                                self.dependency_graph.successors(pkg_name))
                                                                                                                                                                                                                                                                            dependents = list(
                                                                                                                                                                                                                                                                                self.dependency_graph.predecessors(pkg_name))

                                                                                                                                                                                                                                                                            table.add_row(
                                                                                                                                                                                                                                                                            pkg_name,
                                                                                                                                                                                                                                                                            f"{score:.3f}",
                                                                                                                                                                                                                                                                            str(
                                                                                                                                                                                                                                                                                len(dependencies)),
                                                                                                                                                                                                                                                                            str(
                                                                                                                                                                                                                                                                                len(dependents)),
                                                                                                                                                                                                                                                                            )

                                                                                                                                                                                                                                                                            self.console.print(
                                                                                                                                                                                                                                                                                table)

                                                                                                                                                                                                                                                                            # Display some graph statistics
                                                                                                                                                                                                                                                                            self.console.print()
                                                                                                                                                                                                                                                                            self.console.print(
                                                                                                                                                                                                                                                                            f"Total packages: [bold]{len(
                                                                                                                                                                                                                                                                                self.dependency_graph.nodes())}[/bold]",
                                                                                                                                                                                                                                                                            )
                                                                                                                                                                                                                                                                            self.console.print(
                                                                                                                                                                                                                                                                            f"Total dependencies: [bold]{len(
                                                                                                                                                                                                                                                                                self.dependency_graph.edges())}[/bold]",
                                                                                                                                                                                                                                                                            )

                                                                                                                                                                                                                                                                            # Find isolated packages
                                                                                                                                                                                                                                                                            isolated = [node for node in self.dependency_graph.nodes(
                                                                                                                                                                                                                                                                                ) if self.dependency_graph.degree(node) == 0]
                                                                                                                                                                                                                                                                            if isolated:
                                                                                                                                                                                                                                                                                self.console.print(
                                                                                                                                                                                                                                                                                f"Isolated packages: [bold yellow]{len(
                                                                                                                                                                                                                                                                                    isolated)}[/bold yellow]",
                                                                                                                                                                                                                                                                                )
                                                                                                                                                                                                                                                                                self.console.print(
                                                                                                                                                                                                                                                                                ", ".join(
                                                                                                                                                                                                                                                                                    isolated[:5]) + ("..." if len(isolated) > 5 else ""),
                                                                                                                                                                                                                                                                                )

                                                                                                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                                                                                                    self.logger.error(
                                                                                                                                                                                                                                                                                        "Failed to visualize dependency graph: %s",
                                                                                                                                                                                                                                                                                        e)
                                                                                                                                                                                                                                                                                    self.console.print(
                                                                                                                                                                                                                                                                                    f"[bold red]Error visualizing dependency graph: {e}[/bold red]",
                                                                                                                                                                                                                                                                                    )


                                                                                                                                                                                                                                                                                    def main():
                                                                                                                                                                                                                                                                                        """Main execution function"""
                                                                                                                                                                                                                                                                                        # Verify Python version
                                                                                                                                                                                                                                                                                        verify_python_version()

                                                                                                                                                                                                                                                                                        try:
                                                                                                                                                                                                                                                                                            manager = UnifiedDependencyManager()

                                                                                                                                                                                                                                                                                            # Run comprehensive dependency analysis
                                                                                                                                                                                                                                                                                            analysis = manager.analyze_dependencies()

                                                                                                                                                                                                                                                                                            # Visualize the dependency graph
                                                                                                                                                                                                                                                                                            manager.visualize_dependency_graph()

                                                                                                                                                                                                                                                                                            # Generate and display optimization recommendations
                                                                                                                                                                                                                                                                                            if analysis.get(
                                                                                                                                                                                                                                                                                                "recommendations"):
                                                                                                                                                                                                                                                                                                print(
                                                                                                                                                                                                                                                                                                    "\nOptimization Recommendations:")
                                                                                                                                                                                                                                                                                                for i, rec in enumerate(
                                                                                                                                                                                                                                                                                                    analysis["recommendations"],
                                                                                                                                                                                                                                                                                                    1):
                                                                                                                                                                                                                                                                                                    print(
                                                                                                                                                                                                                                                                                                        f"{i}. {rec}")

                                                                                                                                                                                                                                                                                                    # Generate comprehensive report
                                                                                                                                                                                                                                                                                                    report = manager.generate_dependency_report()
                                                                                                                                                                                                                                                                                                    print(
                                                                                                                                                                                                                                                                                                    f"\nDependency report generated with {report.total_dependencies} packages analyzed.",
                                                                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                                                                                                                                        print(
                                                                                                                                                                                                                                                                                                            f"Dependency management failed: {e}")
                                                                                                                                                                                                                                                                                                        sys.exit(
                                                                                                                                                                                                                                                                                                            1)


                                                                                                                                                                                                                                                                                                        if __name__ == "__main__":
                                                                                                                                                                                                                                                                                                            main()
