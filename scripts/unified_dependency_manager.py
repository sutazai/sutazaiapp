#!/usr/bin/env python3
"""
SutazAI Unified Dependency Manager

Advanced dependency management system providing:
- Comprehensive dependency tracking and analysis
- Vulnerability detection and security assessment
- Dependency graph visualization
- Automated dependency updates and conflict resolution
- Package optimization recommendations
"""

import importlib
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pkg_resources
import requests
from packaging import version
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


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


@dataclass
class PackageInfo:
    """
    Comprehensive package information
    """
    name: str
    current_version: str
    latest_version: Optional[str] = None
    is_outdated: bool = False
    is_vulnerable: bool = False
    vulnerability_details: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    used_by: List[str] = field(default_factory=list)
    update_priority: int = 0  # 0-10, 10 being highest priority


@dataclass
class DependencyReport:
    """
    Comprehensive dependency management report
    """
    timestamp: str
    total_dependencies: int
    outdated_dependencies: List[PackageInfo]
    vulnerable_dependencies: List[PackageInfo]
    dependency_graph: Dict[str, Any]
    optimization_recommendations: List[str]


class UnifiedDependencyManager:
    """
    Advanced dependency management system with comprehensive capabilities
    """

    def __init__(
        self,
        requirements_path: str = "/opt/sutazaiapp/requirements.txt",
        policy_path: str = "/opt/sutazaiapp/config/dependency_policy.yml",
        log_dir: str = "/opt/sutazaiapp/logs",
    ):
        """
        Initialize unified dependency manager

        Args:
            requirements_path (str): Path to requirements.txt file
            policy_path (str): Path to dependency policy configuration
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
        log_file = os.path.join(log_dir, "dependency_manager.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("SutazAI.DependencyManager")

        # Package registry for API access
        self.pypi_url = "https://pypi.org/pypi"
        
        # Initialize dependency graph
        self.dependency_graph = nx.DiGraph()

    def analyze_dependencies(self) -> Dict[str, Any]:
        """
        Perform comprehensive dependency analysis

        Returns:
            Detailed dependency analysis report
        """
        try:
            # Get currently installed packages
            installed_packages = self._get_installed_packages()
            
            # Build dependency graph
            self._build_dependency_graph(installed_packages)
            
            # Check for outdated packages
            outdated_packages = self._check_outdated_packages(installed_packages)
            
            # Check for vulnerabilities
            vulnerable_packages = self._check_vulnerabilities(installed_packages)
            
            # Analyze for optimization
            optimization_recommendations = self.generate_optimization_recommendations(
                installed_packages, outdated_packages, vulnerable_packages
            )
            
            # Create analysis report
            analysis_report = {
                "timestamp": datetime.now().isoformat(),
                "total_packages": len(installed_packages),
                "outdated_packages": [asdict(pkg) for pkg in outdated_packages],
                "vulnerable_packages": [asdict(pkg) for pkg in vulnerable_packages],
                "outdated_count": len(outdated_packages),
                "vulnerable_count": len(vulnerable_packages),
                "recommendations": optimization_recommendations,
            }
            
            self.logger.info(
                f"Dependency analysis completed: "
                f"{analysis_report['total_packages']} packages analyzed, "
                f"{analysis_report['outdated_count']} outdated, "
                f"{analysis_report['vulnerable_count']} vulnerable."
            )
            
            return analysis_report
            
        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _get_installed_packages(self) -> Dict[str, PackageInfo]:
        """
        Get information about all installed packages

        Returns:
            Dictionary of installed packages with comprehensive information
        """
        installed_packages = {}
        
        try:
            # Get all installed packages
            for pkg in pkg_resources.working_set:
                name = pkg.key
                current_version = pkg.version
                
                # Create package info
                package_info = PackageInfo(
                    name=name,
                    current_version=current_version,
                )
                
                # Get direct dependencies
                try:
                    package_info.dependencies = [
                        d.key for d in pkg.requires()
                    ]
                except Exception:
                    # Some packages might have distribution issues
                    pass
                
                installed_packages[name] = package_info
                
            # Build "used by" relationships
            for name, pkg_info in installed_packages.items():
                for dep in pkg_info.dependencies:
                    if dep in installed_packages:
                        installed_packages[dep].used_by.append(name)
            
            return installed_packages
            
        except Exception as e:
            self.logger.error(f"Failed to get installed packages: {e}")
            return {}

    def _build_dependency_graph(self, packages: Dict[str, PackageInfo]) -> None:
        """
        Build a comprehensive dependency graph

        Args:
            packages (Dict): Dictionary of package information
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
                for dep in pkg_info.dependencies:
                    if dep in packages:
                        self.dependency_graph.add_edge(name, dep)
            
            self.logger.info(
                f"Dependency graph built: {len(self.dependency_graph.nodes())} nodes, "
                f"{len(self.dependency_graph.edges())} edges."
            )
            
        except Exception as e:
            self.logger.error(f"Failed to build dependency graph: {e}")

    def _check_outdated_packages(
        self, packages: Dict[str, PackageInfo]
    ) -> List[PackageInfo]:
        """
        Check for outdated packages

        Args:
            packages (Dict): Dictionary of package information

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
            
            if process.returncode == 0 and process.stdout:
                outdated_data = json.loads(process.stdout)
                
                for pkg_data in outdated_data:
                    name = pkg_data["name"].lower()
                    latest_version = pkg_data.get("latest_version")
                    
                    if name in packages:
                        # Update package info
                        packages[name].latest_version = latest_version
                        packages[name].is_outdated = True
                        
                        # Calculate update priority
                        if latest_version and packages[name].current_version:
                            try:
                                current = version.parse(packages[name].current_version)
                                latest = version.parse(latest_version)
                                
                                # Higher priority for major updates and security fixes
                                if latest.major > current.major:
                                    packages[name].update_priority = 8
                                elif latest.minor > current.minor:
                                    packages[name].update_priority = 5
                                else:
                                    packages[name].update_priority = 3
                            except Exception:
                                # Can't parse version, assume moderate priority
                                packages[name].update_priority = 4
                        
                        outdated_packages.append(packages[name])
            
            # Sort by update priority
            outdated_packages.sort(key=lambda x: x.update_priority, reverse=True)
            
            self.logger.info(f"Found {len(outdated_packages)} outdated packages.")
            return outdated_packages
            
        except Exception as e:
            self.logger.error(f"Failed to check outdated packages: {e}")
            return []

    def _check_vulnerabilities(
        self, packages: Dict[str, PackageInfo]
    ) -> List[PackageInfo]:
        """
        Check for security vulnerabilities in dependencies

        Args:
            packages (Dict): Dictionary of package information

        Returns:
            List of vulnerable packages
        """
        vulnerable_packages = []
        
        try:
            # Run safety check for vulnerabilities
            process = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            
            if process.stdout:
                try:
                    # Parse safety output
                    vulnerabilities = json.loads(process.stdout)
                    
                    for vuln in vulnerabilities:
                        name = vuln[0].lower()
                        
                        if name in packages:
                            # Update package info
                            packages[name].is_vulnerable = True
                            packages[name].vulnerability_details.append({
                                "vulnerability_id": vuln[1],
                                "affected_versions": vuln[2],
                                "description": vuln[3],
                            })
                            
                            # Set highest update priority for vulnerable packages
                            packages[name].update_priority = 10
                            
                            vulnerable_packages.append(packages[name])
                except json.JSONDecodeError:
                    # Safety might not return valid JSON
                    pass
            
            self.logger.info(f"Found {len(vulnerable_packages)} vulnerable packages.")
            return vulnerable_packages
            
        except Exception as e:
            self.logger.error(f"Failed to check vulnerabilities: {e}")
            return []

    def generate_optimization_recommendations(
        self,
        packages: Dict[str, PackageInfo],
        outdated_packages: List[PackageInfo],
        vulnerable_packages: List[PackageInfo],
    ) -> List[str]:
        """
        Generate intelligent optimization recommendations

        Args:
            packages (Dict): Dictionary of package information
            outdated_packages (List): List of outdated packages
            vulnerable_packages (List): List of vulnerable packages

        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        try:
            # Security recommendations (highest priority)
            if vulnerable_packages:
                recommendations.append(
                    f"CRITICAL: Update {len(vulnerable_packages)} packages with security vulnerabilities immediately"
                )
                
                # Add specific recommendations for top vulnerabilities
                for pkg in vulnerable_packages[:3]:
                    recommendations.append(
                        f"Update vulnerable package {pkg.name} from {pkg.current_version} "
                        f"to {pkg.latest_version or 'latest version'}"
                    )
            
            # Major outdated package recommendations
            major_outdated = [
                pkg for pkg in outdated_packages
                if pkg.update_priority >= 8 and pkg not in vulnerable_packages
            ]
            if major_outdated:
                recommendations.append(
                    f"Update {len(major_outdated)} packages with major version updates available"
                )
            
            # Dependency structure recommendations
            try:
                # Find highly connected packages
                centrality = nx.degree_centrality(self.dependency_graph)
                central_packages = sorted(
                    centrality.items(), key=lambda x: x[1], reverse=True
                )[:5]
                
                for name, score in central_packages:
                    if score > 0.5 and name in packages:
                        pkg = packages[name]
                        if pkg.is_outdated or pkg.is_vulnerable:
                            recommendations.append(
                                f"Prioritize update of {name} as it has high centrality "
                                f"({len(pkg.used_by)} dependent packages)"
                            )
            except Exception:
                # Graph analysis might fail
                pass
            
            # Development workflow recommendations
            if len(packages) > 100:
                recommendations.append(
                    "Consider using dependency groups or environment markers to reduce "
                    "the number of installed dependencies"
                )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            return ["Error generating recommendations: " + str(e)]

    def update_dependencies(
        self, interactive: bool = True, only_vulnerable: bool = False
    ) -> Dict[str, Any]:
        """
        Update dependencies based on analysis

        Args:
            interactive (bool): Whether to prompt for updates
            only_vulnerable (bool): Whether to only update vulnerable packages

        Returns:
            Update results
        """
        try:
            # Analyze dependencies first
            analysis = self.analyze_dependencies()
            
            # Extract packages to update
            if only_vulnerable:
                packages_to_update = [
                    pkg["name"] for pkg in analysis["vulnerable_packages"]
                ]
            else:
                packages_to_update = [
                    pkg["name"] for pkg in analysis["outdated_packages"]
                ]
            
            if not packages_to_update:
                return {
                    "status": "success",
                    "updated_packages": [],
                    "message": "No packages need updating",
                }
            
            # Interactive mode
            if interactive:
                self._display_update_prompt(analysis, packages_to_update)
                
                # Get user confirmation
                response = input("\nProceed with updates? (y/n): ").strip().lower()
                if response != "y":
                    return {
                        "status": "cancelled",
                        "message": "Update cancelled by user",
                    }
            
            # Perform updates
            updated_packages = []
            for package_name in packages_to_update:
                try:
                    self.logger.info(f"Updating package: {package_name}")
                    process = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--upgrade", package_name],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    
                    if process.returncode == 0:
                        updated_packages.append(package_name)
                    else:
                        self.logger.error(
                            f"Failed to update {package_name}: {process.stderr}"
                        )
                except Exception as e:
                    self.logger.error(f"Error updating {package_name}: {e}")
            
            # Return results
            return {
                "status": "success",
                "updated_packages": updated_packages,
                "total_updated": len(updated_packages),
                "total_attempted": len(packages_to_update),
            }
            
        except Exception as e:
            self.logger.error(f"Dependency update failed: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    def _display_update_prompt(
        self, analysis: Dict[str, Any], packages_to_update: List[str]
    ) -> None:
        """
        Display an interactive update prompt

        Args:
            analysis (Dict): Dependency analysis results
            packages_to_update (List): List of packages to update
        """
        # Create a nice UI with rich
        self.console.print()
        self.console.rule("[bold blue]SutazAI Dependency Update[/bold blue]")
        
        # Summary panel
        summary = Panel(
            f"Found [bold]{len(packages_to_update)}[/bold] packages to update\n"
            f"[bold red]{analysis['vulnerable_count']}[/bold red] with security vulnerabilities\n"
            f"[bold yellow]{analysis['outdated_count']}[/bold yellow] outdated packages",
            title="Update Summary",
            expand=False,
        )
        self.console.print(summary)
        
        # Create table of packages to update
        table = Table(title="Packages to Update")
        table.add_column("Package", style="cyan")
        table.add_column("Current Version", style="yellow")
        table.add_column("Latest Version", style="green")
        table.add_column("Status", style="bold")
        
        for pkg_name in packages_to_update:
            # Find package in analysis
            pkg_data = None
            for pkg in analysis["vulnerable_packages"]:
                if pkg["name"] == pkg_name:
                    pkg_data = pkg
                    status = "[bold red]Vulnerable[/bold red]"
                    break
                    
            if not pkg_data:
                for pkg in analysis["outdated_packages"]:
                    if pkg["name"] == pkg_name:
                        pkg_data = pkg
                        status = "[yellow]Outdated[/yellow]"
                        break
            
            if pkg_data:
                table.add_row(
                    pkg_data["name"],
                    pkg_data["current_version"],
                    pkg_data["latest_version"] or "Unknown",
                    status,
                )
        
        self.console.print(table)

    def generate_dependency_report(self) -> DependencyReport:
        """
        Generate comprehensive dependency report

        Returns:
            Detailed dependency report
        """
        # Run dependency analysis
        analysis = self.analyze_dependencies()
        
        # Convert to PackageInfo objects
        outdated_dependencies = []
        for pkg_data in analysis.get("outdated_packages", []):
            outdated_dependencies.append(PackageInfo(
                name=pkg_data["name"],
                current_version=pkg_data["current_version"],
                latest_version=pkg_data["latest_version"],
                is_outdated=True,
                update_priority=pkg_data["update_priority"],
            ))
            
        vulnerable_dependencies = []
        for pkg_data in analysis.get("vulnerable_packages", []):
            vulnerable_dependencies.append(PackageInfo(
                name=pkg_data["name"],
                current_version=pkg_data["current_version"],
                latest_version=pkg_data["latest_version"],
                is_vulnerable=True,
                vulnerability_details=pkg_data["vulnerability_details"],
                update_priority=pkg_data["update_priority"],
            ))
        
        # Create report
        report = DependencyReport(
            timestamp=datetime.now().isoformat(),
            total_dependencies=analysis.get("total_packages", 0),
            outdated_dependencies=outdated_dependencies,
            vulnerable_dependencies=vulnerable_dependencies,
            dependency_graph=nx.to_dict_of_lists(self.dependency_graph),
            optimization_recommendations=analysis.get("recommendations", []),
        )
        
        # Save report to file
        self._save_dependency_report(report)
        
        return report

    def _save_dependency_report(self, report: DependencyReport) -> None:
        """
        Save dependency report to file

        Args:
            report (DependencyReport): Dependency report to save
        """
        try:
            # Ensure log directory exists
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.log_dir, f"dependency_report_{timestamp}.json"
            )
            
            # Convert to dictionary and save
            with open(filename, "w") as f:
                json.dump(asdict(report), f, indent=2)
                
            self.logger.info(f"Dependency report saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save dependency report: {e}")

    def visualize_dependency_graph(self) -> None:
        """
        Visualize dependency graph using rich
        """
        try:
            if not self.dependency_graph.nodes():
                self.console.print("[yellow]No dependency graph available. Run analyze_dependencies() first.[/yellow]")
                return
                
            # Generate a simplified text representation of the graph
            self.console.print()
            self.console.rule("[bold blue]SutazAI Dependency Graph[/bold blue]")
            
            # Find central packages
            centrality = nx.degree_centrality(self.dependency_graph)
            central_packages = sorted(
                centrality.items(), key=lambda x: x[1], reverse=True
            )[:10]
            
            # Create table of central packages
            table = Table(title="Most Central Packages")
            table.add_column("Package", style="cyan")
            table.add_column("Centrality Score", style="magenta")
            table.add_column("Dependencies", style="green")
            table.add_column("Dependents", style="yellow")
            
            for pkg_name, score in central_packages:
                dependencies = list(self.dependency_graph.successors(pkg_name))
                dependents = list(self.dependency_graph.predecessors(pkg_name))
                
                table.add_row(
                    pkg_name,
                    f"{score:.3f}",
                    str(len(dependencies)),
                    str(len(dependents)),
                )
            
            self.console.print(table)
            
            # Display some graph statistics
            self.console.print()
            self.console.print(f"Total packages: [bold]{len(self.dependency_graph.nodes())}[/bold]")
            self.console.print(f"Total dependencies: [bold]{len(self.dependency_graph.edges())}[/bold]")
            
            # Find isolated packages
            isolated = [
                node for node in self.dependency_graph.nodes()
                if self.dependency_graph.degree(node) == 0
            ]
            if isolated:
                self.console.print(f"Isolated packages: [bold yellow]{len(isolated)}[/bold yellow]")
                self.console.print(", ".join(isolated[:5]) + ("..." if len(isolated) > 5 else ""))
            
        except Exception as e:
            self.logger.error(f"Failed to visualize dependency graph: {e}")
            self.console.print(f"[bold red]Error visualizing dependency graph: {e}[/bold red]")


def main():
    """
    Main execution function
    """
    # Verify Python version
    verify_python_version()
    
    try:
        manager = UnifiedDependencyManager()
        
        # Run comprehensive dependency analysis
        analysis = manager.analyze_dependencies()
        
        # Visualize the dependency graph
        manager.visualize_dependency_graph()
        
        # Generate and display optimization recommendations
        if analysis.get("recommendations"):
            print("\nOptimization Recommendations:")
            for i, rec in enumerate(analysis["recommendations"], 1):
                print(f"{i}. {rec}")
        
        # Generate comprehensive report
        report = manager.generate_dependency_report()
        print(f"\nDependency report generated with {report.total_dependencies} packages analyzed.")
        
    except Exception as e:
        print(f"Dependency management failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 