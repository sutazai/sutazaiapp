#!/usr/bin/env python3
"""
SutazAI Advanced Dependency Tracking and Management System

Provides comprehensive dependency analysis, tracking,
and intelligent optimization capabilities.
"""

import ast
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import networkx as nx
import pkg_resources


class AdvancedDependencyTracker:
    """
    Ultra-Comprehensive Dependency Tracking and Analysis Framework

    Key Capabilities:
    - Detailed module and package dependency mapping
    - Circular dependency detection
    - Compatibility and conflict analysis
    - Intelligent dependency optimization
    """

    def __init__(
        self,
        base_dir: str = "/opt/sutazai_project/SutazAI",
        log_dir: Optional[str] = None,
    ):
        """
        Initialize Advanced Dependency Tracker

        Args:
            base_dir (str): Base project directory
            log_dir (Optional[str]): Custom log directory
        """
        self.base_dir = base_dir
        self.log_dir = log_dir or os.path.join(base_dir, "logs", "dependency_tracking")

        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            filename=os.path.join(self.log_dir, "dependency_tracker.log"),
        )
        self.logger = logging.getLogger("SutazAI.DependencyTracker")

        # Initialize dependency graphs
        self.module_dependency_graph = nx.DiGraph()
        self.package_dependency_graph = nx.DiGraph()

    def generate_module_dependency_graph(self) -> nx.DiGraph:
        """
        Generate a comprehensive module dependency graph

        Returns:
            NetworkX Directed Graph of module dependencies
        """
        try:
            # Reset graph
            self.module_dependency_graph = nx.DiGraph()

            # Walk through Python files
            for root, _, files in os.walk(self.base_dir):
                for file in files:
                    if file.endswith(".py"):
                        full_path = os.path.join(root, file)

                        try:
                            # Convert file path to module name
                            module_name = os.path.relpath(
                                full_path, self.base_dir
                            ).replace("/", ".")[:-3]

                            # Add module as a node
                            self.module_dependency_graph.add_node(module_name)

                            # Analyze module dependencies
                            module_dependencies = self._analyze_module_dependencies(
                                full_path
                            )

                            # Add edges for dependencies
                            for dep in module_dependencies:
                                if dep != module_name:
                                    self.module_dependency_graph.add_edge(
                                        module_name, dep
                                    )

                        except Exception as e:
                            self.logger.warning(
                                f"Module dependency analysis failed for {full_path}: {e}"
                            )

            return self.module_dependency_graph

        except Exception as e:
            self.logger.error(f"Module dependency graph generation failed: {e}")
            return nx.DiGraph()

    def _analyze_module_dependencies(self, file_path: str) -> List[str]:
        """
        Analyze dependencies for a specific module

        Args:
            file_path (str): Path to the Python file

        Returns:
            List of detected module dependencies
        """
        try:
            with open(file_path, "r") as f:
                source_code = f.read()

            # Parse module with AST
            module = ast.parse(source_code)

            dependencies = []

            # Track imports
            for node in ast.walk(module):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name.split(".")[0])

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module.split(".")[0])

            return list(set(dependencies))

        except Exception as e:
            self.logger.warning(f"Dependency tracking failed for {file_path}: {e}")
            return []

    def detect_circular_dependencies(
        self, dependency_graph: nx.DiGraph
    ) -> List[List[str]]:
        """
        Detect circular dependencies in the module graph

        Args:
            dependency_graph (nx.DiGraph): Module dependency graph

        Returns:
            List of circular dependency cycles
        """
        try:
            # Find all cycles in the graph
            cycles = list(nx.simple_cycles(dependency_graph))

            if cycles:
                self.logger.warning(
                    f"Detected {len(cycles)} circular dependency cycles"
                )
                for cycle in cycles:
                    self.logger.warning(
                        f"Circular Dependency Cycle: {' -> '.join(cycle)}"
                    )

            return cycles

        except Exception as e:
            self.logger.error(f"Circular dependency detection failed: {e}")
            return []

    def generate_package_dependency_graph(self) -> nx.DiGraph:
        """
        Generate a comprehensive package dependency graph

        Returns:
            NetworkX Directed Graph of package dependencies
        """
        try:
            # Reset graph
            self.package_dependency_graph = nx.DiGraph()

            # Get installed packages
            for pkg in pkg_resources.working_set:
                self.package_dependency_graph.add_node(pkg.key, version=pkg.version)

                # Add edges for dependencies
                for req in pkg.requires():
                    self.package_dependency_graph.add_edge(pkg.key, req.key)

            return self.package_dependency_graph

        except Exception as e:
            self.logger.error(f"Package dependency graph generation failed: {e}")
            return nx.DiGraph()

    def analyze_dependency_compatibility(self) -> Dict[str, Any]:
        """
        Analyze package dependencies for compatibility and conflicts

        Returns:
            Dictionary of dependency compatibility insights
        """
        compatibility_analysis = {
            "total_packages": 0,
            "potential_conflicts": [],
            "version_distribution": {},
        }

        try:
            # Generate package dependency graph
            package_graph = self.generate_package_dependency_graph()

            compatibility_analysis["total_packages"] = len(package_graph.nodes())

            # Analyze version distribution
            for node, data in package_graph.nodes(data=True):
                version = data.get("version", "unknown")
                compatibility_analysis["version_distribution"][node] = version

            # Detect potential conflicts
            for pkg, version in compatibility_analysis["version_distribution"].items():
                # Example conflict detection (can be expanded)
                conflicting_versions = [
                    other_pkg
                    for other_pkg, other_ver in compatibility_analysis[
                        "version_distribution"
                    ].items()
                    if other_pkg != pkg and other_ver != version
                ]

                if conflicting_versions:
                    compatibility_analysis["potential_conflicts"].append(
                        {
                            "package": pkg,
                            "current_version": version,
                            "conflicting_packages": conflicting_versions,
                        }
                    )

            return compatibility_analysis

        except Exception as e:
            self.logger.error(f"Dependency compatibility analysis failed: {e}")
            return compatibility_analysis

    def generate_comprehensive_dependency_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive dependency analysis report

        Returns:
            Detailed dependency analysis report
        """
        try:
            # Generate module dependency graph
            module_graph = self.generate_module_dependency_graph()

            # Detect circular dependencies
            circular_dependencies = self.detect_circular_dependencies(module_graph)

            # Analyze package compatibility
            compatibility_analysis = self.analyze_dependency_compatibility()

            # Compile comprehensive report
            dependency_report = {
                "timestamp": datetime.now().isoformat(),
                "module_dependency_graph": {
                    "nodes": list(module_graph.nodes()),
                    "edges": list(module_graph.edges()),
                },
                "circular_dependencies": circular_dependencies,
                "compatibility_analysis": compatibility_analysis,
            }

            # Persist report
            report_path = os.path.join(
                self.log_dir,
                f'dependency_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            )

            with open(report_path, "w") as f:
                json.dump(dependency_report, f, indent=2)

            self.logger.info(
                f"Comprehensive dependency report generated: {report_path}"
            )

            return dependency_report

        except Exception as e:
            self.logger.error(f"Comprehensive dependency report generation failed: {e}")
            return {}


def main():
    """
    Main execution for advanced dependency tracking
    """
    try:
        dependency_tracker = AdvancedDependencyTracker()
        dependency_report = (
            dependency_tracker.generate_comprehensive_dependency_report()
        )

        # Print key insights
        print("Dependency Tracking Insights:")
        print(
            f"Total Modules: {len(dependency_report.get('module_dependency_graph', {}).get('nodes', []))}"
        )
        print(
            f"Circular Dependencies: {len(dependency_report.get('circular_dependencies', []))}"
        )
        print("\nPackage Compatibility:")
        compatibility = dependency_report.get("compatibility_analysis", {})
        print(f"Total Packages: {compatibility.get('total_packages', 0)}")
        print("\nPotential Conflicts:")
        for conflict in compatibility.get("potential_conflicts", []):
            print(
                f"- {conflict['package']}: Conflicting with {conflict['conflicting_packages']}"
            )

    except Exception as e:
        print(f"Dependency tracking failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
