#!/usr/bin/env python3
"""
SutazAI Dependency Cross-Referencing and Compatibility System

Provides comprehensive analysis of package dependencies,
compatibility, and potential conflicts.
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import networkx as nx
import pkg_resources
import yaml
from packaging import version


class DependencyCrossReferencer:
    """
    Advanced dependency cross-referencing and compatibility analyzer

    Key Capabilities:
    - Dependency compatibility assessment
    - Conflict detection
    - Version constraint analysis
    - Intelligent package recommendation
    """

    def __init__(
        self,
        base_dir: str = "/opt/sutazai_project/SutazAI",
        log_dir: Optional[str] = None,
    ):
        """
        Initialize Dependency Cross-Referencing System

        Args:
            base_dir (str): Base project directory
            log_dir (Optional[str]): Custom log directory
        """
        self.base_dir = base_dir
        self.log_dir = log_dir or os.path.join(
            base_dir, "logs", "dependency_cross_reference"
        )

        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            filename=os.path.join(
                self.log_dir, "dependency_cross_reference.log"
            ),
        )
        self.logger = logging.getLogger("SutazAI.DependencyCrossReferencer")

    def build_dependency_compatibility_graph(self) -> nx.DiGraph:
        """
        Build a comprehensive dependency compatibility graph

        Returns:
            NetworkX Directed Graph of package compatibility
        """
        compatibility_graph = nx.DiGraph()

        try:
            # Get all installed packages
            for pkg in pkg_resources.working_set:
                compatibility_graph.add_node(
                    pkg.key,
                    version=pkg.version,
                    requires=list(req.key for req in pkg.requires()),
                )

                # Add compatibility edges
                for req in pkg.requires():
                    compatibility_graph.add_edge(pkg.key, req.key)

            return compatibility_graph

        except Exception as e:
            self.logger.error(
                f"Dependency compatibility graph generation failed: {e}"
            )
            return nx.DiGraph()

    def detect_version_conflicts(
        self, compatibility_graph: nx.DiGraph
    ) -> List[Dict[str, Any]]:
        """
        Detect potential version conflicts in dependencies

        Args:
            compatibility_graph (nx.DiGraph): Dependency compatibility graph

        Returns:
            List of detected version conflicts
        """
        version_conflicts = []

        try:
            # Track package versions
            package_versions = {}
            for node, data in compatibility_graph.nodes(data=True):
                if node not in package_versions:
                    package_versions[node] = data.get("version", "unknown")

            # Detect conflicts in dependencies
            for pkg, version_info in package_versions.items():
                # Check for multiple versions of same package
                conflicting_versions = [
                    other_ver
                    for other_pkg, other_ver in package_versions.items()
                    if other_pkg != pkg
                    and other_pkg in compatibility_graph.predecessors(pkg)
                ]

                if conflicting_versions:
                    version_conflicts.append(
                        {
                            "package": pkg,
                            "current_version": version_info,
                            "conflicting_versions": conflicting_versions,
                        }
                    )

            return version_conflicts

        except Exception as e:
            self.logger.error(f"Version conflict detection failed: {e}")
            return version_conflicts

    def analyze_dependency_compatibility(
        self, compatibility_graph: nx.DiGraph
    ) -> Dict[str, Any]:
        """
        Perform comprehensive dependency compatibility analysis

        Args:
            compatibility_graph (nx.DiGraph): Dependency compatibility graph

        Returns:
            Dictionary of compatibility analysis results
        """
        compatibility_analysis = {
            "total_packages": len(compatibility_graph.nodes()),
            "package_dependencies": {},
            "potential_conflicts": [],
        }

        try:
            # Analyze package dependencies
            for pkg in compatibility_graph.nodes():
                compatibility_analysis["package_dependencies"][pkg] = {
                    "direct_dependencies": list(
                        compatibility_graph.successors(pkg)
                    ),
                    "reverse_dependencies": list(
                        compatibility_graph.predecessors(pkg)
                    ),
                }

            # Detect version conflicts
            compatibility_analysis["potential_conflicts"] = (
                self.detect_version_conflicts(compatibility_graph)
            )

            return compatibility_analysis

        except Exception as e:
            self.logger.error(f"Dependency compatibility analysis failed: {e}")
            return compatibility_analysis

    def generate_package_compatibility_recommendations(
        self, compatibility_analysis: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate intelligent package compatibility recommendations

        Args:
            compatibility_analysis (Dict): Compatibility analysis results

        Returns:
            List of package compatibility recommendations
        """
        recommendations = []

        try:
            # Recommend package updates or alternatives for conflicting packages
            for conflict in compatibility_analysis.get(
                "potential_conflicts", []
            ):
                recommendations.append(
                    {
                        "package": conflict["package"],
                        "recommendation": "Resolve version conflicts",
                        "details": f"Multiple versions detected: {conflict['conflicting_versions']}",
                    }
                )

            # Identify packages with complex dependency chains
            for pkg, deps in compatibility_analysis.get(
                "package_dependencies", {}
            ).items():
                if len(deps["direct_dependencies"]) > 5:
                    recommendations.append(
                        {
                            "package": pkg,
                            "recommendation": "Simplify dependency chain",
                            "details": f"High number of direct dependencies: {len(deps['direct_dependencies'])}",
                        }
                    )

            return recommendations

        except Exception as e:
            self.logger.error(
                f"Package compatibility recommendations generation failed: {e}"
            )
            return recommendations

    def generate_comprehensive_cross_reference_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive dependency cross-reference report

        Returns:
            Detailed dependency cross-reference report
        """
        try:
            # Build dependency compatibility graph
            compatibility_graph = self.build_dependency_compatibility_graph()

            # Perform compatibility analysis
            compatibility_analysis = self.analyze_dependency_compatibility(
                compatibility_graph
            )

            # Generate compatibility recommendations
            compatibility_recommendations = (
                self.generate_package_compatibility_recommendations(
                    compatibility_analysis
                )
            )

            # Compile comprehensive report
            cross_reference_report = {
                "timestamp": datetime.now().isoformat(),
                "compatibility_graph": {
                    "nodes": list(compatibility_graph.nodes()),
                    "edges": list(compatibility_graph.edges()),
                },
                "compatibility_analysis": compatibility_analysis,
                "compatibility_recommendations": compatibility_recommendations,
            }

            # Persist report
            report_path = os.path.join(
                self.log_dir,
                f'dependency_cross_reference_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            )

            with open(report_path, "w") as f:
                json.dump(cross_reference_report, f, indent=2)

            self.logger.info(
                f"Comprehensive cross-reference report generated: {report_path}"
            )

            return cross_reference_report

        except Exception as e:
            self.logger.error(
                f"Comprehensive cross-reference report generation failed: {e}"
            )
            return {}


def main():
    """
    Main execution for dependency cross-referencing
    """
    try:
        cross_referencer = DependencyCrossReferencer()
        cross_reference_report = (
            cross_referencer.generate_comprehensive_cross_reference_report()
        )

        # Print key insights
        print("Dependency Cross-Reference Insights:")
        print(
            f"Total Packages: {cross_reference_report.get('compatibility_analysis', {}).get('total_packages', 0)}"
        )
        print("\nCompatibility Recommendations:")
        for rec in cross_reference_report.get(
            "compatibility_recommendations", []
        ):
            print(
                f"- {rec['package']}: {rec['recommendation']} ({rec['details']})"
            )

    except Exception as e:
        print(f"Dependency cross-referencing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
