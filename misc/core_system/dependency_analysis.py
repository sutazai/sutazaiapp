#!/usr/bin/env python3
"""
SutazAI Advanced Dependency Analysis and Optimization System

Provides comprehensive dependency intelligence,
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import networkx as nx
import pkg_resources


class AdvancedDependencyAnalyzer:
    """
    Comprehensive dependency intelligence system

    Key Capabilities:
    - Dependency graph generation
    - Performance impact analysis
    - Optimization recommendations
    """

    def __init__(
        self,
        base_dir: str = "/opt/SutazAI",
        log_dir: Optional[str] = None,
    ):
        """
        Initialize Advanced Dependency Analyzer

        Args:
            base_dir (str): Base project directory
            log_dir (Optional[str]): Custom log directory
        """
        self.base_dir = base_dir
        self.log_dir = log_dir or os.path.join(
            base_dir, "logs", "dependency_analysis"
        )

        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            filename=os.path.join(self.log_dir, "dependency_analysis.log"),
        )
        self.logger = logging.getLogger("SutazAI.DependencyAnalyzer")

    def generate_dependency_graph(self) -> nx.DiGraph:
        """
        Generate a comprehensive dependency graph

        Returns:
            NetworkX Directed Graph of package dependencies
        """
        dependency_graph = nx.DiGraph()

        try:
            # Get installed packages
            for pkg in pkg_resources.working_set:
                dependency_graph.add_node(
                    pkg.key, version=pkg.version, location=pkg.location
                )

                # Add edges for dependencies
                for req in pkg.requires():
                    dependency_graph.add_edge(pkg.key, req.key)

            return dependency_graph

        except Exception as e:
            self.logger.error(f"Dependency graph generation failed: {e}")
            return nx.DiGraph()

    def analyze_dependency_performance(
        self, dependency_graph: nx.DiGraph
    ) -> Dict[str, Any]:
        """
        Perform performance analysis of dependencies

        Args:
            dependency_graph (nx.DiGraph): Dependency graph to analyze

        Returns:
            Dictionary of performance insights
        """
        performance_analysis = {
            "total_packages": len(dependency_graph.nodes()),
            "dependency_depth": {},
            "centrality_metrics": {},
        }

        try:
            # Calculate dependency depth
            for node in dependency_graph.nodes():
                performance_analysis["dependency_depth"][node] = (
                    nx.shortest_path_length(dependency_graph, source=node)
                )

            # Calculate centrality metrics
            performance_analysis["centrality_metrics"] = {
                "degree_centrality": nx.degree_centrality(dependency_graph),
                "betweenness_centrality": nx.betweenness_centrality(
                    dependency_graph
                ),
                "eigenvector_centrality": nx.eigenvector_centrality(
                    dependency_graph
                ),
            }

            return performance_analysis

        except Exception as e:
            self.logger.error(f"Dependency performance analysis failed: {e}")
            return performance_analysis

    def identify_optimization_opportunities(
        self,
        dependency_graph: nx.DiGraph,
        performance_analysis: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Identify potential dependency optimization opportunities

        Args:
            dependency_graph (nx.DiGraph): Dependency graph
            performance_analysis (Dict): Performance analysis results

        Returns:
            List of optimization recommendations
        """
        optimization_recommendations = []

        try:
            # Identify highly central packages
            high_centrality_threshold = 0.5
            high_centrality_packages = [
                pkg
                for pkg, centrality in performance_analysis[
                    "centrality_metrics"
                ]["degree_centrality"].items()
                if centrality > high_centrality_threshold
            ]

            for pkg in high_centrality_packages:
                optimization_recommendations.append(
                    {
                        "package": pkg,
                        "recommendation": "Consider performance optimization or alternative package",
                        "reason": "High dependency centrality",
                    }
                )

            # Identify deep dependency chains
            deep_dependency_threshold = 5
            deep_dependency_packages = [
                pkg
                for pkg, depth in performance_analysis[
                    "dependency_depth"
                ].items()
                if depth > deep_dependency_threshold
            ]

            for pkg in deep_dependency_packages:
                optimization_recommendations.append(
                    {
                        "package": pkg,
                        "recommendation": "Evaluate dependency chain complexity",
                        "reason": "Deep dependency hierarchy",
                    }
                )

            return optimization_recommendations

        except Exception as e:
            self.logger.error(
                f"Optimization opportunity identification failed: {e}"
            )
            return optimization_recommendations

    def generate_comprehensive_dependency_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive dependency analysis report

        Returns:
            Detailed dependency analysis report
        """
        try:
            # Generate dependency graph
            dependency_graph = self.generate_dependency_graph()

            # Perform performance analysis
            performance_analysis = self.analyze_dependency_performance(
                dependency_graph
            )

            # Identify optimization opportunities
            optimization_recommendations = (
                self.identify_optimization_opportunities(
                    dependency_graph, performance_analysis
                )
            )

            # Compile comprehensive report
            dependency_report = {
                "timestamp": datetime.now().isoformat(),
                "dependency_graph": {
                    "nodes": list(dependency_graph.nodes()),
                    "edges": list(dependency_graph.edges()),
                },
                "performance_analysis": performance_analysis,
                "optimization_recommendations": optimization_recommendations,
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
            self.logger.error(
                f"Comprehensive dependency report generation failed: {e}"
            )
            return {}


def main():
    """
    Main execution for advanced dependency analysis
    """
    try:
        dependency_analyzer = AdvancedDependencyAnalyzer()
        dependency_report = (
            dependency_analyzer.generate_comprehensive_dependency_report()
        )

        # Print key insights
        print("Dependency Analysis Insights:")
        print(
            f"Total Packages: {dependency_report.get('performance_analysis', {}).get('total_packages', 0)}"
        )
        print("\nOptimization Recommendations:")
        for rec in dependency_report.get("optimization_recommendations", []):
            print(
                f"- {rec['package']}: {rec['recommendation']} ({rec['reason']})"
            )

    except Exception as e:
        print(f"Dependency analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
