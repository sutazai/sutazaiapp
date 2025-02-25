#!/usr/bin/env python3
"""
SutazAI Advanced Dependency Mapping and Cross-Referencing Framework

Comprehensive tool for:
- Detailed dependency tracking
- Inter-component relationship mapping
- Architectural dependency visualization
- Dependency health assessment

Key Responsibilities:
- Generate comprehensive dependency graphs
- Identify complex dependency relationships
- Assess dependency health and potential risks
- Provide actionable dependency optimization insights
"""

import ast
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

# Internal system imports
from core_system.monitoring.advanced_logger import AdvancedLogger


@dataclass
class DependencyAnalysisReport:
    """
    Comprehensive dependency analysis report

    Captures intricate details about system dependencies,
    their relationships, and potential optimization strategies
    """

    timestamp: str
    total_modules: int
    dependency_graph: Dict[str, List[str]]
    circular_dependencies: List[Tuple[str, str]]
    high_coupling_modules: List[Dict[str, Any]]
    dependency_health_score: float
    optimization_recommendations: List[str]


class AdvancedDependencyMapper:
    """
    Advanced dependency mapping and analysis framework

    Provides deep insights into system component dependencies
    and architectural relationships
    """

    def __init__(
        self,
        base_dir: str = "/opt/sutazai_project/SutazAI",
        logger: Optional[AdvancedLogger] = None,
    ):
        """
        Initialize advanced dependency mapping framework

        Args:
            base_dir (str): Base directory of the project
            logger (AdvancedLogger): Advanced logging system
        """
        self.base_dir = base_dir
        self.logger = logger or AdvancedLogger()
        self.dependency_graph = nx.DiGraph()

    def _get_stdlib_modules(self) -> frozenset[str]:
        """Get Python standard library modules"""
        if hasattr(sys, "stdlib_module_names"):
            return sys.stdlib_module_names
        # Convert to frozenset and use union operator
        return frozenset(sys.builtin_module_names).union(
            {"os", "sys", "math", "datetime", "json", "re", "ast", "types"}
        )

    def _is_valid_python_module(self, file_path: str) -> bool:
        """
        Validate if a file is a valid Python module

        Args:
            file_path (str): Path to the file

        Returns:
            Boolean indicating module validity
        """
        return (
            file_path.endswith(".py")
            and not file_path.startswith("__")
            and not file_path.startswith(".")
        )

    def _extract_module_dependencies(self, file_path: str) -> Set[str]:
        """
        Extract module-level dependencies for a given Python file

        Args:
            file_path (str): Path to the Python file

        Returns:
            Set of imported module names
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    module = node.module.split(".")[0] if node.module else ""
                    if module and module not in {"__future__", "typing"}:
                        imports.add(module)

            return {i for i in imports if i not in self._get_stdlib_modules()}

        except Exception as e:
            self.logger.log(
                f"Could not extract dependencies from {file_path}: {e}",
                level="warning",
            )
            return set()

    def build_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Build a comprehensive dependency graph for the entire project

        Returns:
            Mapping of modules and their dependencies
        """
        dependency_graph = {}

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                file_path = os.path.join(root, file)

                if self._is_valid_python_module(file_path):
                    try:
                        # Relative path from base directory
                        rel_path = os.path.relpath(file_path, self.base_dir)

                        # Extract dependencies
                        dependencies = self._extract_module_dependencies(
                            file_path
                        )

                        # Add to dependency graph
                        dependency_graph[rel_path] = list(dependencies)

                        # Build NetworkX graph for advanced analysis
                        self.dependency_graph.add_node(rel_path)
                        for dep in dependencies:
                            self.dependency_graph.add_edge(rel_path, dep)

                    except Exception as e:
                        self.logger.log(
                            f"Could not process module {file_path}: {e}",
                            level="warning",
                        )

        return dependency_graph

    def detect_circular_dependencies(self) -> List[Tuple[str, str]]:
        """
        Detect circular dependencies in the project

        Returns:
            List of circular dependency pairs
        """
        try:
            # Use NetworkX to find cycles
            cycles = list(nx.simple_cycles(self.dependency_graph))

            return [(cycle[0], cycle[1]) for cycle in cycles]

        except Exception as e:
            self.logger.log(
                f"Circular dependency detection failed: {e}", level="error"
            )
            return []

    def identify_high_coupling_modules(self) -> List[Dict[str, Any]]:
        """
        Identify modules with high coupling

        Returns:
            List of high-coupling module details
        """
        high_coupling_modules = []

        for node in self.dependency_graph.nodes():
            in_degree = self.dependency_graph.in_degree(node)
            out_degree = self.dependency_graph.out_degree(node)

            # Define high coupling threshold
            if in_degree + out_degree > 10:
                high_coupling_modules.append(
                    {
                        "module": node,
                        "in_degree": in_degree,
                        "out_degree": out_degree,
                        "total_coupling": in_degree + out_degree,
                    }
                )

        return sorted(
            high_coupling_modules,
            key=lambda x: x["total_coupling"],
            reverse=True,
        )

    def calculate_dependency_health_score(self) -> float:
        """
        Calculate an overall dependency health score

        Returns:
            Dependency health score (0-100)
        """
        try:
            total_modules = len(self.dependency_graph.nodes())
            circular_dependencies = len(self.detect_circular_dependencies())
            high_coupling_modules = len(self.identify_high_coupling_modules())

            # Complex scoring mechanism
            base_score = 100.0
            circular_penalty = min(circular_dependencies * 10, 50)
            coupling_penalty = min(high_coupling_modules * 5, 30)

            health_score = max(
                base_score - circular_penalty - coupling_penalty, 0
            )

            return round(health_score, 2)

        except Exception as e:
            self.logger.log(
                f"Dependency health score calculation failed: {e}",
                level="error",
            )
            return 0.0

    def generate_optimization_recommendations(
        self,
        dependency_graph: Dict[str, List[str]],
        circular_dependencies: List[Tuple[str, str]],
        high_coupling_modules: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Generate intelligent dependency optimization recommendations

        Args:
            dependency_graph (Dict): Project dependency mapping
            circular_dependencies (List): Detected circular dependencies
            high_coupling_modules (List): Modules with high coupling

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Circular dependency recommendations
        for dep_pair in circular_dependencies:
            recommendations.append(
                f"Break circular dependency between "
                f"{dep_pair[0]} and {dep_pair[1]}"
            )

        # High coupling module recommendations
        for module in high_coupling_modules[:3]:  # Top 3 high-coupling modules
            recommendations.append(
                "Refactor module "
                f"{module['module']} to reduce coupling "
                f"(In: {module['in_degree']}, Out: {module['out_degree']})"
            )

        # General dependency optimization
        if len(dependency_graph) > 100:
            recommendations.append(
                "Consider modularizing the project "
                "to reduce overall complexity"
            )

        return recommendations

    def generate_comprehensive_dependency_report(
        self,
    ) -> DependencyAnalysisReport:
        """
        Generate a comprehensive dependency analysis report

        Returns:
            Detailed dependency analysis report
        """
        with self.logger.trace("generate_comprehensive_dependency_report"):
            # Build dependency graph
            dependency_graph = self.build_dependency_graph()

            # Detect circular dependencies
            circular_dependencies = self.detect_circular_dependencies()

            # Identify high coupling modules
            high_coupling_modules = self.identify_high_coupling_modules()

            # Calculate dependency health score
            dependency_health_score = self.calculate_dependency_health_score()

            # Generate optimization recommendations
            optimization_recommendations = (
                self.generate_optimization_recommendations(
                    dependency_graph,
                    circular_dependencies,
                    high_coupling_modules,
                )
            )

            # Create comprehensive dependency report
            dependency_report = DependencyAnalysisReport(
                timestamp=datetime.now().isoformat(),
                total_modules=len(dependency_graph),
                dependency_graph=dependency_graph,
                circular_dependencies=circular_dependencies,
                high_coupling_modules=high_coupling_modules,
                dependency_health_score=dependency_health_score,
                optimization_recommendations=optimization_recommendations,
            )

            # Persist dependency report
            report_path = (
                "/opt/sutazai_project/SutazAI/logs/"
                f"dependency_analysis_report_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            with open(report_path, "w") as f:
                json.dump(asdict(dependency_report), f, indent=2)

            self.logger.log(
                f"Dependency analysis report generated: {report_path}",
                level="info",
            )

            return dependency_report


def main():
    """
    Main execution point for dependency mapping
    """
    try:
        dependency_mapper = AdvancedDependencyMapper()

        # Generate comprehensive dependency report
        report = dependency_mapper.generate_comprehensive_dependency_report()

        # Print optimization recommendations
        print("Dependency Optimization Recommendations:")
        for recommendation in report.optimization_recommendations:
            print(f"- {recommendation}")

        # Print dependency health score
        print(
            f"\nDependency Health Score: "
            f"{report.dependency_health_score}/100"
        )

    except Exception as e:
        print(f"Dependency mapping failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
