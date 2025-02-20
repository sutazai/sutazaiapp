#!/usr/bin/env python3
"""
Ultra-Comprehensive Dependency Cross-Referencing System

An advanced framework designed to:
- Perform deep, multi-dimensional dependency analysis
- Identify complex system interactions
- Generate comprehensive architectural insights
- Ensure system integrity and optimize component relationships
"""

import ast
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class DependencyRelationship:
    """
    Comprehensive representation of a dependency relationship
    """

    source_module: str
    target_module: str
    relationship_type: str
    interaction_strength: float
    interaction_details: Dict[str, Any]


class UltraComprehensiveDependencyCrossReferencer:
    """
    Advanced dependency cross-referencing and analysis system
    """

    def __init__(
        self,
        base_dir: str = "/opt/sutazai_project/SutazAI",
        log_dir: Optional[str] = None,
    ):
        """
        Initialize Ultra-Comprehensive Dependency Cross-Referencing System

        Args:
            base_dir (str): Base project directory
            log_dir (Optional[str]): Custom log directory
        """
        # Core configuration
        self.base_dir = base_dir
        self.log_dir = log_dir or os.path.join(
            base_dir, "logs", "dependency_cross_referencing"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(
                        self.log_dir, "dependency_cross_referencing.log"
                    )
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("SutazAI.DependencyCrossReferencer")

        # Dependency tracking
        self.dependency_graph = nx.DiGraph()
        self.module_relationships: List[DependencyRelationship] = []

    def analyze_project_dependencies(self) -> Dict[str, Any]:
        """
        Perform comprehensive project-wide dependency analysis

        Returns:
            Detailed dependency analysis report
        """
        dependency_report = {
            "timestamp": datetime.now().isoformat(),
            "total_modules": 0,
            "total_dependencies": 0,
            "circular_dependencies": [],
            "module_categories": {},
            "dependency_metrics": {},
        }

        # Reset dependency graph and relationships
        self.dependency_graph.clear()
        self.module_relationships.clear()

        # Traverse project directories
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.base_dir)

                    try:
                        # Analyze module dependencies
                        module_dependencies = (
                            self._analyze_module_dependencies(file_path)
                        )

                        # Update dependency graph
                        self._update_dependency_graph(
                            relative_path, module_dependencies
                        )

                        # Categorize module
                        module_category = self._categorize_module(
                            relative_path
                        )
                        dependency_report["module_categories"][
                            module_category
                        ] = (
                            dependency_report["module_categories"].get(
                                module_category, 0
                            )
                            + 1
                        )

                        dependency_report["total_modules"] += 1

                    except Exception as e:
                        self.logger.warning(
                            f"Dependency analysis failed for {file_path}: {e}"
                        )

        # Detect circular dependencies
        dependency_report["circular_dependencies"] = list(
            nx.simple_cycles(self.dependency_graph)
        )
        dependency_report["total_dependencies"] = len(
            self.dependency_graph.edges()
        )

        # Calculate dependency metrics
        dependency_report["dependency_metrics"] = (
            self._calculate_dependency_metrics()
        )

        # Visualize dependency graph
        self._visualize_dependency_graph()

        # Persist dependency report
        self._persist_dependency_report(dependency_report)

        return dependency_report

    def _analyze_module_dependencies(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze dependencies for a specific module

        Args:
            file_path (str): Path to the Python module

        Returns:
            Detailed module dependency information
        """
        dependencies = {
            "imports": [],
            "function_calls": [],
            "class_references": [],
        }

        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content)

            # Analyze imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    dependencies["imports"].extend(
                        [alias.name for alias in node.names]
                    )

                elif isinstance(node, ast.ImportFrom):
                    base_module = node.module or ""
                    dependencies["imports"].extend(
                        [
                            (
                                f"{base_module}.{alias.name}"
                                if base_module
                                else alias.name
                            )
                            for alias in node.names
                        ]
                    )

                # Analyze function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        dependencies["function_calls"].append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        dependencies["function_calls"].append(
                            f"{node.func.value.id}.{node.func.attr}"
                            if isinstance(node.func.value, ast.Name)
                            else node.func.attr
                        )

                # Analyze class references
                elif isinstance(node, ast.ClassDef):
                    dependencies["class_references"].append(node.name)

        except Exception as e:
            self.logger.warning(
                f"Module dependency analysis failed for {file_path}: {e}"
            )

        return dependencies

    def _update_dependency_graph(
        self, module_path: str, dependencies: Dict[str, Any]
    ):
        """
        Update dependency graph with module relationships

        Args:
            module_path (str): Relative path of the module
            dependencies (Dict): Module dependencies
        """
        # Add module as a node
        self.dependency_graph.add_node(module_path)

        # Add import dependencies
        for imp in dependencies.get("imports", []):
            if imp:
                relationship = DependencyRelationship(
                    source_module=module_path,
                    target_module=imp,
                    relationship_type="import",
                    interaction_strength=1.0,
                    interaction_details={"type": "direct_import"},
                )
                self.module_relationships.append(relationship)
                self.dependency_graph.add_edge(module_path, imp)

        # Add function call dependencies
        for func_call in dependencies.get("function_calls", []):
            if func_call:
                relationship = DependencyRelationship(
                    source_module=module_path,
                    target_module=func_call,
                    relationship_type="function_call",
                    interaction_strength=0.5,
                    interaction_details={"type": "runtime_dependency"},
                )
                self.module_relationships.append(relationship)
                self.dependency_graph.add_edge(module_path, func_call)

    def _categorize_module(self, module_path: str) -> str:
        """
        Categorize module based on its path and characteristics

        Args:
            module_path (str): Relative path of the module

        Returns:
            Module category
        """
        categories = [
            ("core_system", "core"),
            ("workers", "worker"),
            ("ai_agents", "ai_agent"),
            ("services", "service"),
            ("scripts", "script"),
            ("tests", "test"),
        ]

        for category_path, category_name in categories:
            if category_path in module_path:
                return category_name

        return "uncategorized"

    def _calculate_dependency_metrics(self) -> Dict[str, Any]:
        """
        Calculate advanced dependency metrics

        Returns:
            Comprehensive dependency metrics
        """
        metrics = {
            "fan_in": {},
            "fan_out": {},
            "coupling_coefficient": {},
            "centrality": {
                "degree": nx.degree_centrality(self.dependency_graph),
                "betweenness": nx.betweenness_centrality(
                    self.dependency_graph
                ),
                "closeness": nx.closeness_centrality(self.dependency_graph),
            },
        }

        # Calculate fan-in and fan-out
        for node in self.dependency_graph.nodes():
            metrics["fan_in"][node] = self.dependency_graph.in_degree(node)
            metrics["fan_out"][node] = self.dependency_graph.out_degree(node)

            # Calculate coupling coefficient
            try:
                metrics["coupling_coefficient"][node] = len(
                    list(self.dependency_graph.predecessors(node))
                ) / (len(list(self.dependency_graph.nodes())) - 1)
            except ZeroDivisionError:
                metrics["coupling_coefficient"][node] = 0

        return metrics

    def _visualize_dependency_graph(self):
        """
        Create visual representation of the dependency graph
        """
        try:
            plt.figure(figsize=(20, 20))
            pos = nx.spring_layout(self.dependency_graph, k=0.5, iterations=50)

            nx.draw(
                self.dependency_graph,
                pos,
                with_labels=True,
                node_color="lightblue",
                node_size=300,
                font_size=8,
                font_weight="bold",
                arrows=True,
            )

            # Save visualization
            plt.title("SutazAI Dependency Graph")
            plt.tight_layout()
            visualization_path = os.path.join(
                self.log_dir,
                f'dependency_graph_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
            )
            plt.savefig(visualization_path, dpi=300)
            plt.close()

            self.logger.info(
                f"Dependency graph visualization saved: {visualization_path}"
            )

        except Exception as e:
            self.logger.error(f"Dependency graph visualization failed: {e}")

    def _persist_dependency_report(self, dependency_report: Dict[str, Any]):
        """
        Persist comprehensive dependency report

        Args:
            dependency_report (Dict): Detailed dependency report
        """
        try:
            report_path = os.path.join(
                self.log_dir,
                f'dependency_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            )

            with open(report_path, "w") as f:
                json.dump(dependency_report, f, indent=2)

            self.logger.info(f"Dependency report persisted: {report_path}")

        except Exception as e:
            self.logger.error(f"Dependency report persistence failed: {e}")

    def generate_dependency_insights(self) -> Dict[str, Any]:
        """
        Generate advanced dependency insights and recommendations

        Returns:
            Comprehensive dependency insights
        """
        insights = {
            "high_coupling_modules": [],
            "potential_refactoring_candidates": [],
            "architectural_recommendations": [],
        }

        # Identify high coupling modules
        for module, coupling in self.module_relationships.get(
            "coupling_coefficient", {}
        ).items():
            if coupling > 0.7:  # High coupling threshold
                insights["high_coupling_modules"].append(
                    {"module": module, "coupling_coefficient": coupling}
                )

        # Identify potential refactoring candidates
        for module, fan_in in self.module_relationships.get(
            "fan_in", {}
        ).items():
            if fan_in > 10:  # High fan-in threshold
                insights["potential_refactoring_candidates"].append(
                    {"module": module, "fan_in": fan_in}
                )

        # Generate architectural recommendations
        if insights["high_coupling_modules"]:
            insights["architectural_recommendations"].append(
                "Consider applying dependency inversion principle to reduce coupling"
            )

        if insights["potential_refactoring_candidates"]:
            insights["architectural_recommendations"].append(
                "Evaluate modularization strategies for highly connected modules"
            )

        return insights


def main():
    """
    Demonstrate Dependency Cross-Referencing System
    """
    dependency_cross_referencer = UltraComprehensiveDependencyCrossReferencer()

    # Perform comprehensive dependency analysis
    dependency_report = (
        dependency_cross_referencer.analyze_project_dependencies()
    )

    print("\n🌐 Dependency Analysis Results 🌐")
    print(f"Total Modules: {dependency_report.get('total_modules', 0)}")
    print(
        f"Total Dependencies: {dependency_report.get('total_dependencies', 0)}"
    )

    print("\nCircular Dependencies:")
    for cycle in dependency_report.get("circular_dependencies", []):
        print(" -> ".join(cycle))

    # Generate dependency insights
    dependency_insights = (
        dependency_cross_referencer.generate_dependency_insights()
    )

    print("\nDependency Insights:")
    for category, insights in dependency_insights.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for insight in insights:
            print(f"- {insight}")


if __name__ == "__main__":
    main()
