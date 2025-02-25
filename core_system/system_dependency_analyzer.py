#!/usr/bin/env python3
"""
Ultra-Comprehensive System Dependency Analysis and Cross-Referencing Framework

Provides advanced capabilities for:
- Systematic dependency tracking
- Comprehensive module interaction analysis
- Intelligent dependency graph generation
- Autonomous architectural insights generation
"""

import ast
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

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


class SystemDependencyAnalyzer:
    """
    Advanced system dependency analysis and cross-referencing system
    """

    def __init__(
        self,
        base_dir: str = "/opt/sutazai_project/SutazAI",
        log_dir: Optional[str] = None,
    ):
        """
        Initialize System Dependency Analyzer

        Args:
            base_dir (str): Base project directory
            log_dir (Optional[str]): Custom log directory
        """
        # Core configuration
        self.base_dir = base_dir
        self.log_dir = log_dir or os.path.join(base_dir, "logs", "system_dependency")
        os.makedirs(self.log_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(self.log_dir, "system_dependency_analyzer.log")
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("SutazAI.SystemDependencyAnalyzer")

        # Dependency tracking
        self.dependency_graph = nx.DiGraph()
        self.module_relationships: List[DependencyRelationship] = []

    def analyze_system_dependencies(
        self, search_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive system-wide dependency analysis

        Args:
            search_paths (Optional[List[str]]): Specific paths to search

        Returns:
            Detailed dependency analysis report
        """
        # Reset dependency graph and relationships
        self.dependency_graph.clear()
        self.module_relationships.clear()

        # Default search paths
        search_paths = search_paths or [
            "core_system",
            "workers",
            "ai_agents",
            "services",
            "scripts",
        ]

        dependency_report = {
            "timestamp": datetime.now().isoformat(),
            "total_modules": 0,
            "total_dependencies": 0,
            "circular_dependencies": [],
            "module_categories": {},
            "dependency_metrics": {},
        }

        # Traverse project directories
        for path in search_paths:
            full_path = os.path.join(self.base_dir, path)

            for root, _, files in os.walk(full_path):
                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, self.base_dir)

                        try:
                            # Analyze module dependencies
                            module_dependencies = self._analyze_module_dependencies(
                                file_path
                            )

                            # Update dependency graph
                            self._update_dependency_graph(
                                relative_path, module_dependencies
                            )

                            # Categorize module
                            module_category = self._categorize_module(relative_path)
                            dependency_report["module_categories"][module_category] = (
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
        dependency_report["total_dependencies"] = len(self.dependency_graph.edges())

        # Calculate dependency metrics
        dependency_report["dependency_metrics"] = self._calculate_dependency_metrics()

        # Detect architectural patterns
        dependency_report["architectural_patterns"] = (
            self._detect_architectural_patterns()
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
                    dependencies["imports"].extend([alias.name for alias in node.names])

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

    def _update_dependency_graph(self, module_path: str, dependencies: Dict[str, Any]):
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
                "betweenness": nx.betweenness_centrality(self.dependency_graph),
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

    def _detect_architectural_patterns(self) -> Dict[str, Any]:
        """
        Detect architectural patterns and characteristics

        Returns:
            Architectural pattern insights
        """
        patterns = {
            "design_patterns": {
                "singleton": 0,
                "factory": 0,
                "strategy": 0,
                "decorator": 0,
            },
            "architectural_styles": {
                "layered": 0,
                "microservices": 0,
                "event_driven": 0,
            },
        }

        # Detect design patterns and architectural styles
        for relationship in self.module_relationships:
            # Simple pattern detection (can be expanded)
            if "getInstance" in relationship.source_module:
                patterns["design_patterns"]["singleton"] += 1

            if "create_" in relationship.source_module:
                patterns["design_patterns"]["factory"] += 1

            if "execute" in relationship.source_module:
                patterns["design_patterns"]["strategy"] += 1

            if "@" in relationship.source_module:
                patterns["design_patterns"]["decorator"] += 1

        return patterns

    def _visualize_dependency_graph(self):
        """
        Create visual representation of the dependency graph
        """
        try:
            import matplotlib.pyplot as plt

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
            plt.title("SutazAI System Dependency Graph")
            plt.tight_layout()
            visualization_path = os.path.join(
                self.log_dir,
                f'system_dependency_graph_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
            )
            plt.savefig(visualization_path, dpi=300)
            plt.close()

            self.logger.info(
                f"System dependency graph visualization saved: {visualization_path}"
            )

        except Exception as e:
            self.logger.error(f"System dependency graph visualization failed: {e}")

    def _persist_dependency_report(self, dependency_report: Dict[str, Any]):
        """
        Persist comprehensive dependency report

        Args:
            dependency_report (Dict): Detailed dependency report
        """
        try:
            report_path = os.path.join(
                self.log_dir,
                f'system_dependency_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            )

            with open(report_path, "w") as f:
                json.dump(dependency_report, f, indent=2)

            self.logger.info(f"System dependency report persisted: {report_path}")

        except Exception as e:
            self.logger.error(f"System dependency report persistence failed: {e}")

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
        for module, coupling in self.dependency_graph.get(
            "coupling_coefficient", {}
        ).items():
            if coupling > 0.7:  # High coupling threshold
                insights["high_coupling_modules"].append(
                    {"module": module, "coupling_coefficient": coupling}
                )

        # Identify potential refactoring candidates
        for module, fan_in in self.dependency_graph.get("fan_in", {}).items():
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
    Demonstrate System Dependency Analysis
    """
    dependency_analyzer = SystemDependencyAnalyzer()

    # Perform comprehensive system dependency analysis
    dependency_report = dependency_analyzer.analyze_system_dependencies()

    print("\n🌐 System Dependency Analysis Results 🌐")
    print(f"Total Modules: {dependency_report.get('total_modules', 0)}")
    print(f"Total Dependencies: {dependency_report.get('total_dependencies', 0)}")

    print("\nCircular Dependencies:")
    for cycle in dependency_report.get("circular_dependencies", []):
        print(" -> ".join(cycle))

    # Generate dependency insights
    dependency_insights = dependency_analyzer.generate_dependency_insights()

    print("\nDependency Insights:")
    for category, insights in dependency_insights.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for insight in insights:
            print(f"- {insight}")


if __name__ == "__main__":
    main()
