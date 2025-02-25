#!/usr/bin/env python3
"""
Ultra-Comprehensive System Architecture Mapping and Cross-Referencing Framework

Provides advanced capabilities for:
- Systematic architectural analysis
- Comprehensive dependency tracking
- Intelligent system structure visualization
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

import matplotlib.pyplot as plt
import networkx as nx

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ArchitecturalComponent:
    """
    Comprehensive representation of a system architectural component
    """

    name: str
    type: str
    path: str
    dependencies: List[str]
    complexity_metrics: Dict[str, Any]
    documentation_status: Dict[str, Any]


class SystemArchitectureMapper:
    """
    Advanced system architecture mapping and cross-referencing system
    """

    def __init__(
        self,
        base_dir: str = "/opt/SutazAI",
        log_dir: Optional[str] = None,
    ):
        """
        Initialize System Architecture Mapper

        Args:
            base_dir (str): Base project directory
            log_dir (Optional[str]): Custom log directory
        """
        # Core configuration
        self.base_dir = base_dir
        self.log_dir = log_dir or os.path.join(base_dir, "logs", "system_architecture")
        os.makedirs(self.log_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(self.log_dir, "system_architecture_mapper.log")
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("SutazAI.SystemArchitectureMapper")

        # Architectural tracking
        self.architectural_graph = nx.DiGraph()
        self.architectural_components: List[ArchitecturalComponent] = []

    def map_system_architecture(
        self, search_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive system architecture mapping

        Args:
            search_paths (Optional[List[str]]): Specific paths to search

        Returns:
            Comprehensive architectural mapping
        """
        # Reset architectural graph and components
        self.architectural_graph.clear()
        self.architectural_components.clear()

        # Default search paths
        search_paths = search_paths or [
            "core_system",
            "workers",
            "ai_agents",
            "services",
            "scripts",
        ]

        architecture_report = {
            "timestamp": datetime.now().isoformat(),
            "total_components": 0,
            "component_types": {},
            "dependency_metrics": {},
            "complexity_distribution": {},
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
                            # Analyze architectural component
                            component = self._analyze_architectural_component(file_path)

                            # Update architectural graph
                            self._update_architectural_graph(component)

                            # Update architecture report
                            component_type = component.type
                            architecture_report["component_types"][component_type] = (
                                architecture_report["component_types"].get(
                                    component_type, 0
                                )
                                + 1
                            )

                            architecture_report["total_components"] += 1

                            # Track complexity distribution
                            architecture_report["complexity_distribution"][
                                relative_path
                            ] = component.complexity_metrics

                        except Exception as e:
                            self.logger.warning(
                                f"Architectural analysis failed for {file_path}: {e}"
                            )

        # Calculate dependency metrics
        architecture_report["dependency_metrics"] = self._calculate_dependency_metrics()

        # Detect architectural patterns
        architecture_report["architectural_patterns"] = (
            self._detect_architectural_patterns()
        )

        # Visualize architectural graph
        self._visualize_architectural_graph()

        # Persist architecture report
        self._persist_architecture_report(architecture_report)

        return architecture_report

    def _analyze_architectural_component(
        self, file_path: str
    ) -> ArchitecturalComponent:
        """
        Analyze a single architectural component

        Args:
            file_path (str): Path to the Python file

        Returns:
            Detailed architectural component representation
        """
        with open(file_path, "r") as f:
            content = f.read()

        # Parse AST
        tree = ast.parse(content)

        # Determine component type
        component_type = self._determine_component_type(tree)

        # Analyze dependencies
        dependencies = self._extract_dependencies(tree)

        # Calculate complexity metrics
        complexity_metrics = self._calculate_complexity_metrics(tree)

        # Check documentation status
        documentation_status = self._check_documentation_status(tree)

        return ArchitecturalComponent(
            name=os.path.splitext(os.path.basename(file_path))[0],
            type=component_type,
            path=file_path,
            dependencies=dependencies,
            complexity_metrics=complexity_metrics,
            documentation_status=documentation_status,
        )

    def _determine_component_type(self, tree: ast.AST) -> str:
        """
        Determine the type of architectural component

        Args:
            tree (ast.AST): Abstract syntax tree of the component

        Returns:
            Component type classification
        """
        # Analyze AST to determine component type
        component_types = {
            "class": sum(
                1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            ),
            "function": sum(
                1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            ),
            "module": 1,  # Every file is a module
        }

        # Prioritize classification
        if component_types["class"] > 0:
            return "class"
        elif component_types["function"] > 0:
            return "function"
        else:
            return "module"

    def _extract_dependencies(self, tree: ast.AST) -> List[str]:
        """
        Extract dependencies from the component

        Args:
            tree (ast.AST): Abstract syntax tree of the component

        Returns:
            List of dependencies
        """
        dependencies = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies.extend([alias.name for alias in node.names])

            elif isinstance(node, ast.ImportFrom):
                base_module = node.module or ""
                dependencies.extend(
                    [
                        (f"{base_module}.{alias.name}" if base_module else alias.name)
                        for alias in node.names
                    ]
                )

        return dependencies

    def _calculate_complexity_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Calculate comprehensive complexity metrics

        Args:
            tree (ast.AST): Abstract syntax tree of the component

        Returns:
            Complexity metrics dictionary
        """
        return {
            "cyclomatic_complexity": sum(
                1
                for node in ast.walk(tree)
                if isinstance(
                    node,
                    (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler),
                )
            ),
            "function_count": sum(
                1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            ),
            "class_count": sum(
                1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            ),
            "lines_of_code": len(ast.unparse(tree).splitlines()),
        }

    def _check_documentation_status(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Check documentation status of the component

        Args:
            tree (ast.AST): Abstract syntax tree of the component

        Returns:
            Documentation status metrics
        """
        total_elements = 0
        documented_elements = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                total_elements += 1

                # Check for docstring
                if ast.get_docstring(node):
                    documented_elements += 1

        return {
            "total_elements": total_elements,
            "documented_elements": documented_elements,
            "documentation_coverage": (
                (documented_elements / total_elements * 100)
                if total_elements > 0
                else 0
            ),
        }

    def _update_architectural_graph(self, component: ArchitecturalComponent):
        """
        Update architectural dependency graph

        Args:
            component (ArchitecturalComponent): Architectural component details
        """
        # Add component as a node
        self.architectural_graph.add_node(component.name)

        # Add dependency edges
        for dependency in component.dependencies:
            self.architectural_graph.add_edge(component.name, dependency)

        # Store component
        self.architectural_components.append(component)

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
                "degree": nx.degree_centrality(self.architectural_graph),
                "betweenness": nx.betweenness_centrality(self.architectural_graph),
                "closeness": nx.closeness_centrality(self.architectural_graph),
            },
        }

        # Calculate fan-in and fan-out
        for node in self.architectural_graph.nodes():
            metrics["fan_in"][node] = self.architectural_graph.in_degree(node)
            metrics["fan_out"][node] = self.architectural_graph.out_degree(node)

            # Calculate coupling coefficient
            try:
                metrics["coupling_coefficient"][node] = len(
                    list(self.architectural_graph.predecessors(node))
                ) / (len(list(self.architectural_graph.nodes())) - 1)
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
        for component in self.architectural_components:
            # Simple pattern detection (can be expanded)
            if "getInstance" in component.name:
                patterns["design_patterns"]["singleton"] += 1

            if "create_" in component.name:
                patterns["design_patterns"]["factory"] += 1

            if "execute" in component.name:
                patterns["design_patterns"]["strategy"] += 1

            if any("@" in str(dep) for dep in component.dependencies):
                patterns["design_patterns"]["decorator"] += 1

        return patterns

    def _visualize_architectural_graph(self):
        """
        Create visual representation of the architectural graph
        """
        try:
            plt.figure(figsize=(20, 20))
            pos = nx.spring_layout(self.architectural_graph, k=0.5, iterations=50)

            nx.draw(
                self.architectural_graph,
                pos,
                with_labels=True,
                node_color="lightblue",
                node_size=300,
                font_size=8,
                font_weight="bold",
                arrows=True,
            )

            # Save visualization
            plt.title("SutazAI System Architecture Graph")
            plt.tight_layout()
            visualization_path = os.path.join(
                self.log_dir,
                f'system_architecture_graph_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
            )
            plt.savefig(visualization_path, dpi=300)
            plt.close()

            self.logger.info(
                f"System architecture graph visualization saved: {visualization_path}"
            )

        except Exception as e:
            self.logger.error(f"System architecture graph visualization failed: {e}")

    def _persist_architecture_report(self, architecture_report: Dict[str, Any]):
        """
        Persist comprehensive architecture report

        Args:
            architecture_report (Dict): Detailed architecture report
        """
        try:
            report_path = os.path.join(
                self.log_dir,
                f'system_architecture_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            )

            with open(report_path, "w") as f:
                json.dump(architecture_report, f, indent=2)

            self.logger.info(f"System architecture report persisted: {report_path}")

        except Exception as e:
            self.logger.error(f"System architecture report persistence failed: {e}")

    def generate_architectural_insights(self) -> Dict[str, Any]:
        """
        Generate advanced architectural insights and recommendations

        Returns:
            Comprehensive architectural insights
        """
        insights = {
            "high_coupling_components": [],
            "potential_refactoring_candidates": [],
            "architectural_recommendations": [],
        }

        # Identify high coupling components
        for component, coupling in self.architectural_graph.get(
            "coupling_coefficient", {}
        ).items():
            if coupling > 0.7:  # High coupling threshold
                insights["high_coupling_components"].append(
                    {"component": component, "coupling_coefficient": coupling}
                )

        # Identify potential refactoring candidates
        for component, fan_in in self.architectural_graph.get("fan_in", {}).items():
            if fan_in > 10:  # High fan-in threshold
                insights["potential_refactoring_candidates"].append(
                    {"component": component, "fan_in": fan_in}
                )

        # Generate architectural recommendations
        if insights["high_coupling_components"]:
            insights["architectural_recommendations"].append(
                "Consider applying dependency inversion principle to reduce coupling"
            )

        if insights["potential_refactoring_candidates"]:
            insights["architectural_recommendations"].append(
                "Evaluate modularization strategies for highly connected components"
            )

        return insights


def main():
    """
    Demonstrate System Architecture Mapping
    """
    architecture_mapper = SystemArchitectureMapper()

    # Perform comprehensive system architecture mapping
    architecture_report = architecture_mapper.map_system_architecture()

    print("\n🏗️ System Architecture Analysis Results 🏗️")
    print(f"Total Components: {architecture_report.get('total_components', 0)}")

    print("\nComponent Types:")
    for category, count in architecture_report.get("component_types", {}).items():
        print(f"- {category.replace('_', ' ').title()}: {count}")

    # Generate architectural insights
    architectural_insights = architecture_mapper.generate_architectural_insights()

    print("\nArchitectural Insights:")
    for category, insights in architectural_insights.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for insight in insights:
            print(f"- {insight}")


if __name__ == "__main__":
    main()
