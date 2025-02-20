#!/usr/bin/env python3
"""
Ultra-Comprehensive Architectural Integrity and Cross-Referencing System

Provides advanced capabilities for:
- Systematic architectural analysis
- Comprehensive cross-referencing
- Structural integrity validation
- Autonomous system optimization
"""

import ast
import importlib
import inspect
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Internal system imports
from core_system.comprehensive_dependency_manager import (
    ComprehensiveDependencyManager,
)


@dataclass
class ArchitecturalIntegrityReport:
    """
    Comprehensive architectural integrity analysis report
    """

    timestamp: str
    structural_analysis: Dict[str, Any]
    code_quality_metrics: Dict[str, Any]
    architectural_patterns: Dict[str, Any]
    integrity_issues: List[Dict[str, Any]]
    optimization_recommendations: List[str]
    cross_reference_map: Dict[str, Any]


class ArchitecturalIntegrityManager:
    """
    Advanced architectural integrity and cross-referencing system
    """

    def __init__(
        self,
        base_dir: str = "/opt/sutazai_project/SutazAI",
        log_dir: Optional[str] = None,
    ):
        """
        Initialize Architectural Integrity Manager

        Args:
            base_dir (str): Base project directory
            log_dir (Optional[str]): Custom log directory
        """
        # Core configuration
        self.base_dir = base_dir
        self.log_dir = log_dir or os.path.join(
            base_dir, "logs", "architectural_integrity"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(self.log_dir, "architectural_integrity.log")
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(
            "SutazAI.ArchitecturalIntegrityManager"
        )

        # Initialize dependency manager
        self.dependency_manager = ComprehensiveDependencyManager(
            base_dir, log_dir
        )

        # Tracking and analysis
        self.architectural_graph = nx.DiGraph()
        self.cross_reference_map = {}

    def perform_architectural_integrity_analysis(
        self,
    ) -> ArchitecturalIntegrityReport:
        """
        Perform comprehensive architectural integrity analysis

        Returns:
            Detailed architectural integrity report
        """
        # Reset architectural graph and cross-reference map
        self.architectural_graph.clear()
        self.cross_reference_map.clear()

        try:
            # Structural analysis
            structural_analysis = self._analyze_project_structure()

            # Code quality metrics
            code_quality_metrics = self._calculate_code_quality_metrics()

            # Architectural patterns
            architectural_patterns = self._identify_architectural_patterns()

            # Integrity issues detection
            integrity_issues = self._detect_architectural_integrity_issues()

            # Cross-referencing
            cross_reference_map = self._generate_cross_reference_map()

            # Generate optimization recommendations
            optimization_recommendations = (
                self._generate_optimization_recommendations(
                    structural_analysis, code_quality_metrics, integrity_issues
                )
            )

            # Create comprehensive architectural integrity report
            architectural_report = ArchitecturalIntegrityReport(
                timestamp=datetime.now().isoformat(),
                structural_analysis=structural_analysis,
                code_quality_metrics=code_quality_metrics,
                architectural_patterns=architectural_patterns,
                integrity_issues=integrity_issues,
                optimization_recommendations=optimization_recommendations,
                cross_reference_map=cross_reference_map,
            )

            # Persist analysis report
            self._persist_architectural_report(architectural_report)

            return architectural_report

        except Exception as e:
            self.logger.error(f"Architectural integrity analysis failed: {e}")
            return ArchitecturalIntegrityReport(
                timestamp=datetime.now().isoformat(),
                structural_analysis={},
                code_quality_metrics={},
                architectural_patterns={},
                integrity_issues=[],
                optimization_recommendations=[],
                cross_reference_map={},
            )

    def _analyze_project_structure(self) -> Dict[str, Any]:
        """
        Perform comprehensive project structure analysis

        Returns:
            Detailed structural analysis
        """
        structural_analysis = {
            "directories": {},
            "module_hierarchy": {},
            "file_distribution": {},
        }

        # Analyze directory structure
        for root, dirs, files in os.walk(self.base_dir):
            relative_path = os.path.relpath(root, self.base_dir)

            # Directory analysis
            structural_analysis["directories"][relative_path] = {
                "total_subdirectories": len(dirs),
                "total_files": len(files),
                "file_types": {},
            }

            # File type distribution
            for file in files:
                file_ext = os.path.splitext(file)[1]
                structural_analysis["directories"][relative_path][
                    "file_types"
                ][file_ext] = (
                    structural_analysis["directories"][relative_path][
                        "file_types"
                    ].get(file_ext, 0)
                    + 1
                )

            # Module hierarchy
            if relative_path != ".":
                module_path = relative_path.replace(os.path.sep, ".")
                structural_analysis["module_hierarchy"][module_path] = {
                    "python_files": [f for f in files if f.endswith(".py")],
                    "depth": len(relative_path.split(os.path.sep)),
                }

        return structural_analysis

    def _calculate_code_quality_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive code quality metrics

        Returns:
            Code quality metrics
        """
        code_quality_metrics = {
            "total_modules": 0,
            "complexity_distribution": {},
            "documentation_coverage": {},
            "type_hint_usage": {},
        }

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.base_dir)

                    try:
                        with open(file_path, "r") as f:
                            content = f.read()

                        # Parse AST
                        tree = ast.parse(content)

                        # Complexity metrics
                        complexity = self._calculate_module_complexity(tree)
                        code_quality_metrics["complexity_distribution"][
                            relative_path
                        ] = complexity

                        # Documentation coverage
                        doc_coverage = self._calculate_documentation_coverage(
                            tree
                        )
                        code_quality_metrics["documentation_coverage"][
                            relative_path
                        ] = doc_coverage

                        # Type hint usage
                        type_hint_usage = self._analyze_type_hint_usage(tree)
                        code_quality_metrics["type_hint_usage"][
                            relative_path
                        ] = type_hint_usage

                        code_quality_metrics["total_modules"] += 1

                    except Exception as e:
                        self.logger.warning(
                            f"Code quality analysis failed for {file_path}: {e}"
                        )

        return code_quality_metrics

    def _calculate_module_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """
        Calculate module complexity metrics

        Args:
            tree (ast.AST): Abstract syntax tree of the module

        Returns:
            Complexity metrics
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
                1
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            ),
            "class_count": sum(
                1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            ),
        }

    def _calculate_documentation_coverage(
        self, tree: ast.AST
    ) -> Dict[str, float]:
        """
        Calculate documentation coverage

        Args:
            tree (ast.AST): Abstract syntax tree of the module

        Returns:
            Documentation coverage metrics
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
            "coverage_percentage": (
                (documented_elements / total_elements * 100)
                if total_elements > 0
                else 0
            ),
        }

    def _analyze_type_hint_usage(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Analyze type hint usage

        Args:
            tree (ast.AST): Abstract syntax tree of the module

        Returns:
            Type hint usage metrics
        """
        type_hint_metrics = {
            "function_type_hints": 0,
            "variable_type_hints": 0,
            "total_functions": 0,
            "total_variables": 0,
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                type_hint_metrics["total_functions"] += 1

                # Check function arguments type hints
                if node.args.annotations:
                    type_hint_metrics["function_type_hints"] += 1

                # Check return type hint
                if node.returns:
                    type_hint_metrics["function_type_hints"] += 1

            elif isinstance(node, ast.AnnAssign):
                type_hint_metrics["total_variables"] += 1
                type_hint_metrics["variable_type_hints"] += 1

        return type_hint_metrics

    def _identify_architectural_patterns(self) -> Dict[str, Any]:
        """
        Identify architectural patterns and characteristics

        Returns:
            Architectural pattern insights
        """
        architectural_patterns = {
            "module_categories": {
                "core_system": 0,
                "workers": 0,
                "services": 0,
                "utils": 0,
                "external": 0,
            },
            "design_patterns": {
                "singleton": 0,
                "factory": 0,
                "strategy": 0,
                "decorator": 0,
            },
        }

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)

                    # Categorize modules
                    if "core_system" in file_path:
                        architectural_patterns["module_categories"][
                            "core_system"
                        ] += 1
                    elif "workers" in file_path:
                        architectural_patterns["module_categories"][
                            "workers"
                        ] += 1
                    elif "services" in file_path:
                        architectural_patterns["module_categories"][
                            "services"
                        ] += 1
                    elif "utils" in file_path:
                        architectural_patterns["module_categories"][
                            "utils"
                        ] += 1
                    else:
                        architectural_patterns["module_categories"][
                            "external"
                        ] += 1

                    # Detect design patterns
                    try:
                        with open(file_path, "r") as f:
                            content = f.read()

                        # Simple pattern detection (can be expanded)
                        if re.search(
                            r"@classmethod\s+def\s+getInstance", content
                        ):
                            architectural_patterns["design_patterns"][
                                "singleton"
                            ] += 1

                        if re.search(r"def\s+create_", content):
                            architectural_patterns["design_patterns"][
                                "factory"
                            ] += 1

                        if re.search(r"def\s+execute\s*\(", content):
                            architectural_patterns["design_patterns"][
                                "strategy"
                            ] += 1

                        if re.search(r"@\w+\s*def\s+\w+\s*\(", content):
                            architectural_patterns["design_patterns"][
                                "decorator"
                            ] += 1

                    except Exception as e:
                        self.logger.warning(
                            f"Design pattern detection failed for {file_path}: {e}"
                        )

        return architectural_patterns

    def _detect_architectural_integrity_issues(self) -> List[Dict[str, Any]]:
        """
        Detect potential architectural integrity issues

        Returns:
            List of architectural integrity issues
        """
        integrity_issues = []

        # Dependency analysis
        dependency_report = (
            self.dependency_manager.analyze_project_dependencies()
        )

        # Circular dependencies
        for cycle in dependency_report.circular_dependencies:
            integrity_issues.append(
                {
                    "type": "circular_dependency",
                    "modules": cycle,
                    "severity": "high",
                }
            )

        # High coupling detection
        for module, metrics in dependency_report.architectural_insights.get(
            "coupling_metrics", {}
        ).items():
            if metrics.get("fan_in", 0) > 5 or metrics.get("fan_out", 0) > 5:
                integrity_issues.append(
                    {
                        "type": "high_coupling",
                        "module": module,
                        "fan_in": metrics.get("fan_in", 0),
                        "fan_out": metrics.get("fan_out", 0),
                        "severity": "medium",
                    }
                )

        return integrity_issues

    def _generate_cross_reference_map(self) -> Dict[str, Any]:
        """
        Generate comprehensive cross-reference map

        Returns:
            Detailed cross-reference mapping
        """
        cross_reference_map = {
            "module_imports": {},
            "inheritance_relationships": {},
            "function_calls": {},
        }

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.base_dir)

                    try:
                        with open(file_path, "r") as f:
                            content = f.read()

                        tree = ast.parse(content)

                        # Module imports
                        cross_reference_map["module_imports"][
                            relative_path
                        ] = [
                            node.names[0].name
                            for node in ast.walk(tree)
                            if isinstance(node, (ast.Import, ast.ImportFrom))
                        ]

                        # Inheritance relationships
                        cross_reference_map["inheritance_relationships"][
                            relative_path
                        ] = [
                            {
                                "class": node.name,
                                "base_classes": [
                                    base.id
                                    for base in node.bases
                                    if isinstance(base, ast.Name)
                                ],
                            }
                            for node in ast.walk(tree)
                            if isinstance(node, ast.ClassDef)
                        ]

                        # Function calls
                        cross_reference_map["function_calls"][
                            relative_path
                        ] = [
                            {
                                "function": (
                                    node.func.id
                                    if isinstance(node.func, ast.Name)
                                    else (
                                        f"{node.func.value.id}.{node.func.attr}"
                                        if isinstance(node.func, ast.Attribute)
                                        else "Unknown"
                                    )
                                ),
                                "line": node.lineno,
                            }
                            for node in ast.walk(tree)
                            if isinstance(node, ast.Call)
                        ]

                    except Exception as e:
                        self.logger.warning(
                            f"Cross-reference mapping failed for {file_path}: {e}"
                        )

        return cross_reference_map

    def _generate_optimization_recommendations(
        self,
        structural_analysis: Dict[str, Any],
        code_quality_metrics: Dict[str, Any],
        integrity_issues: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Generate intelligent optimization recommendations

        Args:
            structural_analysis (Dict): Project structural analysis
            code_quality_metrics (Dict): Code quality metrics
            integrity_issues (List): Detected architectural integrity issues

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Structural recommendations
        for dir_path, dir_info in structural_analysis.get(
            "directories", {}
        ).items():
            if dir_info.get("total_files", 0) > 50:
                recommendations.append(
                    f"Refactor directory {dir_path} with high file count: {dir_info['total_files']} files"
                )

        # Code quality recommendations
        for module, complexity in code_quality_metrics.get(
            "complexity_distribution", {}
        ).items():
            if complexity.get("cyclomatic_complexity", 0) > 15:
                recommendations.append(
                    f"Refactor high-complexity module {module} (Complexity: {complexity['cyclomatic_complexity']})"
                )

        # Integrity issue recommendations
        for issue in integrity_issues:
            if issue["type"] == "circular_dependency":
                recommendations.append(
                    f"Resolve circular dependency between modules: {', '.join(issue['modules'])}"
                )

            if issue["type"] == "high_coupling":
                recommendations.append(
                    f"Reduce coupling for module {issue['module']} (Fan-In: {issue['fan_in']}, Fan-Out: {issue['fan_out']})"
                )

        return recommendations

    def _persist_architectural_report(
        self, report: ArchitecturalIntegrityReport
    ):
        """
        Persist architectural integrity analysis report

        Args:
            report (ArchitecturalIntegrityReport): Comprehensive architectural report
        """
        try:
            output_file = os.path.join(
                self.log_dir,
                f'architectural_integrity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            )

            with open(output_file, "w") as f:
                json.dump(asdict(report), f, indent=2)

            self.logger.info(
                f"Architectural integrity report persisted: {output_file}"
            )

        except Exception as e:
            self.logger.error(f"Architectural report persistence failed: {e}")

    def visualize_architectural_graph(self, output_path: Optional[str] = None):
        """
        Generate visual representation of architectural graph

        Args:
            output_path (Optional[str]): Custom output path for graph visualization
        """
        try:
            import matplotlib.pyplot as plt

            # Build architectural graph from cross-reference map
            for module, imports in self.cross_reference_map.get(
                "module_imports", {}
            ).items():
                for imported_module in imports:
                    self.architectural_graph.add_edge(module, imported_module)

            plt.figure(figsize=(20, 20))
            pos = nx.spring_layout(
                self.architectural_graph, k=0.5, iterations=50
            )
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
            output_file = output_path or os.path.join(
                self.log_dir,
                f'architectural_graph_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
            )
            plt.title("SutazAI Architectural Dependency Graph")
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.close()

            self.logger.info(
                f"Architectural graph visualization saved: {output_file}"
            )

        except Exception as e:
            self.logger.error(f"Architectural graph visualization failed: {e}")


def main():
    """
    Demonstrate Architectural Integrity Management
    """
    architectural_manager = ArchitecturalIntegrityManager()

    # Perform architectural integrity analysis
    architectural_report = (
        architectural_manager.perform_architectural_integrity_analysis()
    )

    print("\n🏗️ Architectural Integrity Analysis Results 🏗️")
    print(
        f"Total Modules: {architectural_report.code_quality_metrics.get('total_modules', 0)}"
    )

    print("\nArchitectural Patterns:")
    for category, count in architectural_report.architectural_patterns.get(
        "module_categories", {}
    ).items():
        print(f"- {category.replace('_', ' ').title()}: {count}")

    print("\nIntegrity Issues:")
    for issue in architectural_report.integrity_issues:
        print(f"- {issue['type'].replace('_', ' ').title()}: {issue}")

    print("\nOptimization Recommendations:")
    for recommendation in architectural_report.optimization_recommendations:
        print(f"- {recommendation}")

    # Visualize architectural graph
    architectural_manager.visualize_architectural_graph()


if __name__ == "__main__":
    main()
