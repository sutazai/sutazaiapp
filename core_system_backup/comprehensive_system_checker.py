#!/usr/bin/env python3
"""
Ultra-Comprehensive System Checker and Optimization Mechanism

Provides an autonomous, intelligent system for:
- Comprehensive codebase analysis
- Cross-referencing dependencies
- Identifying potential issues
- Proactive optimization
- Systematic documentation of changes
"""

import ast
import importlib
import inspect
import json
import logging
import os
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Internal system imports
from core_system.dependency_mapper import AdvancedDependencyMapper
from core_system.system_integration_framework import (
    UltraComprehensiveSystemIntegrationFramework,
)
from scripts.comprehensive_system_audit import UltraComprehensiveSystemAuditor
from scripts.system_optimizer import AdvancedSystemOptimizer


class ComprehensiveSystemChecker:
    """
    Ultra-Comprehensive Autonomous System Checking and Optimization Mechanism

    Capabilities:
    - Deep codebase analysis
    - Intelligent cross-referencing
    - Proactive issue detection
    - Systematic optimization
    - Comprehensive change tracking
    """

    def __init__(
        self,
        base_dir: str = "/opt/sutazai_project/SutazAI",
        log_dir: Optional[str] = None,
    ):
        """
        Initialize Comprehensive System Checker

        Args:
            base_dir (str): Base project directory
            log_dir (Optional[str]): Custom log directory
        """
        # Core configuration
        self.base_dir = base_dir
        self.log_dir = log_dir or os.path.join(
            base_dir, "logs", "system_checker"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(
                        self.log_dir, "comprehensive_system_checker.log"
                    )
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("SutazAI.ComprehensiveSystemChecker")

        # Initialize core system components
        self.dependency_mapper = AdvancedDependencyMapper(base_dir)
        self.system_auditor = UltraComprehensiveSystemAuditor(base_dir)
        self.system_optimizer = AdvancedSystemOptimizer(base_dir)
        self.integration_framework = (
            UltraComprehensiveSystemIntegrationFramework(base_dir)
        )

        # Tracking and analysis
        self.system_dependency_graph = nx.DiGraph()
        self.comprehensive_analysis_results = {}
        self.change_log = []

    def perform_comprehensive_system_check(self) -> Dict[str, Any]:
        """
        Perform an ultra-comprehensive system check and analysis

        Returns:
            Comprehensive system analysis results
        """
        # Reset analysis results
        self.comprehensive_analysis_results = {
            "timestamp": time.time(),
            "dependency_analysis": {},
            "code_structure_analysis": {},
            "potential_issues": [],
            "optimization_recommendations": [],
        }

        try:
            # Dependency mapping
            dependency_map = self._analyze_system_dependencies()

            # Code structure analysis
            code_structure = self._analyze_code_structure()

            # Cross-reference and validate
            self._cross_reference_dependencies(dependency_map)

            # Identify potential issues
            potential_issues = self._identify_potential_system_issues()

            # Generate optimization recommendations
            optimization_recommendations = (
                self._generate_optimization_recommendations()
            )

            # Update comprehensive analysis results
            self.comprehensive_analysis_results.update(
                {
                    "dependency_analysis": dependency_map,
                    "code_structure_analysis": code_structure,
                    "potential_issues": potential_issues,
                    "optimization_recommendations": optimization_recommendations,
                }
            )

            # Persist analysis results
            self._persist_analysis_results()

            return self.comprehensive_analysis_results

        except Exception as e:
            self.logger.error(f"Comprehensive system check failed: {e}")
            return {}

    def _analyze_system_dependencies(self) -> Dict[str, Any]:
        """
        Perform advanced system dependency analysis

        Returns:
            Comprehensive dependency mapping
        """
        dependency_map = {
            "module_dependencies": {},
            "import_graph": {},
            "circular_dependencies": [],
            "cross_module_interactions": {},
        }

        # Use dependency mapper to create comprehensive dependency graph
        mapped_dependencies = self.dependency_mapper.map_system_dependencies()

        # Extract and process dependency information
        dependency_map["module_dependencies"] = mapped_dependencies.get(
            "modules", {}
        )
        dependency_map["circular_dependencies"] = mapped_dependencies.get(
            "circular_dependencies", []
        )

        # Build import graph and cross-module interactions
        for module, details in dependency_map["module_dependencies"].items():
            dependencies = details.get("dependencies", [])

            # Build import graph
            dependency_map["import_graph"][module] = dependencies

            # Analyze cross-module interactions
            for dep in dependencies:
                if dep not in dependency_map["cross_module_interactions"]:
                    dependency_map["cross_module_interactions"][dep] = []
                dependency_map["cross_module_interactions"][dep].append(module)

        return dependency_map

    def _analyze_code_structure(self) -> Dict[str, Any]:
        """
        Perform comprehensive code structure analysis

        Returns:
            Code structure analysis results
        """
        code_structure = {
            "files": {},
            "complexity_metrics": {},
            "architectural_patterns": {},
        }

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.base_dir)

                    try:
                        with open(file_path, "r") as f:
                            content = f.read()

                        # Analyze file structure
                        structure = self._extract_file_structure(content)

                        # Calculate complexity
                        complexity = self._calculate_file_complexity(content)

                        code_structure["files"][relative_path] = structure
                        code_structure["complexity_metrics"][
                            relative_path
                        ] = complexity

                    except Exception as e:
                        self.logger.warning(
                            f"Code structure analysis failed for {file_path}: {e}"
                        )

        return code_structure

    def _extract_file_structure(self, content: str) -> Dict[str, Any]:
        """
        Extract detailed file structure information

        Args:
            content (str): File content

        Returns:
            File structure details
        """
        try:
            tree = ast.parse(content)

            structure = {
                "classes": [],
                "functions": [],
                "imports": [],
                "global_variables": [],
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    structure["classes"].append(
                        {
                            "name": node.name,
                            "methods": [
                                method.name
                                for method in node.body
                                if isinstance(method, ast.FunctionDef)
                            ],
                        }
                    )

                elif isinstance(node, ast.FunctionDef):
                    structure["functions"].append(
                        {
                            "name": node.name,
                            "args": [arg.arg for arg in node.args.args],
                            "complexity": sum(
                                1
                                for child in ast.walk(node)
                                if isinstance(
                                    child, (ast.If, ast.For, ast.While)
                                )
                            ),
                        }
                    )

                elif isinstance(node, ast.Import):
                    structure["imports"].extend(
                        [alias.name for alias in node.names]
                    )

                elif isinstance(node, ast.ImportFrom):
                    structure["imports"].extend(
                        [f"{node.module}.{alias.name}" for alias in node.names]
                    )

                elif isinstance(node, ast.Assign) and isinstance(
                    node.targets[0], ast.Name
                ):
                    structure["global_variables"].append(node.targets[0].id)

            return structure

        except Exception as e:
            self.logger.warning(f"File structure extraction failed: {e}")
            return {}

    def _calculate_file_complexity(self, content: str) -> Dict[str, Any]:
        """
        Calculate comprehensive file complexity metrics

        Args:
            content (str): File content

        Returns:
            Complexity metrics
        """
        try:
            tree = ast.parse(content)

            return {
                "lines_of_code": len(content.splitlines()),
                "cyclomatic_complexity": sum(
                    1
                    for node in ast.walk(tree)
                    if isinstance(
                        node,
                        (
                            ast.If,
                            ast.While,
                            ast.For,
                            ast.Try,
                            ast.ExceptHandler,
                        ),
                    )
                ),
                "function_count": sum(
                    1
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                ),
                "class_count": sum(
                    1
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef)
                ),
            }

        except Exception as e:
            self.logger.warning(f"Complexity calculation failed: {e}")
            return {}

    def _cross_reference_dependencies(self, dependency_map: Dict[str, Any]):
        """
        Cross-reference and validate system dependencies

        Args:
            dependency_map (Dict): Comprehensive dependency mapping
        """
        # Build dependency graph
        for module, dependencies in dependency_map.get(
            "import_graph", {}
        ).items():
            for dep in dependencies:
                self.system_dependency_graph.add_edge(module, dep)

    def _identify_potential_system_issues(self) -> List[Dict[str, Any]]:
        """
        Identify potential system issues through comprehensive analysis

        Returns:
            List of potential issues
        """
        potential_issues = []

        # Check for circular dependencies
        circular_deps = list(nx.simple_cycles(self.system_dependency_graph))
        for cycle in circular_deps:
            potential_issues.append(
                {
                    "type": "circular_dependency",
                    "modules": cycle,
                    "severity": "high",
                }
            )

        # Analyze code complexity
        for file, complexity in (
            self.comprehensive_analysis_results.get(
                "code_structure_analysis", {}
            )
            .get("complexity_metrics", {})
            .items()
        ):
            if complexity.get("cyclomatic_complexity", 0) > 15:
                potential_issues.append(
                    {
                        "type": "high_complexity",
                        "file": file,
                        "complexity": complexity.get("cyclomatic_complexity"),
                        "severity": "medium",
                    }
                )

        return potential_issues

    def _generate_optimization_recommendations(self) -> List[str]:
        """
        Generate intelligent optimization recommendations

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Recommendations based on dependency analysis
        if self.comprehensive_analysis_results.get("potential_issues"):
            for issue in self.comprehensive_analysis_results[
                "potential_issues"
            ]:
                if issue["type"] == "circular_dependency":
                    recommendations.append(
                        f"Resolve circular dependency between modules: {', '.join(issue['modules'])}"
                    )

                if issue["type"] == "high_complexity":
                    recommendations.append(
                        f"Refactor high-complexity file {issue['file']} (Complexity: {issue['complexity']})"
                    )

        return recommendations

    def _persist_analysis_results(self):
        """
        Persist comprehensive analysis results
        """
        try:
            output_file = os.path.join(
                self.log_dir,
                f'comprehensive_system_check_{time.strftime("%Y%m%d_%H%M%S")}.json',
            )

            with open(output_file, "w") as f:
                json.dump(self.comprehensive_analysis_results, f, indent=2)

            self.logger.info(
                f"Comprehensive system check results persisted: {output_file}"
            )

        except Exception as e:
            self.logger.error(f"Analysis results persistence failed: {e}")

    def start_continuous_system_checking(self, interval: int = 3600):
        """
        Start continuous autonomous system checking

        Args:
            interval (int): Checking cycle interval in seconds
        """

        def system_check_worker():
            """
            Background worker for continuous system checking
            """
            while True:
                try:
                    # Perform comprehensive system check
                    system_check_results = (
                        self.perform_comprehensive_system_check()
                    )

                    # Log key insights
                    self.logger.info(
                        "Continuous system checking cycle completed"
                    )
                    self.logger.info(
                        f"Potential Issues: {len(system_check_results.get('potential_issues', []))}"
                    )
                    self.logger.info(
                        f"Optimization Recommendations: {system_check_results.get('optimization_recommendations', [])}"
                    )

                    # Wait for next checking cycle
                    time.sleep(interval)

                except Exception as e:
                    self.logger.error(
                        f"Continuous system checking failed: {e}"
                    )
                    time.sleep(interval)  # Backoff on continuous errors

        # Start system checking thread
        system_check_thread = threading.Thread(
            target=system_check_worker, daemon=True
        )
        system_check_thread.start()

        self.logger.info("Continuous system checking started")


def main():
    """
    Main execution for comprehensive system checking
    """
    try:
        # Initialize comprehensive system checker
        system_checker = ComprehensiveSystemChecker()

        # Perform initial comprehensive system check
        system_check_results = (
            system_checker.perform_comprehensive_system_check()
        )

        print("\n🔍 Comprehensive System Check Results 🔍")

        print("\nPotential Issues:")
        for issue in system_check_results.get("potential_issues", []):
            print(f"- {issue['type'].replace('_', ' ').title()}: {issue}")

        print("\nOptimization Recommendations:")
        for recommendation in system_check_results.get(
            "optimization_recommendations", []
        ):
            print(f"- {recommendation}")

        # Optional: Start continuous system checking
        system_checker.start_continuous_system_checking()

        # Keep main thread alive
        while True:
            time.sleep(3600)

    except Exception as e:
        logging.critical(f"Comprehensive system checking failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
