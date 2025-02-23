#!/usr/bin/env python3
"""
Ultra-Comprehensive Project Structure and Integrity Validation Framework

Provides an autonomous, multi-dimensional approach to:
- Systematic project structure analysis
- Comprehensive integrity checks
- Dependency mapping
- Security vulnerability detection
- Documentation quality assessment
- Architectural compliance validation
"""

import ast
import hashlib
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Advanced analysis libraries
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer


class UltraComprehensiveProjectValidator:
    """
    Advanced autonomous project structure and integrity validation system

    Capabilities:
    - Comprehensive project structure mapping
    - Semantic code analysis
    - Dependency tracking
    - Security vulnerability detection
    - Documentation quality assessment
    """

    def __init__(
        self,
        base_dir: str = "/opt/sutazai_project/SutazAI",
        log_dir: Optional[str] = None,
    ):
        """
        Initialize Ultra-Comprehensive Project Validator

        Args:
            base_dir (str): Base project directory
            log_dir (Optional[str]): Custom log directory
        """
        # Core configuration
        self.base_dir = base_dir
        self.log_dir = log_dir or os.path.join(base_dir, "logs", "project_validation")
        os.makedirs(self.log_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(self.log_dir, "project_validator.log")
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("SutazAI.ProjectValidator")

        # Validation tracking
        self.project_structure_graph = nx.DiGraph()
        self.semantic_code_map = {}
        self.validation_report = {
            "timestamp": None,
            "directory_structure": {},
            "code_quality": {},
            "security_issues": [],
            "documentation_gaps": [],
            "dependency_analysis": {},
            "optimization_recommendations": [],
        }

    def validate_project_structure(self) -> Dict[str, Any]:
        """
        Perform comprehensive project structure validation

        Returns:
            Detailed project validation report
        """
        self.validation_report["timestamp"] = os.times()

        try:
            # 1. Directory Structure Analysis
            self._analyze_directory_structure()

            # 2. Code Quality Assessment
            self._assess_code_quality()

            # 3. Security Vulnerability Detection
            self._detect_security_vulnerabilities()

            # 4. Documentation Quality Check
            self._check_documentation_quality()

            # 5. Dependency Analysis
            self._analyze_project_dependencies()

            # 6. Generate Optimization Recommendations
            self._generate_optimization_recommendations()

            # Persist validation report
            self._persist_validation_report()

        except Exception as e:
            self.logger.error(f"Comprehensive project validation failed: {e}")

        return self.validation_report

    def _analyze_directory_structure(self):
        """
        Perform comprehensive directory structure analysis
        """
        directory_analysis = {
            "total_directories": 0,
            "directory_types": {},
            "potential_issues": [],
        }

        for root, dirs, files in os.walk(self.base_dir):
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                relative_path = os.path.relpath(full_path, self.base_dir)

                # Categorize directory
                dir_type = self._categorize_directory(relative_path)
                directory_analysis["directory_types"][dir_type] = (
                    directory_analysis["directory_types"].get(dir_type, 0) + 1
                )
                directory_analysis["total_directories"] += 1

                # Add to project structure graph
                self.project_structure_graph.add_node(relative_path, type=dir_type)

        # Identify potential directory structure issues
        if len(directory_analysis["directory_types"]) < 5:
            directory_analysis["potential_issues"].append(
                "Limited directory diversity. Consider expanding project structure."
            )

        self.validation_report["directory_structure"] = directory_analysis

    def _categorize_directory(self, path: str) -> str:
        """
        Categorize directory based on its path and contents

        Args:
            path (str): Relative directory path

        Returns:
            Directory category
        """
        categories = [
            ("core", ["core_system", "system"]),
            ("ai", ["ai_agents", "machine_learning"]),
            ("backend", ["backend", "services"]),
            ("frontend", ["web_ui", "frontend"]),
            ("tests", ["tests", "verification"]),
            ("docs", ["documentation", "docs"]),
            ("config", ["configuration", "settings"]),
            ("utils", ["utilities", "helpers"]),
            ("scripts", ["deployment", "maintenance"]),
            ("security", ["authentication", "encryption"]),
        ]

        for category, keywords in categories:
            if any(keyword in path.lower() for keyword in keywords):
                return category

        return "uncategorized"

    def _assess_code_quality(self):
        """
        Perform comprehensive code quality assessment
        """
        code_quality = {
            "total_files": 0,
            "complexity_metrics": {},
            "type_hint_coverage": {},
            "potential_issues": [],
        }

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, "r") as f:
                            content = f.read()

                        # Parse AST
                        tree = ast.parse(content)

                        # Complexity analysis
                        complexity = self._calculate_code_complexity(tree)
                        code_quality["complexity_metrics"][file_path] = complexity

                        # Type hint coverage
                        type_hint_coverage = self._analyze_type_hint_coverage(tree)
                        code_quality["type_hint_coverage"][
                            file_path
                        ] = type_hint_coverage

                        code_quality["total_files"] += 1

                    except Exception as e:
                        self.logger.warning(
                            f"Code quality analysis failed for {file_path}: {e}"
                        )

        # Identify code quality issues
        high_complexity_files = [
            file
            for file, metrics in code_quality["complexity_metrics"].items()
            if metrics.get("cyclomatic_complexity", 0) > 15
        ]

        low_type_hint_files = [
            file
            for file, coverage in code_quality["type_hint_coverage"].items()
            if coverage < 0.5
        ]

        if high_complexity_files:
            code_quality["potential_issues"].append(
                f"High complexity in {len(high_complexity_files)} files"
            )

        if low_type_hint_files:
            code_quality["potential_issues"].append(
                f"Low type hint coverage in {len(low_type_hint_files)} files"
            )

        self.validation_report["code_quality"] = code_quality

    def _calculate_code_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Calculate comprehensive code complexity metrics

        Args:
            tree (ast.AST): Abstract Syntax Tree of the code

        Returns:
            Code complexity metrics
        """
        complexity = {"cyclomatic_complexity": 1, "function_count": 0, "class_count": 0}

        for node in ast.walk(tree):
            # Cyclomatic complexity calculation
            if isinstance(
                node, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)
            ):
                complexity["cyclomatic_complexity"] += 1

            # Function and class count
            if isinstance(node, ast.FunctionDef):
                complexity["function_count"] += 1
            elif isinstance(node, ast.ClassDef):
                complexity["class_count"] += 1

        return complexity

    def _analyze_type_hint_coverage(self, tree: ast.AST) -> float:
        """
        Analyze type hint coverage in the code

        Args:
            tree (ast.AST): Abstract Syntax Tree of the code

        Returns:
            Type hint coverage percentage
        """
        total_annotatable = 0
        annotated = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check function arguments
                for arg in node.args.args:
                    total_annotatable += 1
                    if arg.annotation:
                        annotated += 1

                # Check return annotation
                if node.returns:
                    total_annotatable += 1
                    annotated += 1

        return annotated / total_annotatable if total_annotatable > 0 else 0

    def _detect_security_vulnerabilities(self):
        """
        Detect potential security vulnerabilities
        """
        security_issues = []

        # Dangerous function patterns
        dangerous_patterns = [
            r"(eval|exec)\(",
            r"os\.system\(",
            r"subprocess\.(?:call|run|Popen)",
            r"pickle\.(?:load|loads)",
            r"yaml\.(?:load|unsafe_load)",
        ]

        # Sensitive information patterns
        sensitive_patterns = [
            r'(password|secret|token|api_key)\s*=\s*[\'"].*?[\'"]',
            r'(mysql|postgresql|sqlite)://.*?:[\'"].*?[\'"]',
        ]

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, "r") as f:
                            content = f.read()

                        # Check for dangerous functions
                        for pattern in dangerous_patterns:
                            matches = re.findall(pattern, content)
                            if matches:
                                security_issues.append(
                                    {
                                        "file": file_path,
                                        "type": "dangerous_function",
                                        "matches": matches,
                                    }
                                )

                        # Check for sensitive information
                        for pattern in sensitive_patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            if matches:
                                security_issues.append(
                                    {
                                        "file": file_path,
                                        "type": "sensitive_information",
                                        "matches": matches,
                                    }
                                )

                    except Exception as e:
                        self.logger.warning(
                            f"Security vulnerability scan failed for {file_path}: {e}"
                        )

        self.validation_report["security_issues"] = security_issues

    def _check_documentation_quality(self):
        """
        Assess documentation quality across the project
        """
        documentation_gaps = []

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, "r") as f:
                            content = f.read()

                        tree = ast.parse(content)

                        # Check module-level docstring
                        module_docstring = ast.get_docstring(tree)
                        if not module_docstring:
                            documentation_gaps.append(
                                {
                                    "file": file_path,
                                    "type": "module_docstring",
                                    "description": "Missing module-level docstring",
                                }
                            )

                        # Check class docstrings
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                class_docstring = ast.get_docstring(node)
                                if not class_docstring:
                                    documentation_gaps.append(
                                        {
                                            "file": file_path,
                                            "type": "class_docstring",
                                            "class_name": node.name,
                                            "description": "Missing class docstring",
                                        }
                                    )

                            # Check function docstrings
                            elif isinstance(node, ast.FunctionDef):
                                function_docstring = ast.get_docstring(node)
                                if not function_docstring:
                                    documentation_gaps.append(
                                        {
                                            "file": file_path,
                                            "type": "function_docstring",
                                            "function_name": node.name,
                                            "description": "Missing function docstring",
                                        }
                                    )

                    except Exception as e:
                        self.logger.warning(
                            f"Documentation quality check failed for {file_path}: {e}"
                        )

        self.validation_report["documentation_gaps"] = documentation_gaps

    def _analyze_project_dependencies(self):
        """
        Perform comprehensive project dependency analysis
        """
        dependency_analysis = {
            "module_dependencies": {},
            "import_graph": nx.DiGraph(),
            "circular_dependencies": [],
            "cross_module_interactions": {},
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

                        # Extract dependencies
                        dependencies = []
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                dependencies.extend(
                                    [alias.name for alias in node.names]
                                )
                            elif isinstance(node, ast.ImportFrom):
                                base_module = node.module or ""
                                dependencies.extend(
                                    [
                                        (
                                            f"{base_module}.{alias.name}"
                                            if base_module
                                            else alias.name
                                        )
                                        for alias in node.names
                                    ]
                                )

                        # Update dependency analysis
                        dependency_analysis["module_dependencies"][
                            relative_path
                        ] = dependencies

                        # Build import graph
                        dependency_analysis["import_graph"].add_node(relative_path)
                        for dep in dependencies:
                            dependency_analysis["import_graph"].add_edge(
                                relative_path, dep
                            )

                    except Exception as e:
                        self.logger.warning(
                            f"Dependency analysis failed for {file_path}: {e}"
                        )

        # Detect circular dependencies
        dependency_analysis["circular_dependencies"] = list(
            nx.simple_cycles(dependency_analysis["import_graph"])
        )

        self.validation_report["dependency_analysis"] = dependency_analysis

    def _generate_optimization_recommendations(self):
        """
        Generate intelligent optimization recommendations
        """
        recommendations = []

        # Directory structure recommendations
        if self.validation_report["directory_structure"].get("potential_issues"):
            recommendations.extend(
                self.validation_report["directory_structure"]["potential_issues"]
            )

        # Code quality recommendations
        code_quality = self.validation_report["code_quality"]
        if code_quality.get("potential_issues"):
            recommendations.extend(code_quality["potential_issues"])

        # Security recommendations
        if self.validation_report["security_issues"]:
            recommendations.append(
                f"Address {len(self.validation_report['security_issues'])} security vulnerabilities"
            )

        # Documentation recommendations
        if self.validation_report["documentation_gaps"]:
            recommendations.append(
                f"Fill {len(self.validation_report['documentation_gaps'])} documentation gaps"
            )

        # Dependency recommendations
        dependency_analysis = self.validation_report["dependency_analysis"]
        if dependency_analysis.get("circular_dependencies"):
            recommendations.append(
                f"Resolve {len(dependency_analysis['circular_dependencies'])} circular dependencies"
            )

        self.validation_report["optimization_recommendations"] = recommendations

    def _persist_validation_report(self):
        """
        Persist comprehensive validation report
        """
        try:
            report_path = os.path.join(
                self.log_dir,
                f'project_validation_report_{time.strftime("%Y%m%d_%H%M%S")}.json',
            )

            with open(report_path, "w") as f:
                json.dump(self.validation_report, f, indent=2)

            self.logger.info(f"Project validation report persisted: {report_path}")

        except Exception as e:
            self.logger.error(f"Validation report persistence failed: {e}")


def main():
    """
    Main execution for project validation
    """
    try:
        # Initialize project validator
        project_validator = UltraComprehensiveProjectValidator()

        # Perform comprehensive project validation
        validation_report = project_validator.validate_project_structure()

        print("\n🔍 Ultra-Comprehensive Project Validation Results 🔍")

        print("\nOptimization Recommendations:")
        for recommendation in validation_report.get("optimization_recommendations", []):
            print(f"- {recommendation}")

        print("\nDetailed Insights:")
        print(
            f"Total Directories: {validation_report['directory_structure'].get('total_directories', 0)}"
        )
        print(
            f"Total Python Files: {validation_report['code_quality'].get('total_files', 0)}"
        )
        print(f"Security Issues: {len(validation_report.get('security_issues', []))}")
        print(
            f"Documentation Gaps: {len(validation_report.get('documentation_gaps', []))}"
        )

    except Exception as e:
        logging.critical(f"Project validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
