#!/usr/bin/env python3
"""
SutazAI System Architecture Validator and Optimizer

Comprehensive script to:
- Validate system architecture integrity
- Identify potential vulnerabilities
- Optimize system structure
"""

import importlib
import inspect
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List


class SystemArchitectureValidator:
    """
    Advanced system architecture validation framework

    Provides comprehensive analysis and optimization of system structure
    """

    def __init__(self, base_dir: str = "/opt/sutazaiapp"):
        """
        Initialize system architecture validator

        Args:
            base_dir (str): Base directory of the project
        """
        self.base_dir = base_dir
        self.validation_report = {
            "directory_structure": {},
            "module_integrity": {},
            "dependency_analysis": {},
            "optimization_recommendations": [],
        }

    def validate_directory_structure(self) -> Dict[str, Any]:
        """
        Validate the overall directory structure and integrity

        Returns:
            Detailed directory structure validation report
        """
        required_dirs = [
            "ai_agents",
            "backend",
            "core_system",
            "scripts",
            "system_integration",
            "advanced_system_analysis",
            "logs",
            "config",
        ]

        dir_validation = {}

        for req_dir in required_dirs:
            full_path = os.path.join(self.base_dir, req_dir)
            dir_validation[req_dir] = {
                "exists": os.path.exists(full_path),
                "is_dir": (
                    os.path.isdir(full_path) if os.path.exists(full_path) else False
                ),
                "contents": (
                    os.listdir(full_path) if os.path.exists(full_path) else []
                ),
            }

        self.validation_report["directory_structure"] = dir_validation
        return dir_validation

    def validate_module_integrity(self) -> Dict[str, Any]:
        """
        Validate integrity of Python modules across the project

        Returns:
            Detailed module integrity report
        """
        module_integrity = {}

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    file_path = os.path.join(root, file)
                    try:
                        # Attempt to import and inspect the module
                        module_name = os.path.relpath(file_path, self.base_dir).replace(
                            "/", "."
                        )[:-3]
                        module = importlib.import_module(module_name)

                        module_integrity[module_name] = {
                            "path": file_path,
                            "classes": [
                                name
                                for name, obj in inspect.getmembers(
                                    module, inspect.isclass
                                )
                                if obj.__module__ == module_name
                            ],
                            "functions": [
                                name
                                for name, obj in inspect.getmembers(
                                    module, inspect.isfunction
                                )
                                if obj.__module__ == module_name
                            ],
                        }
                    except Exception as e:
                        module_integrity[module_name] = {"import_error": str(e)}

        self.validation_report["module_integrity"] = module_integrity
        return module_integrity

    def analyze_dependencies(self) -> Dict[str, Any]:
        """
        Analyze project dependencies and their relationships

        Returns:
            Detailed dependency analysis report
        """
        dependency_analysis = {"requirements": {}, "import_graph": {}}

        # Check requirements.txt
        requirements_path = os.path.join(self.base_dir, "requirements.txt")
        if os.path.exists(requirements_path):
            with open(requirements_path, "r") as f:
                dependencies = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
                dependency_analysis["requirements"] = dependencies

        # Build import graph
        import_graph = {}
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r") as f:
                            content = f.read()
                            imports = [
                                line.split()[-1].strip()
                                for line in content.split("\n")
                                if line.startswith("import ")
                                or line.startswith("from ")
                            ]
                            import_graph[file_path] = imports
                    except Exception:
                        pass

        dependency_analysis["import_graph"] = import_graph

        self.validation_report["dependency_analysis"] = dependency_analysis
        return dependency_analysis

    def analyze_security(self) -> Dict[str, Any]:
        """
        Analyze security aspects of the project

        Returns:
            Detailed security analysis report
        """
        security_analysis = {
            "sensitive_files": [],
            "potential_vulnerabilities": [],
        }

        # Check for potential sensitive files
        sensitive_patterns = [".env", "secret", "key", "token", "credentials"]

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if any(pattern in file.lower() for pattern in sensitive_patterns):
                    security_analysis["sensitive_files"].append(
                        os.path.join(root, file)
                    )

        # Basic vulnerability checks
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r") as f:
                            content = f.read()

                            if "eval(" in content or "exec(" in content:
                                security_analysis["potential_vulnerabilities"].append(
                                    {
                                        "file": file_path,
                                        "issue": "Potential code injection "
                                        "vulnerability",
                                    }
                                )

                            if "subprocess.call(" in content or "os.system(" in content:
                                security_analysis["potential_vulnerabilities"].append(
                                    {
                                        "file": file_path,
                                        "issue": "Potential shell injection "
                                        "vulnerability",
                                    }
                                )
                    except Exception:
                        pass

        self.validation_report["security_analysis"] = security_analysis
        return security_analysis

    def generate_optimization_recommendations(self) -> List[str]:
        """
        Generate system optimization recommendations

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Directory structure recommendations
        dir_structure = self.validation_report["directory_structure"]
        missing_dirs = [
            dir_name
            for dir_name, details in dir_structure.items()
            if not details["exists"]
        ]
        if missing_dirs:
            recommendations.append(
                f"Create missing directories: {', '.join(missing_dirs)}"
            )

        # Module integrity recommendations
        module_integrity = self.validation_report["module_integrity"]
        import_errors = [
            module
            for module, details in module_integrity.items()
            if "import_error" in details
        ]
        if import_errors:
            recommendations.append(
                f"Fix import errors in modules: {', '.join(import_errors)}"
            )

        # Dependency recommendations
        dependencies = self.validation_report["dependency_analysis"]
        if not dependencies["requirements"]:
            recommendations.append("Update requirements.txt with project dependencies")

        # Security recommendations
        security = self.validation_report["security_analysis"]
        if security["sensitive_files"]:
            recommendations.append("Review and secure sensitive files")

        self.validation_report["optimization_recommendations"] = recommendations
        return recommendations

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive system architecture report

        Returns:
            Detailed system architecture report
        """
        # Run all validation checks
        self.validate_directory_structure()
        self.validate_module_integrity()
        self.analyze_dependencies()
        self.analyze_security()
        self.generate_optimization_recommendations()

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            self.base_dir,
            f"logs/system_architecture_report_{timestamp}.json",
        )

        with open(report_path, "w") as f:
            json.dump(self.validation_report, f, indent=2)

        return self.validation_report


def main():
    """
    Main execution for system architecture validation
    """
    try:
        validator = SystemArchitectureValidator()
        report = validator.generate_comprehensive_report()

        print("System Architecture Validation Report:")
        print("Optimization Recommendations:")
        for recommendation in report["optimization_recommendations"]:
            print(f"- {recommendation}")

    except Exception as e:
        print(f"System architecture validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
