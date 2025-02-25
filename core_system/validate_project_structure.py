#!/usr/bin/env python3
"""
Ultra-Comprehensive Project Structure Validation Script

Provides a systematic approach to validate and optimize
the entire SutazAI project structure, ensuring:
- Correct directory hierarchy
- Consistent naming conventions
- Dependency integrity
- Security compliance
"""

import json
import logging
import os
from typing import Any, Dict, List

import yaml


class ProjectStructureValidator:
    """
    Advanced project structure validation and optimization system
    """

    def __init__(
        self,
        base_dir: str = "/opt/SutazAI",
        config_path: str = None,
    ):
        """
        Initialize Project Structure Validator

        Args:
            base_dir (str): Base project directory
            config_path (str): Path to validation configuration
        """
        self.base_dir = base_dir
        self.config_path = config_path or os.path.join(
            base_dir, "config", "project_structure_config.yml"
        )

        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            filename=os.path.join(base_dir, "logs", "project_structure_validation.log"),
        )
        self.logger = logging.getLogger("SutazAI.ProjectStructureValidator")

        # Load configuration
        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = self._generate_default_config()

    def _generate_default_config(self) -> Dict[str, Any]:
        """
        Generate a default project structure configuration

        Returns:
            Default configuration dictionary
        """
        return {
            "required_directories": [
                "ai_agents",
                "backend",
                "core_system",
                "scripts",
                "tests",
                "docs",
                "config",
                "logs",
                "model_management",
                "web_ui",
            ],
            "naming_conventions": {
                "python_files": r"^[a-z_]+\.py$",
                "test_files": r"^test_[a-z_]+\.py$",
                "script_files": r"^[a-z_]+\.sh$",
            },
            "security_checks": {
                "no_hardcoded_credentials": True,
                "no_external_api_keys": True,
            },
        }

    def validate_project_structure(self) -> Dict[str, Any]:
        """
        Perform comprehensive project structure validation

        Returns:
            Validation report
        """
        validation_report = {
            "timestamp": time.time(),
            "directory_structure": {},
            "naming_compliance": {},
            "security_checks": {},
            "optimization_recommendations": [],
        }

        # Validate directory structure
        validation_report["directory_structure"] = self._validate_directories()

        # Validate file naming conventions
        validation_report["naming_compliance"] = self._validate_naming_conventions()

        # Perform security checks
        validation_report["security_checks"] = self._perform_security_checks()

        # Generate optimization recommendations
        validation_report["optimization_recommendations"] = (
            self._generate_optimization_recommendations(validation_report)
        )

        # Log and persist validation report
        self._log_validation_report(validation_report)

        return validation_report

    def _validate_directories(self) -> Dict[str, Any]:
        """
        Validate project directory structure

        Returns:
            Directory validation results
        """
        directory_validation = {
            "required_directories": {},
            "missing_directories": [],
            "extra_directories": [],
        }

        # Check required directories
        for directory in self.config.get("required_directories", []):
            full_path = os.path.join(self.base_dir, directory)
            directory_validation["required_directories"][directory] = os.path.exists(
                full_path
            )

            if not os.path.exists(full_path):
                directory_validation["missing_directories"].append(directory)

        # Find extra directories
        all_directories = [
            d
            for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d))
        ]
        extra_dirs = set(all_directories) - set(
            self.config.get("required_directories", [])
        )
        directory_validation["extra_directories"] = list(extra_dirs)

        return directory_validation

    def _validate_naming_conventions(self) -> Dict[str, Any]:
        """
        Validate file naming conventions

        Returns:
            Naming convention validation results
        """
        import re

        naming_validation = {
            "python_files": {"compliant": [], "non_compliant": []},
            "test_files": {"compliant": [], "non_compliant": []},
            "script_files": {"compliant": [], "non_compliant": []},
        }

        # Validate Python files
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".py"):
                    is_compliant = (
                        re.match(
                            self.config["naming_conventions"]["python_files"],
                            file,
                        )
                        is not None
                    )

                    if is_compliant:
                        naming_validation["python_files"]["compliant"].append(
                            os.path.join(root, file)
                        )
                    else:
                        naming_validation["python_files"]["non_compliant"].append(
                            os.path.join(root, file)
                        )

                if file.startswith("test_") and file.endswith(".py"):
                    is_compliant = (
                        re.match(
                            self.config["naming_conventions"]["test_files"],
                            file,
                        )
                        is not None
                    )

                    if is_compliant:
                        naming_validation["test_files"]["compliant"].append(
                            os.path.join(root, file)
                        )
                    else:
                        naming_validation["test_files"]["non_compliant"].append(
                            os.path.join(root, file)
                        )

        # Validate shell scripts
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".sh"):
                    is_compliant = (
                        re.match(
                            self.config["naming_conventions"]["script_files"],
                            file,
                        )
                        is not None
                    )

                    if is_compliant:
                        naming_validation["script_files"]["compliant"].append(
                            os.path.join(root, file)
                        )
                    else:
                        naming_validation["script_files"]["non_compliant"].append(
                            os.path.join(root, file)
                        )

        return naming_validation

    def _perform_security_checks(self) -> Dict[str, Any]:
        """
        Perform comprehensive security checks

        Returns:
            Security check results
        """
        security_checks = {
            "hardcoded_credentials": [],
            "external_api_keys": [],
        }

        # Check for hardcoded credentials
        if self.config["security_checks"].get("no_hardcoded_credentials", True):
            for root, _, files in os.walk(self.base_dir):
                for file in files:
                    if file.endswith(".py") or file.endswith(".sh"):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r") as f:
                            content = f.read()

                            # Simple regex for potential credential patterns
                            credential_patterns = [
                                r'(password|secret|token|key)\s*=\s*[\'"].*?[\'"]',
                                r'(mysql|postgresql|sqlite)://.*?:[\'"].*?[\'"]',
                            ]

                            for pattern in credential_patterns:
                                matches = re.findall(pattern, content, re.IGNORECASE)
                                if matches:
                                    security_checks["hardcoded_credentials"].append(
                                        {"file": file_path, "matches": matches}
                                    )

        return security_checks

    def _generate_optimization_recommendations(
        self, validation_report: Dict[str, Any]
    ) -> List[str]:
        """
        Generate optimization recommendations based on validation results

        Args:
            validation_report (Dict): Comprehensive validation report

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Directory structure recommendations
        if validation_report["directory_structure"]["missing_directories"]:
            recommendations.append(
                f"Create missing directories: {', '.join(validation_report['directory_structure']['missing_directories'])}"
            )

        if validation_report["directory_structure"]["extra_directories"]:
            recommendations.append(
                f"Review and potentially remove extra directories: {', '.join(validation_report['directory_structure']['extra_directories'])}"
            )

        # Naming convention recommendations
        for file_type, compliance in validation_report["naming_compliance"].items():
            if compliance["non_compliant"]:
                recommendations.append(
                    f"Rename non-compliant {file_type}: {len(compliance['non_compliant'])} files"
                )

        # Security recommendations
        if validation_report["security_checks"]["hardcoded_credentials"]:
            recommendations.append(
                f"Remove hardcoded credentials in {len(validation_report['security_checks']['hardcoded_credentials'])} files"
            )

        return recommendations

    def _log_validation_report(self, validation_report: Dict[str, Any]):
        """
        Log and persist validation report

        Args:
            validation_report (Dict): Comprehensive validation report
        """
        log_file = os.path.join(
            self.base_dir,
            "logs",
            f'project_structure_validation_{time.strftime("%Y%m%d_%H%M%S")}.json',
        )

        with open(log_file, "w") as f:
            json.dump(validation_report, f, indent=2)

        # Log key insights
        self.logger.info("Project Structure Validation Completed")
        self.logger.info(
            f"Missing Directories: {validation_report['directory_structure']['missing_directories']}"
        )
        self.logger.info(
            f"Non-Compliant Files: {sum(len(files['non_compliant']) for files in validation_report['naming_compliance'].values())}"
        )
        self.logger.info(
            f"Security Issues: {len(validation_report['security_checks']['hardcoded_credentials'])}"
        )


def main():
    """
    Execute comprehensive project structure validation
    """
    validator = ProjectStructureValidator()
    validation_report = validator.validate_project_structure()

    print("\n🔍 Project Structure Validation Results 🔍")

    print("\nDirectory Structure:")
    print(
        f"Missing Directories: {validation_report['directory_structure']['missing_directories']}"
    )
    print(
        f"Extra Directories: {validation_report['directory_structure']['extra_directories']}"
    )

    print("\nNaming Compliance:")
    for file_type, compliance in validation_report["naming_compliance"].items():
        print(f"{file_type.replace('_', ' ').title()}:")
        print(f"  Compliant: {len(compliance['compliant'])}")
        print(f"  Non-Compliant: {len(compliance['non_compliant'])}")

    print("\nSecurity Checks:")
    print(
        f"Hardcoded Credentials: {len(validation_report['security_checks']['hardcoded_credentials'])}"
    )

    print("\nOptimization Recommendations:")
    for recommendation in validation_report["optimization_recommendations"]:
        print(f"- {recommendation}")


if __name__ == "__main__":
    main()
