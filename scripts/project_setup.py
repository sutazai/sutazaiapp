#!/usr/bin/env python3
"""
SutazAI Comprehensive Project Setup and Initialization Script

This script provides a robust, multi-stage initialization process
for the entire SutazAI project ecosystem.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List

import yaml

from ultimate_system_audit import SystemAuditManager as UltimateSystemAuditor


class ProjectSetup:
    def __init__(self, project_root: str):
        """
        Initialize the project setup process.

        Args:
            project_root (str): Root directory of the project
        """
        self.project_root = os.path.abspath(project_root)
        self._configure_logging()

    def _configure_logging(self):
        """
        Set up comprehensive logging for the setup process.
        """
        logs_dir = os.path.join(self.project_root, "logs", "setup")
        os.makedirs(logs_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)8s | %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(logs_dir, "project_setup.log")),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("SutazAI.ProjectSetup")

    def validate_system_requirements(self):
        """
        Validate system requirements before setup.
        """
        self.logger.info("Validating System Requirements")

        # Python version check
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        required_version = "3.8"

        if sys.version_info.major < 3 or sys.version_info.minor < 8:
            self.logger.critical(
                f"Unsupported Python version. Required: {required_version}+, Current: {python_version}"
            )
            raise RuntimeError(f"Python {required_version}+ is required")

        # Check required system tools
        required_tools = [
            "git",
            "pip",
            "virtualenv",
            "semgrep",
            "safety",
            "pylint",
            "black",
            "mypy",
        ]

        for tool in required_tools:
            try:
                subprocess.run(
                    [tool, "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                self.logger.info(f"{tool.capitalize()} is installed")
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.logger.warning(f"{tool.capitalize()} is not installed")

    def create_project_structure(self):
        """
        Create comprehensive project directory structure.
        """
        self.logger.info("Creating Project Directory Structure")

        project_structure = {
            "root_dirs": [
                "ai_agents",
                "backend",
                "web_ui",
                "scripts",
                "model_management",
                "packages",
                "logs",
                "doc_data",
                "security",
                "infrastructure",
                "advanced_system_tools",
            ],
            "ai_agents_subdirs": [
                "supreme_ai",
                "auto_gpt",
                "superagi",
                "langchain_agents",
                "tabbyml",
                "semgrep",
                "gpt_engineer",
                "aider",
            ],
            "backend_subdirs": ["services", "config", "tests", "middleware"],
            "web_ui_subdirs": ["src", "public", "components", "styles"],
            "model_management_subdirs": ["gpt4all", "deepseek", "llama2", "molmo"],
            "packages_subdirs": ["wheels", "node"],
            "security_subdirs": ["authentication", "encryption", "access_control"],
        }

        for category, directories in project_structure.items():
            if category.endswith("_subdirs"):
                parent_dir = category.replace("_subdirs", "")
                for subdir in directories:
                    full_path = os.path.join(self.project_root, parent_dir, subdir)
                    os.makedirs(full_path, exist_ok=True)
                    self.logger.info(f"Created directory: {full_path}")

    def setup_virtual_environment(self):
        """
        Create and set up a Python virtual environment.
        """
        self.logger.info("Setting Up Virtual Environment")

        venv_path = os.path.join(self.project_root, "venv")

        try:
            # Create virtual environment
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)

            # Activate virtual environment and upgrade pip
            pip_path = os.path.join(venv_path, "bin", "pip")
            subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)

            # Install requirements
            requirements_path = os.path.join(self.project_root, "requirements.txt")
            subprocess.run([pip_path, "install", "-r", requirements_path], check=True)

            self.logger.info("Virtual Environment Created and Dependencies Installed")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Virtual Environment Setup Failed: {e}")
            raise

    def create_configuration_files(self):
        """
        Create initial configuration files for the project.
        """
        self.logger.info("Creating Configuration Files")

        config_dir = os.path.join(self.project_root, "config")
        os.makedirs(config_dir, exist_ok=True)

        # Project Configuration
        project_config = {
            "project": {"name": "SutazAI", "version": "20.1.0"},
            "system_configuration": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "environment": "development",
            },
            "security": {
                "authentication": {"method": "otp", "root_user": "Florin Cristian Suta"}
            },
        }

        # Write project configuration
        with open(os.path.join(config_dir, "project_config.yaml"), "w") as f:
            yaml.dump(project_config, f, default_flow_style=False)

        # Create .env.example
        env_example = (
            "# SutazAI Environment Configuration\n"
            "SECRET_KEY=your-secret-key-here\n"
            "DEPLOYMENT_ENV=development\n"
            "ROOT_USER_EMAIL=chrissuta01@gmail.com\n"
            "ROOT_USER_PHONE=+48517716005\n"
        )

        with open(os.path.join(config_dir, ".env.example"), "w") as f:
            f.write(env_example)

        self.logger.info("Configuration files created successfully")

    def initialize_git_repository(self):
        """
        Initialize Git repository and create .gitignore.
        """
        self.logger.info("Initializing Git Repository")

        try:
            # Initialize Git repository
            subprocess.run(["git", "init"], cwd=self.project_root, check=True)

            # Create comprehensive .gitignore
            gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
venv/
*.log
*.json

# AI Models
*.bin
*.pt
*.pth

# Node.js
node_modules/
*.npm
*.tgz
build/
dist/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Sensitive Data
.env
*.key
*.pem

# Logs and Databases
logs/
*.sqlite
*.db

# Misc
.pytest_cache/
.coverage
htmlcov/
"""

            with open(os.path.join(self.project_root, ".gitignore"), "w") as f:
                f.write(gitignore_content)

            self.logger.info("Git repository initialized with comprehensive .gitignore")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git repository initialization failed: {e}")

    def run_initial_audit(self):
        """
        Run the initial system audit after setup.
        """
        self.logger.info("Running Initial System Audit")

        try:
            auditor = UltimateSystemAuditor()
            audit_results = auditor.audit()

            # Save audit results
            audit_log_dir = os.path.join(self.project_root, "logs", "audits")
            os.makedirs(audit_log_dir, exist_ok=True)

            with open(
                os.path.join(audit_log_dir, "initial_setup_audit.json"), "w"
            ) as f:
                json.dump(audit_results, f, indent=4)

            self.logger.info("Initial system audit completed successfully")
        except Exception as e:
            self.logger.error(f"Initial system audit failed: {e}")

    def run_setup(self):
        """
        Execute the complete project setup process.
        """
        try:
            self.validate_system_requirements()
            self.create_project_structure()
            self.setup_virtual_environment()
            self.create_configuration_files()
            self.initialize_git_repository()
            self.run_initial_audit()

            self.logger.info("🎉 SutazAI Project Setup Completed Successfully! 🎉")
        except Exception as e:
            self.logger.critical(f"Project Setup Failed: {e}")
            raise


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    setup = ProjectSetup(project_root)
    setup.run_setup()


if __name__ == "__main__":
    main()
