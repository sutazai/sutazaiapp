#!/usr/bin/env python3
"""
SutazAI Project Initializer

Comprehensive script to set up the entire project structure,
create necessary directories, and perform initial configuration.
"""

import logging
import os
import subprocess
import sys

import yaml


class ProjectInitializer:
    def __init__(self, project_root: str):
        """
        Initialize the project initializer.

        Args:
            project_root (str): Root directory of the SutazAI project
        """
        self.project_root = os.path.abspath(project_root)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(project_root, "logs", "project_init.log")
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("SutazAI.ProjectInitializer")

    def initialize_project_structure(self):
        """
        Create the comprehensive project directory structure.
        """
        self.logger.info("Initializing Project Structure")

        # Define the complete project structure
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
            "model_management_subdirs": [
                "gpt4all",
                "deepseek",
                "llama2",
                "molmo",
            ],
            "packages_subdirs": ["wheels", "node"],
                "authentication",
                "encryption",
                "access_control",
            ],
        }

        # Create root directories
        for directory in project_structure["root_dirs"]:
            dir_path = os.path.join(self.project_root, directory)
            os.makedirs(dir_path, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")

        # Create subdirectories
        for parent_dir, subdirs in project_structure.items():
            if parent_dir.endswith("_subdirs"):
                parent = parent_dir.replace("_subdirs", "")
                for subdir in subdirs:
                    subdir_path = os.path.join(
                        self.project_root, parent, subdir
                    )
                    os.makedirs(subdir_path, exist_ok=True)
                    self.logger.info(f"Created subdirectory: {subdir_path}")

        # Create logs directory and initial log files
        log_dir = os.path.join(self.project_root, "logs")
        log_files = [
            "system_audit.log",
            "project_init.log",
            "deploy.log",
            "online_calls.log",
        ]
        for log_file in log_files:
            open(os.path.join(log_dir, log_file), "a").close()

        self.logger.info("Project Structure Initialization Completed")

    def setup_virtual_environment(self):
        """
        Create and set up a Python virtual environment.
        """
        self.logger.info("Setting Up Virtual Environment")

        venv_path = os.path.join(self.project_root, "venv")

        try:
            # Create virtual environment
            subprocess.run(
                [sys.executable, "-m", "venv", venv_path], check=True
            )

            # Activate virtual environment and upgrade pip
            pip_path = os.path.join(venv_path, "bin", "pip")
            subprocess.run(
                [pip_path, "install", "--upgrade", "pip"], check=True
            )

            self.logger.info("Virtual Environment Created Successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Virtual Environment Setup Failed: {e}")
            raise

    def install_dependencies(self):
        """
        Install project dependencies from requirements.txt.
        """
        self.logger.info("Installing Project Dependencies")

        requirements_path = os.path.join(self.project_root, "requirements.txt")
        pip_path = os.path.join(self.project_root, "venv", "bin", "pip")

        try:
            # Install dependencies
            subprocess.run(
                [pip_path, "install", "-r", requirements_path], check=True
            )

            self.logger.info("Dependencies Installed Successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Dependency Installation Failed: {e}")
            raise

    def create_configuration_files(self):
        """
        Create initial configuration files for the project.
        """
        self.logger.info("Creating Configuration Files")

        # Configuration templates
        config_templates = {
            "project_config.yaml": {
                "project": {"name": "SutazAI", "version": "2.0.0"},
                "system_configuration": {
                    "python_version": "3.11",
                    "environment": "development",
                },
            },
            ".env.example": (
                "# SutazAI Environment Configuration\n"
                "SECRET_KEY=your-secret-key-here\n"
                "API_ENV=development\n"
                "# Add more environment variables as needed"
            ),
        }

        config_dir = os.path.join(self.project_root, "config")

        for filename, content in config_templates.items():
            file_path = os.path.join(config_dir, filename)

            if filename.endswith(".yaml"):
                with open(file_path, "w") as f:
                    yaml.dump(content, f, default_flow_style=False)
            else:
                with open(file_path, "w") as f:
                    f.write(content)

            self.logger.info(f"Created configuration file: {file_path}")

    def initialize_git_repository(self):
        """
        Initialize a Git repository and create a .gitignore file.
        """
        self.logger.info("Initializing Git Repository")

        try:
            # Initialize Git repository
            subprocess.run(["git", "init"], cwd=self.project_root, check=True)

            # Create .gitignore
            gitignore_path = os.path.join(self.project_root, ".gitignore")
            gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
venv/
*.log

# Node.js
node_modules/
*.npm
*.tgz

# IDE
.vscode/
.idea/

# Misc
.DS_Store
"""
            with open(gitignore_path, "w") as f:
                f.write(gitignore_content)

            self.logger.info("Git Repository Initialized")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git Repository Initialization Failed: {e}")

    def run_initialization(self):
        """
        Execute the complete project initialization process.
        """
        try:
            self.initialize_project_structure()
            self.setup_virtual_environment()
            self.install_dependencies()
            self.create_configuration_files()
            self.initialize_git_repository()

            self.logger.info(
                "SutazAI Project Initialization Completed Successfully"
            )
        except Exception as e:
            self.logger.critical(f"Project Initialization Failed: {e}")
            raise


def main():
    # Determine project root (one directory up from the script)
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    initializer = ProjectInitializer(project_root)
    initializer.run_initialization()


if __name__ == "__main__":
    main()
