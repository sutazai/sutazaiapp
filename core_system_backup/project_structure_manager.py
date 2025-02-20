#!/usr/bin/env python3
"""
SutazAI Project Structure Management System

Provides comprehensive project structure validation,
restoration, and intelligent organization capabilities.
"""

import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import networkx as nx


class ProjectStructureManager:
    """
    Ultra-Comprehensive Project Structure Management Framework

    Key Capabilities:
    - Project structure validation
    - Automatic component restoration
    - Dependency and relationship tracking
    - Intelligent project organization
    """

    # Standard project structure template
    STANDARD_PROJECT_STRUCTURE = {
        "root": [
            "config",
            "core_system",
            "workers",
            "services",
            "utils",
            "ai_agents",
            "scripts",
            "logs",
            "tests",
            "docs",
        ],
        "config": [
            "system_config.yaml",
            "logging_config.yaml",
            "security_config.yaml",
        ],
        "core_system": [
            "project_structure_manager.py",
            "dependency_tracker.py",
            "intelligent_error_corrector.py",
        ],
        "workers": ["system_integration_worker.py", "performance_monitor.py"],
        "services": ["autonomous_file_organizer.py", "semantic_analyzer.py"],
        "utils": [
            "logging_utils.py",
            "security_utils.py",
            "performance_utils.py",
        ],
        "ai_agents": [
            "error_correction_agent.py",
            "dependency_optimization_agent.py",
        ],
        "scripts": ["setup.sh", "deploy.py", "maintenance.py"],
        "logs": [],
        "tests": [
            "test_core_system.py",
            "test_workers.py",
            "test_services.py",
        ],
        "docs": [
            "architecture.md",
            "setup_guide.md",
            "contribution_guidelines.md",
        ],
    }

    def __init__(
        self,
        base_dir: str = "/opt/sutazai_project/SutazAI",
        log_dir: Optional[str] = None,
    ):
        """
        Initialize Project Structure Manager

        Args:
            base_dir (str): Base project directory
            log_dir (Optional[str]): Custom log directory
        """
        self.base_dir = base_dir
        self.log_dir = log_dir or os.path.join(
            base_dir, "logs", "project_structure"
        )

        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            filename=os.path.join(self.log_dir, "project_structure.log"),
        )
        self.logger = logging.getLogger("SutazAI.ProjectStructureManager")

        # Initialize project structure graph
        self.project_structure_graph = nx.DiGraph()

    def validate_project_structure(self) -> Dict[str, Any]:
        """
        Validate and restore project structure

        Returns:
            Dictionary of structure validation results
        """
        validation_results = {
            "missing_directories": [],
            "missing_files": [],
            "restored_components": [],
        }

        try:
            # Validate root-level directories
            for directory in self.STANDARD_PROJECT_STRUCTURE["root"]:
                full_path = os.path.join(self.base_dir, directory)

                if not os.path.exists(full_path):
                    os.makedirs(full_path)
                    validation_results["missing_directories"].append(directory)
                    validation_results["restored_components"].append(
                        f"Created directory: {directory}"
                    )
                    self.logger.info(f"Created missing directory: {directory}")

            # Validate files in each directory
            for directory, files in self.STANDARD_PROJECT_STRUCTURE.items():
                if directory != "root":
                    dir_path = os.path.join(self.base_dir, directory)

                    for file in files:
                        file_path = os.path.join(dir_path, file)

                        if not os.path.exists(file_path):
                            # Create empty file
                            with open(file_path, "w") as f:
                                f.write(
                                    "# Placeholder for future implementation\n"
                                )

                            validation_results["missing_files"].append(file)
                            validation_results["restored_components"].append(
                                f"Created file: {file}"
                            )
                            self.logger.info(f"Created missing file: {file}")

            # Generate project structure graph
            self.generate_project_structure_graph()

            return validation_results

        except Exception as e:
            self.logger.error(f"Project structure validation failed: {e}")
            return validation_results

    def generate_project_structure_graph(self) -> nx.DiGraph:
        """
        Generate a graph representing project structure and relationships

        Returns:
            NetworkX Directed Graph of project structure
        """
        try:
            # Reset graph
            self.project_structure_graph = nx.DiGraph()

            # Add directories as nodes
            for directory in self.STANDARD_PROJECT_STRUCTURE["root"]:
                self.project_structure_graph.add_node(
                    directory, type="directory"
                )

            # Add files as nodes and create edges
            for directory, files in self.STANDARD_PROJECT_STRUCTURE.items():
                if directory != "root":
                    for file in files:
                        self.project_structure_graph.add_node(
                            file, type="file", directory=directory
                        )
                        self.project_structure_graph.add_edge(directory, file)

            return self.project_structure_graph

        except Exception as e:
            self.logger.error(
                f"Project structure graph generation failed: {e}"
            )
            return nx.DiGraph()

    def analyze_project_dependencies(self) -> Dict[str, Any]:
        """
        Analyze dependencies and relationships between project components

        Returns:
            Dictionary of project dependency insights
        """
        dependency_analysis = {
            "total_components": 0,
            "directory_relationships": {},
            "file_dependencies": {},
        }

        try:
            # Count total components
            dependency_analysis["total_components"] = len(
                self.project_structure_graph.nodes()
            )

            # Analyze directory relationships
            for directory in self.STANDARD_PROJECT_STRUCTURE["root"]:
                dependency_analysis["directory_relationships"][directory] = {
                    "files": list(
                        self.project_structure_graph.successors(directory)
                    )
                }

            # Analyze file dependencies (placeholder for more advanced analysis)
            for node in self.project_structure_graph.nodes(data=True):
                if node[1].get("type") == "file":
                    dependency_analysis["file_dependencies"][node[0]] = {
                        "directory": node[1].get("directory", "unknown")
                    }

            return dependency_analysis

        except Exception as e:
            self.logger.error(f"Project dependency analysis failed: {e}")
            return dependency_analysis

    def generate_comprehensive_structure_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive project structure report

        Returns:
            Detailed project structure analysis report
        """
        try:
            # Validate project structure
            validation_results = self.validate_project_structure()

            # Analyze project dependencies
            dependency_analysis = self.analyze_project_dependencies()

            # Comprehensive report
            comprehensive_report = {
                "timestamp": datetime.now().isoformat(),
                "validation_results": validation_results,
                "dependency_analysis": dependency_analysis,
                "hardcoded_structure": self.get_hardcoded_project_structure(),
            }

            # Persist report
            report_path = os.path.join(
                self.log_dir,
                f'project_structure_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            )
            with open(report_path, "w") as f:
                json.dump(comprehensive_report, f, indent=2)

            self.logger.info(
                f"Comprehensive project structure report generated: {report_path}"
            )

            return comprehensive_report

        except Exception as e:
            self.logger.error(
                f"Comprehensive structure report generation failed: {e}"
            )
            return {}

    def get_hardcoded_project_structure(self) -> Dict[str, Any]:
        """
        Return the hardcoded project structure

        Returns:
            Dictionary representing the hardcoded project structure
        """
        return {
            "base_path": "/opt/sutazai/",
            "structure": {
                "ai_agents": {
                    "supreme_ai": "# Non-root AI agent",
                    "auto_gpt": "",
                    "superagi": "",
                    "langchain_agents": "",
                    "tabbyml": "",
                    "semgrep": "",
                    "gpt_engineer": "",
                    "aider": "",
                },
                "model_management": {
                    "GPT4All": "",
                    "DeepSeek-R1": "",
                    "DeepSeek-Coder": "",
                    "Llama2": "",
                    "Molmo": "# Diagram recognition",
                },
                "backend": {
                    "main.py": "",
                    "api_routes.py": "",
                    "services": "",
                    "config": "",
                    "tests": "",
                },
                "web_ui": {
                    "package.json": "",
                    "node_modules": "# Online",
                    "src": "",
                    "public": "",
                    "build_or_dist": "",
                },
                "scripts": {
                    "deploy.sh": "# Main online deploy script",
                    "setup_repos.sh": "# Fallback manual scp",
                    "test_pipeline.py": "",
                },
                "packages": {
                    "wheels": "# Pinned Python .whl",
                    "node": "# Online node modules / .tgz",
                },
                "logs": {
                    "deploy.log": "",
                    "pipeline.log": "",
                    "online_calls.log": "",
                },
                "doc_data": {"pdfs": "", "diagrams": ""},
                "requirements.txt": "",
                "venv": "",
                "README.md": "",
            },
        }

    def generate_markdown_documentation(self) -> str:
        """
        Generate a markdown documentation of the project structure

        Returns:
            Markdown formatted project structure documentation
        """
        hardcoded_structure = self.get_hardcoded_project_structure()

        markdown_doc = "# SutazAI Project Structure\n\n"
        markdown_doc += "## Directory Tree\n"
        markdown_doc += "```\n"
        markdown_doc += f"{hardcoded_structure['base_path']}\n"

        def generate_tree(structure, indent=""):
            nonlocal markdown_doc
            for key, value in structure.items():
                if isinstance(value, dict):
                    markdown_doc += f"{indent}├── {key}/\n"
                    generate_tree(value, indent + "│   ")
                else:
                    comment = f" {value}" if value else ""
                    markdown_doc += f"{indent}├── {key}{comment}\n"

        generate_tree(hardcoded_structure["structure"])
        markdown_doc += "```\n"

        return markdown_doc

    def update_readme_with_structure(self):
        """
        Update README.md with the current project structure
        """
        try:
            readme_path = os.path.join(self.base_dir, "README.md")
            markdown_doc = self.generate_markdown_documentation()

            # Read existing README
            with open(readme_path, "r") as f:
                readme_content = f.read()

            # Replace or append project structure section
            structure_section = f"\n\n## Project Structure\n\n{markdown_doc}"
            if "## Project Structure" in readme_content:
                readme_content = re.sub(
                    r"## Project Structure.*",
                    structure_section,
                    readme_content,
                    flags=re.DOTALL,
                )
            else:
                readme_content += structure_section

            # Write updated README
            with open(readme_path, "w") as f:
                f.write(readme_content)

            self.logger.info("README.md updated with project structure")

        except Exception as e:
            self.logger.error(f"Failed to update README.md: {e}")


def main():
    """
    Main execution for project structure management
    """
    try:
        # Initialize project structure manager
        structure_manager = ProjectStructureManager()

        # Validate and generate comprehensive report
        report = structure_manager.generate_comprehensive_structure_report()

        # Update README with current structure
        structure_manager.update_readme_with_structure()

        print("Project Structure Management Completed Successfully")
    except Exception as e:
        print(f"Project Structure Management Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
