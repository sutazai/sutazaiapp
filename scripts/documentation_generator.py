#!/usr/bin/env python3
"""
Documentation Generator for SutazAI Project
"""

import ast
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict

import networkx as nx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def verify_python_version():
    """
    Verify that Python 3.11 or higher is being used.
    """
    major, minor = sys.version_info.major, sys.version_info.minor
    if major < 3 or (major == 3 and minor < 11):
        print("❌ Error: Python 3.11 or higher is required.")
        print(f"Current Python version: {sys.version}")
        print("Please install Python 3.11 and try again.")
        sys.exit(1)
    print(f"✅ Python {major}.{minor} detected.")


class DocumentationGenerator:
    def __init__(self, base_path: str = "/opt/sutazaiapp"):
        """
        Comprehensive Documentation Generation Tool

        Args:
            base_path (str): Base path of the SutazAI project
        """
        self.base_path = base_path
        self.console = Console()

        # Logging setup
        self.log_dir = os.path.join(base_path, "logs", "documentation")
        os.makedirs(self.log_dir, exist_ok=True)

        self.doc_log = os.path.join(
            self.log_dir,
            f"documentation_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(self.doc_log),
                logging.StreamHandler(sys.stdout),
            ],
        )

        # Documentation tracking
        self.documentation = {
            "modules": {},
            "classes": {},
            "functions": {},
            "dependencies": {},
        }

    def generate_module_documentation(self, module_path: str) -> Dict[str, Any]:
        """
        Generate comprehensive documentation for a Python module

        Args:
            module_path (str): Path to the Python module

        Returns:
            Detailed module documentation
        """
        module_doc = {
            "name": os.path.splitext(os.path.basename(module_path))[0],
            "path": module_path,
            "docstring": "",
            "classes": {},
            "functions": {},
            "imports": [],
        }

        try:
            # Read module content
            with open(module_path, encoding="utf-8") as f:
                content = f.read()

            # Parse module AST
            tree = ast.parse(content)

            # Extract module-level docstring
            module_doc["docstring"] = ast.get_docstring(tree) or "No module-level documentation"

            # Extract imports
            module_doc["imports"] = [
                node.names[0].name for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
            ]

            # Analyze classes and functions
            for node in tree.body:
                # Class documentation
                if isinstance(node, ast.ClassDef):
                    class_doc = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "No class documentation",
                        "methods": {},
                    }

                    # Analyze class methods
                    for method_node in node.body:
                        if isinstance(method_node, ast.FunctionDef):
                            class_doc["methods"][method_node.name] = {
                                "docstring": ast.get_docstring(method_node) or "No method documentation",
                                "arguments": [arg.arg for arg in method_node.args.args],
                            }

                    module_doc["classes"][node.name] = class_doc

                # Function documentation
                elif isinstance(node, ast.FunctionDef):
                    module_doc["functions"][node.name] = {
                        "docstring": ast.get_docstring(node) or "No function documentation",
                        "arguments": [arg.arg for arg in node.args.args],
                    }

        except Exception as e:
            logging.exception("Error generating docs: %s", e)

        return module_doc

    def scan_project_modules(self) -> Dict[str, Any]:
        """
        Scan and document all Python modules in the project

        Returns:
            Comprehensive project module documentation
        """
        project_docs = {}

        # Walk through project files
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)

                    # Skip test and init files
                    if not (file.startswith("test_") or file == "__init__.py"):
                        module_doc = self.generate_module_documentation(file_path)
                        project_docs[file_path] = module_doc

        return project_docs

    def generate_dependency_graph(self, project_docs: Dict[str, Any]) -> nx.DiGraph:
        """
        Create a dependency graph based on module imports

        Args:
            project_docs (Dict): Project module documentation

        Returns:
            Dependency graph
        """
        dependency_graph = nx.DiGraph()

        for module_path, module_info in project_docs.items():
            # Add nodes
            dependency_graph.add_node(module_path)

            # Add edges for imports
            for imp in module_info.get("imports", []):
                dependency_graph.add_edge(module_path, imp)

        return dependency_graph

    def generate_markdown_documentation(self, project_docs: Dict[str, Any]) -> str:
        """
        Generate comprehensive Markdown documentation

        Args:
            project_docs (Dict): Project module documentation

        Returns:
            Markdown-formatted documentation
        """
        markdown_doc = "# SutazAI Project Documentation\n\n"
        markdown_doc += f"## Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Project overview
        markdown_doc += "## Project Overview\n\n"
        markdown_doc += "SutazAI is an advanced, autonomous AI development platform.\n\n"

        # Module documentation
        markdown_doc += "## Modules\n\n"
        for module_path, module_info in project_docs.items():
            markdown_doc += f"### {module_info['name']} Module\n\n"
            markdown_doc += f"**Path**: `{module_path}`\n\n"
            markdown_doc += f"**Docstring**: {module_info['docstring']}\n\n"

            # Classes
            if module_info["classes"]:
                markdown_doc += "#### Classes\n\n"
                for class_name, class_info in module_info["classes"].items():
                    markdown_doc += f"##### {class_name}\n\n"
                    markdown_doc += f"**Docstring**: {class_info['docstring']}\n\n"

                    # Methods
                    if class_info["methods"]:
                        markdown_doc += "**Methods**:\n\n"
                        for method_name, method_info in class_info["methods"].items():
                            method_args = ", ".join(method_info["arguments"])
                            method_entry = f"- `{method_name}("
                            method_entry += f"{method_args})`\n"
                            markdown_doc += method_entry
                            markdown_doc += f"  - {method_info['docstring']}\n\n"

            # Functions
            if module_info["functions"]:
                markdown_doc += "#### Functions\n\n"
                for func_name, func_info in module_info["functions"].items():
                    markdown_doc += f"##### {func_name}\n\n"
                    markdown_doc += f"**Docstring**: {func_info['docstring']}\n\n"

        return markdown_doc

    def comprehensive_documentation_generation(self) -> Dict[str, Any]:
        """
        Perform comprehensive documentation generation

        Returns:
            Documentation generation results
        """
        logging.info("Starting Comprehensive Documentation Generation")

        # Scan project modules
        project_docs = self.scan_project_modules()

        # Generate dependency graph
        dependency_graph = self.generate_dependency_graph(project_docs)

        # Generate Markdown documentation
        markdown_doc = self.generate_markdown_documentation(project_docs)

        # Save Markdown documentation
        markdown_path = os.path.join(
            self.base_path,
            "docs",
            f"project_docs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        )
        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)

        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_doc)

        # Visualization
        self._visualize_documentation_results(project_docs, dependency_graph)

        return {
            "timestamp": datetime.now().isoformat(),
            "modules_documented": len(project_docs),
            "markdown_path": markdown_path,
        }

    def _visualize_documentation_results(
        self,
        project_docs: Dict[str, Any],
        dependency_graph: nx.DiGraph,
    ):
        """
        Visualize documentation generation results

        Args:
            project_docs (Dict): Project module documentation
            dependency_graph (nx.DiGraph): Module dependency graph
        """
        self.console.rule("[bold blue]SutazAI Documentation Generation[/bold blue]")

        # Documentation Summary Panel
        total_modules = len(project_docs)
        total_deps = dependency_graph.number_of_edges()
        doc_panel = Panel(
            f"Total Modules Documented: {total_modules}\nTotal Dependency Connections: {total_deps}",
            title="Documentation Summary",
            border_style="green",
        )
        self.console.print(doc_panel)

        # Module Types
        module_types_table = Table(title="Module Types")
        module_types_table.add_column("Type", style="cyan")
        module_types_table.add_column("Count", style="magenta")

        module_types = {
            "Modules with Classes": sum(1 for module in project_docs.values() if module["classes"]),
            "Modules with Functions": sum(1 for module in project_docs.values() if module["functions"]),
        }

        for type_name, count in module_types.items():
            module_types_table.add_row(type_name, str(count))

        self.console.print(module_types_table)


def main():
    # Verify Python version
    verify_python_version()

    doc_generator = DocumentationGenerator()
    doc_generator.comprehensive_documentation_generation()


if __name__ == "__main__":
    main()
