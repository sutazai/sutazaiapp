#!/usr/bin/env python3
"""
SutazAI Autonomous Documentation Management System

Comprehensive framework for:
- Automatic documentation generation
- Self-referencing documentation
- Cross-component tracking
- Intelligent documentation updates
- Semantic analysis and documentation
"""

import ast
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    filename="/opt/SutazAI/logs/autonomous_documentation.log",
)
logger = logging.getLogger("SutazAI.AutonomousDocumentationManager")


class AutonomousDocumentationManager:
    """
    Advanced autonomous documentation management system

    Provides comprehensive capabilities for:
    - Automatic documentation discovery
    - Self-referencing documentation generation
    - Intelligent cross-component tracking
    - Semantic documentation analysis
    """

    def __init__(
        self,
        base_dir: str = "/opt/SutazAI",
        docs_dir: str = "/opt/SutazAI/docs",
    ):
        """
        Initialize autonomous documentation manager

        Args:
            base_dir (str): Base project directory
            docs_dir (str): Documentation output directory
        """
        self.base_dir = base_dir
        self.docs_dir = docs_dir

        # Ensure documentation directory exists
        os.makedirs(self.docs_dir, exist_ok=True)

        # Documentation tracking
        self.documentation_index: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "dependencies": {},
            "documentation_stats": {},
        }

    def discover_components(self) -> Dict[str, List[str]]:
        """
        Discover and categorize project components

        Returns:
            Dictionary of components by category
        """
        components = {
            "ai_agents": [],
            "core_system": [],
            "backend": [],
            "security": [],
            "scripts": [],
            "system_integration": [],
        }

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)

                    # Categorize components
                    if "ai_agents" in full_path:
                        components["ai_agents"].append(full_path)
                    elif "core_system" in full_path:
                        components["core_system"].append(full_path)
                    elif "backend" in full_path:
                        components["backend"].append(full_path)
                    elif "security" in full_path:
                        components["security"].append(full_path)
                    elif "scripts" in full_path:
                        components["scripts"].append(full_path)
                    elif "system_integration" in full_path:
                        components["system_integration"].append(full_path)

        return components

    def build_dependency_graph(
        self, components: Dict[str, List[str]]
    ) -> nx.DiGraph:
        """
        Build a comprehensive dependency graph

        Args:
            components (Dict): Discovered project components

        Returns:
            NetworkX Directed Graph of component dependencies
        """
        dependency_graph = nx.DiGraph()

        for category, files in components.items():
            for file_path in files:
                try:
                    with open(file_path, "r") as f:
                        tree = ast.parse(f.read())

                    # Track imports
                    imports = set()
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for n in node.names:
                                imports.add(n.name.split(".")[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.add(node.module.split(".")[0])

                    # Add nodes and edges
                    module_name = os.path.splitext(
                        os.path.basename(file_path)
                    )[0]
                    dependency_graph.add_node(module_name, category=category)

                    for imp in imports:
                        dependency_graph.add_edge(module_name, imp)

                except Exception as e:
                    logger.warning(
                        f"Dependency analysis failed for {file_path}: {e}"
                    )

        return dependency_graph

    def generate_component_documentation(
        self, dependency_graph: nx.DiGraph
    ) -> Dict[str, str]:
        """
        Generate documentation for each component

        Args:
            dependency_graph (nx.DiGraph): Project dependency graph

        Returns:
            Dictionary of component documentation
        """
        component_docs = {}

        for node in dependency_graph.nodes():
            try:
                file_path = os.path.join(self.base_dir, f"{node}.py")

                with open(file_path, "r") as f:
                    content = f.read()
                    tree = ast.parse(content)

                # Extract docstrings and metadata
                module_doc = (
                    ast.get_docstring(tree)
                    or "No module documentation available."
                )

                # Extract classes and functions
                classes = [
                    {
                        "name": node.name,
                        "docstring": ast.get_docstring(node)
                        or "No class documentation available.",
                    }
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef)
                ]

                functions = [
                    {
                        "name": node.name,
                        "docstring": ast.get_docstring(node)
                        or "No function documentation available.",
                    }
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                ]

                # Generate markdown documentation
                markdown_doc = f"# {node} Component Documentation\n\n"
                markdown_doc += f"## Module Overview\n{module_doc}\n\n"

                markdown_doc += "## Classes\n"
                for cls in classes:
                    markdown_doc += (
                        f"### {cls['name']}\n{cls['docstring']}\n\n"
                    )

                markdown_doc += "## Functions\n"
                for func in functions:
                    markdown_doc += (
                        f"### {func['name']}\n{func['docstring']}\n\n"
                    )

                # Add dependency information
                markdown_doc += "## Dependencies\n"
                markdown_doc += "### Incoming Dependencies\n"
                markdown_doc += (
                    "\n".join(
                        f"- {predecessor}"
                        for predecessor in dependency_graph.predecessors(node)
                    )
                    + "\n\n"
                )

                markdown_doc += "### Outgoing Dependencies\n"
                markdown_doc += (
                    "\n".join(
                        f"- {successor}"
                        for successor in dependency_graph.successors(node)
                    )
                    + "\n"
                )

                component_docs[node] = markdown_doc

            except Exception as e:
                logger.warning(
                    f"Documentation generation failed for {node}: {e}"
                )

        return component_docs

    def update_readme(
        self, components: Dict[str, List[str]], dependency_graph: nx.DiGraph
    ):
        """
        Update project README with comprehensive information

        Args:
            components (Dict): Discovered project components
            dependency_graph (nx.DiGraph): Project dependency graph
        """
        readme_content = f"""# SutazAI: Autonomous AI Development Platform

## 🚀 Project Overview

SutazAI is an advanced, self-improving AI development platform designed to push the boundaries of artificial intelligence through comprehensive, secure, and autonomous systems.

## 📊 Project Statistics

- **Total Components**: {sum(len(files) for files in components.values())}
- **Total Dependencies**: {dependency_graph.number_of_edges()}
- **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🏗️ Project Structure

"""

        # Add component breakdown
        for category, files in components.items():
            readme_content += (
                f"### {category.replace('_', ' ').title()} Components\n"
            )
            readme_content += f"- Total Components: {len(files)}\n"
            readme_content += "- Components:\n"
            for file_path in files:
                readme_content += (
                    f"  - {os.path.splitext(os.path.basename(file_path))[0]}\n"
                )
            readme_content += "\n"

        # Add dependency graph visualization
        readme_content += """
## 🔗 Dependency Graph

```mermaid
graph TD
"""
        for node in dependency_graph.nodes():
            readme_content += f"    {node}\n"

        for edge in dependency_graph.edges():
            readme_content += f"    {edge[0]} --> {edge[1]}\n"

        readme_content += "```\n"

        # Write README
        with open(os.path.join(self.base_dir, "README.md"), "w") as f:
            f.write(readme_content)

        logger.info("README updated successfully")

    def run_documentation_management(self):
        """
        Execute comprehensive documentation management workflow
        """
        try:
            # Discover project components
            components = self.discover_components()

            # Build dependency graph
            dependency_graph = self.build_dependency_graph(components)

            # Generate component documentation
            component_docs = self.generate_component_documentation(
                dependency_graph
            )

            # Update README
            self.update_readme(components, dependency_graph)

            # Persist component documentation
            for component, doc in component_docs.items():
                doc_path = os.path.join(
                    self.docs_dir, f"{component}_documentation.md"
                )
                with open(doc_path, "w") as f:
                    f.write(doc)

            logger.info(
                "Autonomous documentation management completed successfully"
            )

        except Exception as e:
            logger.error(f"Documentation management failed: {e}")


def main():
    """
    Main execution for autonomous documentation management
    """
    try:
        doc_manager = AutonomousDocumentationManager()
        doc_manager.run_documentation_management()

    except Exception as e:
        print(f"Autonomous documentation management failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
