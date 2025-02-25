#!/usr/bin/env python3
"""
SutazAI Comprehensive Project Organizer and Auto-Indexer

Advanced framework for:
- Automatic project structure management
- Intelligent file categorization
- Comprehensive self-indexing
- Dependency tracking
- Structural integrity validation
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    filename="/opt/SutazAI/logs/project_organizer.log",
)
logger = logging.getLogger("SutazAI.ProjectOrganizer")


class ProjectOrganizer:
    """
    Comprehensive project organization and auto-indexing system

    Provides advanced capabilities for:
    - Intelligent file categorization
    - Automatic project structure management
    - Comprehensive self-indexing
    - Structural integrity validation
    """

    # Predefined project structure template
    PROJECT_STRUCTURE = {
        "ai_agents": [
            "supreme_ai",
            "auto_gpt",
            "superagi",
            "langchain_agents",
            "tabbyml",
            "semgrep",
            "gpt_engineer",
            "aider",
        ],
        "model_management": [
            "GPT4All",
            "DeepSeek-R1",
            "DeepSeek-Coder",
            "Llama2",
            "Molmo",
        ],
        "backend": ["services", "config", "tests"],
        "web_ui": ["src", "public", "build"],
        "core_system": [],
        "security": [],
        "system_integration": [],
        "scripts": [],
        "packages": ["wheels", "node"],
        "logs": [],
        "doc_data": ["pdfs", "diagrams"],
        "docs": [],
    }

    def __init__(self, base_dir: str = "/opt/SutazAI"):
        """
        Initialize project organizer

        Args:
            base_dir (str): Base project directory
        """
        self.base_dir = base_dir
        self.project_index: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "structure": {},
            "file_types": {},
            "dependencies": {},
        }

    def validate_project_structure(self) -> bool:
        """
        Validate and ensure project structure integrity

        Returns:
            Boolean indicating structural validity
        """
        for category, subcategories in self.PROJECT_STRUCTURE.items():
            category_path = os.path.join(self.base_dir, category)

            # Ensure category directory exists
            if not os.path.exists(category_path):
                try:
                    os.makedirs(category_path)
                    logger.info(f"Created missing directory: {category_path}")
                except Exception as e:
                    logger.error(f"Failed to create directory {category_path}: {e}")
                    return False

            # Ensure subcategory directories exist
            for subcategory in subcategories:
                subcategory_path = os.path.join(category_path, subcategory)
                if not os.path.exists(subcategory_path):
                    try:
                        os.makedirs(subcategory_path)
                        logger.info(f"Created missing subdirectory: {subcategory_path}")
                    except Exception as e:
                        logger.error(
                            f"Failed to create subdirectory {subcategory_path}: {e}"
                        )
                        return False

        return True

    def categorize_files(self) -> Dict[str, List[str]]:
        """
        Intelligently categorize files across the project

        Returns:
            Dictionary of file categories and their contents
        """
        file_categories = {
            "source_code": [],
            "documentation": [],
            "configuration": [],
            "scripts": [],
            "logs": [],
            "data": [],
        }

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                file_path = os.path.join(root, file)

                # Categorize by file extension
                if file.endswith((".py", ".js", ".ts", ".cpp", ".c", ".h")):
                    file_categories["source_code"].append(file_path)
                elif file.endswith((".md", ".rst", ".txt", ".pdf")):
                    file_categories["documentation"].append(file_path)
                elif file.endswith((".yml", ".yaml", ".json", ".ini", ".conf")):
                    file_categories["configuration"].append(file_path)
                elif file.endswith((".sh", ".bash", ".zsh")):
                    file_categories["scripts"].append(file_path)
                elif file.endswith(".log"):
                    file_categories["logs"].append(file_path)
                elif file.endswith((".csv", ".json", ".xml", ".db")):
                    file_categories["data"].append(file_path)

        return file_categories

    def generate_project_index(self) -> Dict[str, Any]:
        """
        Generate a comprehensive project index

        Returns:
            Detailed project index
        """
        # Validate project structure
        self.validate_project_structure()

        # Categorize files
        file_categories = self.categorize_files()

        # Build project index
        self.project_index["structure"] = {
            category: len(files) for category, files in file_categories.items()
        }

        # File type tracking
        file_types = {}
        for category, files in file_categories.items():
            for file_path in files:
                ext = os.path.splitext(file_path)[1]
                file_types[ext] = file_types.get(ext, 0) + 1

        self.project_index["file_types"] = file_types

        # Dependency tracking
        self.project_index["dependencies"] = self._track_dependencies()

        return self.project_index

    def _track_dependencies(self) -> Dict[str, List[str]]:
        """
        Track project dependencies across different modules

        Returns:
            Dictionary of module dependencies
        """
        dependencies = {}

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".py"):
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
                            dependencies[file_path] = imports
                    except Exception as e:
                        logger.warning(f"Could not process {file_path}: {e}")

        return dependencies

    def persist_project_index(self):
        """
        Persist project index to file
        """
        index_path = os.path.join(
            self.base_dir,
            f'logs/project_index_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
        )

        with open(index_path, "w") as f:
            json.dump(self.project_index, f, indent=2)

        logger.info(f"Project index generated: {index_path}")

    def run_project_organization(self):
        """
        Execute comprehensive project organization workflow
        """
        try:
            # Generate project index
            self.generate_project_index()

            # Persist project index
            self.persist_project_index()

            logger.info("Project organization completed successfully")

        except Exception as e:
            logger.error(f"Project organization failed: {e}")


def main():
    """
    Main execution for project organization
    """
    try:
        organizer = ProjectOrganizer()
        organizer.run_project_organization()

    except Exception as e:
        print(f"Project organization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
