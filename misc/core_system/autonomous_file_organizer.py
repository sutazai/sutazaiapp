#!/usr/bin/env python3
"""
SutazAI Autonomous File Organization Utility

Provides intelligent, self-organizing file management
with semantic categorization and optimization capabilities.
"""

import hashlib
import logging
import os
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Optional

import magic
import networkx as nx


class AutonomousFileOrganizer:
    """
    Comprehensive Autonomous File Management System

    Key Capabilities:
    - Intelligent file categorization
    - Semantic file organization
    - Duplicate detection and management
    """

    FILE_TYPE_MAPPING = {
        "application/python": "code",
        "text/x-python": "code",
        "text/markdown": "docs",
        "application/json": "data",
        "application/yaml": "config",
        "text/plain": "text",
        "application/pdf": "documents",
        "image/": "media",
        "video/": "media",
        "audio/": "media",
        "application/zip": "archives",
        "application/x-tar": "archives",
    }

    def __init__(
        self,
        base_dir: str = "/opt/SutazAI",
        log_dir: Optional[str] = None,
    ):
        """
        Initialize Autonomous File Organizer

        Args:
            base_dir (str): Base project directory
            log_dir (Optional[str]): Custom log directory
        """
        self.base_dir = base_dir
        self.log_dir = log_dir or os.path.join(
            base_dir, "logs", "file_organization"
        )

        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            filename=os.path.join(
                self.log_dir, "autonomous_file_organizer.log"
            ),
        )
        self.logger = logging.getLogger("SutazAI.AutonomousFileOrganizer")

        # Initialize file tracking graph
        self.file_graph = nx.DiGraph()

    def categorize_file(self, file_path: str) -> str:
        """
        Intelligently categorize a file based on its MIME type

        Args:
            file_path (str): Path to the file

        Returns:
            Categorized file type
        """
        try:
            # Use python-magic for MIME type detection
            mime = magic.Magic(mime=True)
            file_mime = mime.from_file(file_path)

            # Match MIME type to category
            for pattern, category in self.FILE_TYPE_MAPPING.items():
                if pattern in file_mime:
                    return category

            return "misc"

        except Exception as e:
            self.logger.warning(
                f"File categorization failed for {file_path}: {e}"
            )
            return "misc"

    def generate_file_hash(self, file_path: str) -> str:
        """
        Generate a secure hash for file identification

        Args:
            file_path (str): Path to the file

        Returns:
            SHA-256 hash of the file
        """
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        except Exception as e:
            self.logger.warning(
                f"File hash generation failed for {file_path}: {e}"
            )
            return ""

    def detect_duplicates(self, directory: str) -> Dict[str, List[str]]:
        """
        Detect and track duplicate files across the project

        Args:
            directory (str): Directory to scan for duplicates

        Returns:
            Dictionary of duplicate files grouped by hash
        """
        file_hashes = {}

        try:
            for root, _, files in os.walk(directory):
                for file in files:
                    full_path = os.path.join(root, file)

                    # Generate file hash
                    file_hash = self.generate_file_hash(full_path)

                    if file_hash:
                        if file_hash not in file_hashes:
                            file_hashes[file_hash] = []
                        file_hashes[file_hash].append(full_path)

        except Exception as e:
            self.logger.error(f"Duplicate file detection failed: {e}")

        # Filter out unique files
        return {
            hash_val: paths
            for hash_val, paths in file_hashes.items()
            if len(paths) > 1
        }

    def organize_files(self, directory: Optional[str] = None):
        """
        Perform comprehensive file organization

        Args:
            directory (Optional[str]): Specific directory to organize
        """
        try:
            # Use base directory if no specific directory provided
            target_dir = directory or self.base_dir

            # Detect duplicates
            duplicates = self.detect_duplicates(target_dir)
            if duplicates:
                self.logger.warning(
                    f"Detected {len(duplicates)} duplicate file groups"
                )
                self._handle_duplicates(duplicates)

            # Organize files by category
            for root, _, files in os.walk(target_dir):
                for file in files:
                    full_path = os.path.join(root, file)

                    # Skip log files and already organized directories
                    if any(
                        skip in full_path
                        for skip in ["logs", "backups", ".git"]
                    ):
                        continue

                    # Categorize file
                    category = self.categorize_file(full_path)

                    # Create category directory if not exists
                    category_dir = os.path.join(self.base_dir, category)
                    os.makedirs(category_dir, exist_ok=True)

                    # Move file to category directory
                    target_path = os.path.join(category_dir, file)

                    # Prevent overwriting
                    if os.path.exists(target_path):
                        base, ext = os.path.splitext(file)
                        target_path = os.path.join(
                            category_dir,
                            f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}",
                        )

                    # Move file
                    shutil.move(full_path, target_path)
                    self.logger.info(
                        f"Organized file: {full_path} -> {target_path}"
                    )

        except Exception as e:
            self.logger.error(f"File organization failed: {e}")

    def _handle_duplicates(self, duplicates: Dict[str, List[str]]):
        """
        Intelligently handle duplicate files

        Args:
            duplicates (Dict): Dictionary of duplicate file groups
        """
        try:
            for hash_val, duplicate_paths in duplicates.items():
                # Keep the most recent file, remove others
                duplicate_paths.sort(key=os.path.getmtime, reverse=True)

                # Keep the first (most recent) file
                keep_file = duplicate_paths[0]

                # Remove other duplicates
                for duplicate_path in duplicate_paths[1:]:
                    try:
                        os.remove(duplicate_path)
                        self.logger.info(
                            f"Removed duplicate file: {duplicate_path}"
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Could not remove duplicate {duplicate_path}: {e}"
                        )

        except Exception as e:
            self.logger.error(f"Duplicate file handling failed: {e}")

    def build_file_relationship_graph(self):
        """
        Build a comprehensive graph of file relationships

        Returns:
            NetworkX Directed Graph of file relationships
        """
        try:
            # Reset graph
            self.file_graph = nx.DiGraph()

            for root, _, files in os.walk(self.base_dir):
                for file in files:
                    full_path = os.path.join(root, file)

                    # Add file as a node
                    self.file_graph.add_node(
                        full_path,
                        category=self.categorize_file(full_path),
                        size=os.path.getsize(full_path),
                        modified=os.path.getmtime(full_path),
                    )

                    # Add edges based on directory structure
                    parent_dir = os.path.dirname(full_path)
                    self.file_graph.add_edge(parent_dir, full_path)

            return self.file_graph

        except Exception as e:
            self.logger.error(
                f"File relationship graph generation failed: {e}"
            )
            return nx.DiGraph()


def main():
    """
    Main execution for autonomous file organization
    """
    try:
        file_organizer = AutonomousFileOrganizer()
        file_organizer.organize_files()

        # Build and log file relationship graph
        file_graph = file_organizer.build_file_relationship_graph()
        print(f"Total files tracked: {len(file_graph.nodes())}")
        print(f"Total file relationships: {len(file_graph.edges())}")

    except Exception as e:
        print(f"Autonomous file organization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
