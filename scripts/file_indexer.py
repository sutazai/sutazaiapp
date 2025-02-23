#!/usr/bin/env python3
"""
SutazAI Comprehensive File Indexer

Automatically generates and maintains a comprehensive index of project files,
providing easy access and navigation across the entire project structure.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class FileIndexer:
    """
    Advanced file indexing system for SutazAI project.
    """

    def __init__(
        self, 
        project_root: str = '/opt/sutazai_project/SutazAI', 
        index_file: str = 'sutazai_file_index.json'
    ):
        """
        Initialize the file indexer.

        Args:
            project_root (str): Root directory of the project
            index_file (str): Name of the index file
        """
        self.project_root = os.path.abspath(project_root)
        self.index_file_path = os.path.join(project_root, index_file)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger('FileIndexer')

    def _generate_file_hash(self, file_path: str) -> str:
        """
        Generate a hash for a file to track changes.

        Args:
            file_path (str): Path to the file

        Returns:
            str: SHA-256 hash of the file contents
        """
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            self.logger.warning(f"Could not hash file {file_path}: {e}")
            return ''

    def index_project(self, exclude_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive index of the project files.

        Args:
            exclude_patterns (Optional[List[str]]): Patterns of files/directories to exclude

        Returns:
            Dict[str, Any]: Comprehensive project file index
        """
        exclude_patterns = exclude_patterns or [
            '.git', '__pycache__', '*.pyc', '.venv', 'venv', 
            'node_modules', '.idea', '.vscode', '*.log'
        ]

        project_index = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'project_root': self.project_root
            },
            'file_tree': {},
            'file_details': {}
        }

        for root, dirs, files in os.walk(self.project_root):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]

            # Relative path from project root
            relative_path = os.path.relpath(root, self.project_root)

            # Create nested dictionary for file tree
            current_level = project_index['file_tree']
            for part in relative_path.split(os.path.sep):
                if part != '.':
                    current_level = current_level.setdefault(part, {})

            # Add files to the current level
            current_level['__files__'] = []
            for file in files:
                # Skip files matching exclusion patterns
                if any(pattern in file for pattern in exclude_patterns):
                    continue

                full_path = os.path.join(root, file)
                relative_file_path = os.path.relpath(full_path, self.project_root)

                try:
                    file_stat = os.stat(full_path)
                    file_details = {
                        'name': file,
                        'path': relative_file_path,
                        'size': file_stat.st_size,
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        'hash': self._generate_file_hash(full_path)
                    }

                    current_level['__files__'].append(file)
                    project_index['file_details'][relative_file_path] = file_details

                except Exception as e:
                    self.logger.warning(f"Could not index file {full_path}: {e}")

        return project_index

    def save_index(self, index: Optional[Dict[str, Any]] = None) -> None:
        """
        Save the project file index to a JSON file.

        Args:
            index (Optional[Dict[str, Any]]): Project file index to save
        """
        if index is None:
            index = self.index_project()

        try:
            with open(self.index_file_path, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2)
            
            self.logger.info(f"Project file index saved to {self.index_file_path}")
        except Exception as e:
            self.logger.error(f"Could not save file index: {e}")

    def load_index(self) -> Dict[str, Any]:
        """
        Load the existing project file index.

        Returns:
            Dict[str, Any]: Loaded project file index
        """
        try:
            with open(self.index_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning("No existing file index found. Generating new index.")
            index = self.index_project()
            self.save_index(index)
            return index
        except Exception as e:
            self.logger.error(f"Could not load file index: {e}")
            return {}

    def find_files(self, query: str) -> List[str]:
        """
        Find files matching a query.

        Args:
            query (str): Search query

        Returns:
            List[str]: List of matching file paths
        """
        index = self.load_index()
        matches: List[str] = []

        for path, details in index.get('file_details', {}).items():
            if query.lower() in path.lower() or query.lower() in details['name'].lower():
                matches.append(path)

        return matches

def main():
    indexer = FileIndexer()
    
    # Generate and save project index
    project_index = indexer.index_project()
    indexer.save_index(project_index)

    # Example: Find files
    print("Finding Python files:")
    python_files = indexer.find_files('.py')
    for file in python_files:
        print(file)

if __name__ == '__main__':
    main() 