#!/usr/bin/env python3
"""
SutazAI File Index CLI

Command-line interface for interacting with the SutazAI file indexing system.
Provides easy access to project file information and search capabilities.
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.file_indexer import FileIndexer

class FileIndexCLI:
    """
    Command-line interface for file indexing operations.
    """

    def __init__(self, project_root: str = '/opt/sutazai_project/SutazAI'):
        """
        Initialize the CLI with a file indexer.

        Args:
            project_root (str): Root directory of the project
        """
        self.indexer = FileIndexer(project_root)

    def search_files(self, query: str) -> None:
        """
        Search for files matching a query.

        Args:
            query (str): Search query
        """
        matches = self.indexer.find_files(query)
        
        if not matches:
            print(f"No files found matching '{query}'.")
            return

        print(f"Files matching '{query}':")
        for match in matches:
            print(f"  - {match}")

    def show_file_details(self, file_path: str) -> None:
        """
        Display details of a specific file.

        Args:
            file_path (str): Path to the file
        """
        index = self.indexer.load_index()
        details = index.get('file_details', {}).get(file_path)

        if not details:
            print(f"No details found for file: {file_path}")
            return

        print(f"File Details for {file_path}:")
        for key, value in details.items():
            print(f"  {key.capitalize()}: {value}")

    def list_project_structure(self, depth: int = 2) -> None:
        """
        List the project file structure.

        Args:
            depth (int): Depth of directory traversal
        """
        index = self.indexer.load_index()
        file_tree = index.get('file_tree', {})

        def print_tree(tree: Dict[str, Any], current_depth: int = 0):
            """
            Recursively print file tree.

            Args:
                tree (Dict[str, Any]): Current tree level
                current_depth (int): Current depth in the tree
            """
            if current_depth > depth:
                return

            for key, value in tree.items():
                if key == '__files__':
                    for file in value:
                        print('  ' * (current_depth + 1) + file)
                else:
                    print('  ' * current_depth + key + '/')
                    if isinstance(value, dict):
                        print_tree(value, current_depth + 1)

        print("Project Structure:")
        print_tree(file_tree)

    def generate_index(self) -> None:
        """
        Manually generate and save the project file index.
        """
        print("Generating project file index...")
        project_index = self.indexer.index_project()
        self.indexer.save_index(project_index)
        print("Project file index generated successfully.")

def main():
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(description='SutazAI File Index CLI')
    parser.add_argument('action', choices=['search', 'details', 'structure', 'index'], 
                        help='Action to perform')
    parser.add_argument('query', nargs='?', default=None, 
                        help='Search query or file path')
    parser.add_argument('--depth', type=int, default=2, 
                        help='Depth for structure listing (default: 2)')

    args = parser.parse_args()

    cli = FileIndexCLI()

    if args.action == 'search':
        if not args.query:
            print("Please provide a search query.")
            sys.exit(1)
        cli.search_files(args.query)
    
    elif args.action == 'details':
        if not args.query:
            print("Please provide a file path.")
            sys.exit(1)
        cli.show_file_details(args.query)
    
    elif args.action == 'structure':
        cli.list_project_structure(args.depth)
    
    elif args.action == 'index':
        cli.generate_index()

if __name__ == '__main__':
    main() 