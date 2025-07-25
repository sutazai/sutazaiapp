#!/usr/bin/env python3.11
"""
Comprehensive Code Cleanup Script

This script provides comprehensive code cleanup utilities for Python projects,
including removing unused imports, fixing formatting, and cleaning up files.
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeCleaner:
    """
    A comprehensive code cleanup utility for Python projects.
    """
    
    def __init__(self, base_path: str):
        """
        Initialize the code cleaner.
        
        Args:
            base_path: Base directory of the project to clean
        """
        self.base_path = base_path
        self.ignored_dirs = {".git", ".mypy_cache", "venv", "__pycache__"}

    def find_python_files(self) -> List[str]:
        """Find all Python files in the project."""
        python_files = []
        for root, _, files in os.walk(self.base_path):
            if not any(ignored in root for ignored in self.ignored_dirs):
                python_files.extend([
                    os.path.join(root, file) 
                    for file in files 
                    if file.endswith(".py")
                ])
        return python_files

    def remove_unused_imports(self, file_path: str) -> bool:
        """Remove unused imports from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            # Parse the source code
            try:
                module = ast.parse(source)
            except SyntaxError:
                logger.warning(f"Syntax error in {file_path}, skipping")
                return False

            # Analyze imports (simplified approach)
            used_names = self._get_used_names(module)
            import_lines = []
            
            lines = source.split('\n')
            new_lines = []
            
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    # Simple heuristic - if any part of the import is used, keep it
                    if any(name in line for name in used_names):
                        new_lines.append(line)
                    else:
                        logger.info(f"Removing unused import in {file_path}: {line.strip()}")
                else:
                    new_lines.append(line)
            
            # Write back if changes were made
            new_source = '\n'.join(new_lines)
            if new_source != source:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_source)
                return True
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            
        return False

    def _get_used_names(self, module: ast.AST) -> Set[str]:
        """Extract all names used in the module."""
        used_names = set()
        
        for node in ast.walk(module):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
                    
        return used_names

    def fix_indentation(self, file_path: str) -> bool:
        """Fix basic indentation issues in a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                # Replace tabs with 4 spaces
                new_line = line.expandtabs(4)
                new_lines.append(new_line)

            new_content = ''.join(new_lines)
            
            # Write back if changes were made
            if new_content != ''.join(lines):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                logger.info(f"Fixed indentation in {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error fixing indentation in {file_path}: {e}")
            
        return False

    def remove_trailing_whitespace(self, file_path: str) -> bool:
        """Remove trailing whitespace from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Remove trailing whitespace from each line
            lines = content.split('\n')
            new_lines = [line.rstrip() for line in lines]
            new_content = '\n'.join(new_lines)

            # Write back if changes were made
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                logger.info(f"Removed trailing whitespace from {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error removing trailing whitespace from {file_path}: {e}")
            
        return False

    def clean_empty_lines(self, file_path: str) -> bool:
        """Clean up excessive empty lines."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Replace multiple consecutive empty lines with max 2 empty lines
            new_content = re.sub(r'\n\s*\n\s*\n+', '\n\n\n', content)
            
            # Write back if changes were made
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                logger.info(f"Cleaned empty lines in {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error cleaning empty lines in {file_path}: {e}")
            
        return False

    def cleanup_project(self) -> Tuple[int, int]:
        """
        Run comprehensive cleanup on the entire project.
        
        Returns:
            Tuple of (files_processed, files_modified)
        """
        python_files = self.find_python_files()
        files_processed = 0
        files_modified = 0
        
        logger.info(f"Found {len(python_files)} Python files to process")
        
        for file_path in python_files:
            files_processed += 1
            modified = False
            
            # Apply various cleanup operations
            if self.fix_indentation(file_path):
                modified = True
            if self.remove_trailing_whitespace(file_path):
                modified = True
            if self.clean_empty_lines(file_path):
                modified = True
            # Note: Commented out as it might be too aggressive
            # if self.remove_unused_imports(file_path):
            #     modified = True
                
            if modified:
                files_modified += 1
                
        logger.info(f"Processed {files_processed} files, modified {files_modified}")
        return files_processed, files_modified


def main():
    """Main entry point for the cleanup script."""
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = os.getcwd()
        
    logger.info(f"Starting comprehensive cleanup of {base_path}")
    
    cleaner = CodeCleaner(base_path)
    files_processed, files_modified = cleaner.cleanup_project()
    
    logger.info(f"Cleanup completed: {files_modified}/{files_processed} files modified")


if __name__ == "__main__":
    main()
