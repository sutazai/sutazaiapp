#!/usr/bin/env python3
"""
Path Correction Utility for SutazAI

This script fixes path inconsistencies throughout the codebase by replacing
incorrect path references with the correct ones.
"""

import os
import re
import sys
from typing import Dict, Pattern, Set


class PathCorrector:
    """Utility to correct paths throughout the codebase."""

    def __init__(self):
        """Initialize the path corrector with common replacements."""
        self.incorrect_path_patterns: Dict[Pattern, str] = {
            re.compile(r"/opt/SutazAI"): "/opt/SutazAI",
            # Add more patterns as needed
        }

        # File extensions to process
        self.file_extensions: Set[str] = {
            ".py",
            ".sh",
            ".yml",
            ".yaml",
            ".json",
            ".md",
            ".txt",
            ".conf",
        }

        # Counter for statistics
        self.files_processed: int = 0
        self.files_modified: int = 0
        self.replacements_made: int = 0

    def process_directory(self, directory: str) -> None:
        """
        Process all relevant files in a directory and its subdirectories.

        Args:
            directory: Root directory to start processing from
        """
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if self._should_process_file(file_path):
                    replacements = self.process_file(file_path)
                    self.files_processed += 1
                    if replacements > 0:
                        self.files_modified += 1
                        self.replacements_made += replacements

    def _should_process_file(self, file_path: str) -> bool:
        """
        Check if the file should be processed based on extension and other criteria.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if the file should be processed
        """
        # Skip git, venv, cache dirs
        if any(part.startswith(".") for part in file_path.split("/")):
            if (
                ".git" in file_path
                or ".venv" in file_path
                or ".cache" in file_path
            ):
                return False

        # Check file extension
        _, ext = os.path.splitext(file_path)
        return ext in self.file_extensions

    def process_file(self, file_path: str) -> int:
        """
        Process a single file, replacing incorrect paths.

        Args:
            file_path: Path to the file to process

        Returns:
            int: Number of replacements made
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content
            replacements = 0

            for pattern, replacement in self.incorrect_path_patterns.items():
                replaced_content, count = pattern.subn(replacement, content)
                if count > 0:
                    content = replaced_content
                    replacements += count

            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Updated {file_path} ({replacements} replacements)")
                return replacements

        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")

        return 0

    def print_summary(self) -> None:
        """Print summary of changes made."""
        print("\n=== Path Correction Summary ===")
        print(f"Files processed: {self.files_processed}")
        print(f"Files modified: {self.files_modified}")
        print(f"Total replacements: {self.replacements_made}")


def main():
    """Main function to run the path corrector."""
    # Determine the root directory (current directory or specified)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]

    print(f"Starting path correction from {root_dir}")
    corrector = PathCorrector()
    corrector.process_directory(root_dir)
    corrector.print_summary()


if __name__ == "__main__":
    main()
