#!/usr/bin/env python3
import argparse
import ast
import io
import logging
import os
import re
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import black


class AdvancedSyntaxFixer:
    def __init__(
        self,
        base_path="/opt/sutazaiapp",
        log_file="/opt/sutazaiapp/logs/syntax_fixer.log",
    ):
        """
        Initialize the syntax fixer.
        Args:
            base_path: Base directory to scan for Python files
            log_file: Path to log file
        """
        self.base_path = base_path
        self.logger = logging.getLogger(__name__)
        self.fixed_files = []
        self.failed_files = []

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def find_python_files(self) -> List[str]:
        """
        Recursively find all Python files in the base path.

        :return: List of Python file paths
        """
        python_files = []
        for root, dirs, files in os.walk(self.base_path):
            # Skip virtual environment directories
            if "venv" in dirs:
                dirs.remove("venv")
            if ".venv" in dirs:
                dirs.remove(".venv")

            # Skip other common directories that shouldn't be modified
            skip_dirs = [".git", "__pycache__", ".pytest_cache", "build", "dist"]
            for skip_dir in skip_dirs:
                if skip_dir in dirs:
                    dirs.remove(skip_dir)

            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files

    def _remove_unused_imports(self, file_path: str) -> bool:
        """
        Remove unused imports from a Python file.

        :param file_path: Path to the Python file
        :return: Boolean indicating if fixes were made
        """
        try:
            with open(file_path) as f:
                source_code = f.read()

            # Parse the source code
            try:
                module = ast.parse(source_code)
            except SyntaxError as e:
                self.logger.warning(f"Syntax error in {file_path}: {e}")
                return False

            # Analyze imports
            imports_to_remove = []
            for node in ast.walk(module):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not self._is_import_used(module, alias.name):
                            imports_to_remove.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module
                    for alias in node.names:
                        full_name = (
                            f"{module_name}.{alias.name}" if module_name else alias.name
                        )
                        if not self._is_import_used(module, full_name):
                            imports_to_remove.append(full_name)

            # Remove unused imports
            lines = source_code.split("\n")
            cleaned_lines = [
                line
                for line in lines
                if not any(imp in line for imp in imports_to_remove)
            ]

            cleaned_source = "\n".join(cleaned_lines)

            # Write cleaned source
            with open(file_path, "w") as f:
                f.write(cleaned_source)

            return True
        except Exception as e:
            self.logger.error(f"Error removing unused imports from {file_path}: {e}")
            return False

    def _is_import_used(self, module: ast.Module, import_name: str) -> bool:
        """
        Check if an import is used in the module.

        :param module: AST module
        :param import_name: Name of the import to check
        :return: Boolean indicating if the import is used
        """
        for node in ast.walk(module):
            if isinstance(node, (ast.Name, ast.Attribute)):
                if import_name in str(node.id if hasattr(node, "id") else node.attr):
                    return True
        return False

    def _format_with_black(self, file_path: str) -> bool:
        """
        Format the file using Black code formatter.

        :param file_path: Path to the Python file to format
        :return: Boolean indicating if formatting was successful
        """
        try:
            with open(file_path) as f:
                source_code = f.read()

            # Use Black to format the code
            try:
                formatted_code = black.format_str(source_code, mode=black.FileMode())
            except black.InvalidInput as e:
                self.logger.warning(f"Black could not format {file_path}: {e}")
                return False

            # Write formatted code if different from original
            if formatted_code != source_code:
                with open(file_path, "w") as f:
                    f.write(formatted_code)
                return True

            return False
        except Exception as e:
            self.logger.error(f"Error formatting {file_path} with Black: {e}")
            return False

    def _advanced_syntax_repair(self, file_path: str) -> bool:
        """
        Perform advanced syntax repair for complex syntax errors.

        :param file_path: Path to the Python file to repair
        :return: Boolean indicating if repairs were successful
        """
        try:
            with open(file_path) as f:
                source_code = f.read()

            # Attempt to parse the source code
            try:
                ast.parse(source_code)
                return False  # No syntax errors to repair
            except SyntaxError as e:
                self.logger.warning(f"Syntax error in {file_path}: {e}")

            # Advanced repair strategies
            repaired_lines = []
            lines = source_code.split("\n")

            # Track indentation and block state
            current_indent = 0
            in_function = False
            in_class = False

            for line_num, line in enumerate(lines, 1):
                stripped_line = line.strip()

                # Skip empty lines
                if not stripped_line:
                    repaired_lines.append(line)
                    continue

                # Detect function and class definitions
                if stripped_line.startswith("def "):
                    in_function = True
                    current_indent = len(line) - len(line.lstrip())
                    repaired_lines.append(line)
                    continue

                if stripped_line.startswith("class "):
                    in_class = True
                    current_indent = len(line) - len(line.lstrip())
                    repaired_lines.append(line)
                    continue

                # Ensure proper indentation for blocks
                if in_function or in_class:
                    # Ensure line is indented
                    if len(line) - len(line.lstrip()) < current_indent + 4:
                        line = " " * (current_indent + 4) + stripped_line

                    # Reset block states for end of blocks
                    if stripped_line == "pass" or stripped_line.startswith("return "):
                        in_function = False
                        in_class = False

                    # Remove trailing whitespaces and ensure newline
                    line = line.rstrip() + "\n"

                repaired_lines.append(line)

            # Ensure file ends with newline
            if not repaired_lines or not repaired_lines[-1].endswith("\n"):
                repaired_lines.append("\n")

            # Write repaired source
            with open(file_path, "w") as f:
                f.writelines(repaired_lines)

            return True
        except Exception as e:
            self.logger.error(f"Advanced syntax repair failed for {file_path}: {e}")
            return False

    def _fix_line_syntax(self, line: str) -> str:
        """
        Apply line-level syntax fixes.

        :param line: Single line of code
        :return: Corrected line of code
        """
        # Remove trailing whitespaces
        line = line.rstrip() + "\n"

        # Fix common syntax patterns
        # Ensure clean function/class definition
        line = re.sub(r":\s*$", ":", line)
        line = re.sub(r"\s+$", "\n", line)  # Remove trailing whitespaces

        return line

    def fix_syntax_errors(self) -> Dict[str, List[str]]:
        """
        Find and fix syntax errors in Python files.

        :return: Dictionary of fixed and failed files
        """
        python_files = self.find_python_files()

        for file_path in python_files:
            try:
                # Remove unused imports
                if self._remove_unused_imports(file_path):
                    self.logger.info(f"Removed unused imports from {file_path}")

                # Advanced syntax repair
                if self._advanced_syntax_repair(file_path):
                    self.logger.info(f"Performed advanced syntax repair on {file_path}")

                # Format with Black
                if self._format_with_black(file_path):
                    self.logger.info(f"Formatted {file_path} with Black")
                    self.fixed_files.append(file_path)
                else:
                    self.failed_files.append(
                        {
                            "file": file_path,
                            "error": "Could not automatically fix syntax",
                        },
                    )
            except Exception as e:
                self.logger.error(f"Unexpected error fixing {file_path}: {e}")
                self.failed_files.append(
                    {
                        "file": file_path,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )

        self.logger.info(
            f"Syntax fixing complete. Fixed {len(self.fixed_files)} files.",
        )

        return {
            "fixed_files": self.fixed_files,
            "failed_files": [f["file"] for f in self.failed_files],
        }

    def generate_report(self) -> str:
        """
        Generate a comprehensive syntax fixing report.

        :return: Formatted report string
        """
        report = ["Comprehensive Syntax Fixing Report", "=" * 40]

        report.append(
            f"Total files processed: {len(self.fixed_files) + len(self.failed_files)}",
        )
        report.append(f"Successfully fixed files: {len(self.fixed_files)}")
        report.append(f"Failed to fix files: {len(self.failed_files)}")

        if self.fixed_files:
            report.append("\nFixed Files:")
            for file in self.fixed_files:
                report.append(f"  - {file}")

        if self.failed_files:
            report.append("\nFailed Files:")
            for file_info in self.failed_files:
                report.append(f"  - {file_info['file']}")
                report.append(f"    Error: {file_info.get('error', 'Unknown error')}")

        return "\n".join(report)


def main():
    # Parse command-line arguments to control backup behavior
    parser = argparse.ArgumentParser(description="Advanced Syntax Fixer")
    args = parser.parse_args()

    # Initialize fixer with backup option
    fixer = AdvancedSyntaxFixer()
    results = fixer.fix_syntax_errors()

    # Generate and print report
    report = fixer.generate_report()
    print(report)

    # Optionally, write the report to a file
    report_path = "/opt/sutazaiapp/logs/syntax_fixing_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nDetailed report saved to {report_path}")


if __name__ == "__main__":
    main()
