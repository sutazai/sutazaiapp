#!/usr/bin/env python3.11
"""
Comprehensive Syntax Fixer for SutazAI Project

This script provides advanced syntax error detection and correction
across multiple Python files, addressing common syntax issues.
"""

import ast
import logging
import os
import re
from collections.abc import Mapping, Sequence
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveSyntaxFixer:
    """
    A comprehensive syntax fixing utility for Python projects.
    """

    def __init__(self, root_dir: str):
        """Initialize the syntax fixer with project root directory."""
        self.root_dir = root_dir
        self.errors_fixed = 0
        self.files_processed = 0

        def fix_syntax_errors(self, file_path: str) -> Optional[str]:
            """
            Attempt to fix syntax errors in a given Python file.

            Args:
            file_path: Path to the Python file to fix

            Returns:
            Optional[str]: Fixed content if successful, None otherwise
            """
            try:
                with open(file_path, encoding="utf-8") as f:
                content = f.read()

                # Fix indentation issues
                fixed_content = self._fix_indentation(content)

                # Validate syntax
                try:
                    ast.parse(fixed_content)
                    except SyntaxError as e:
                        logger.warning("Could not fully resolve syntax: %s", e)
                    return None

                    # Write fixed content
                    with open(file_path, "w", encoding="utf-8") as f:
                    f.write(fixed_content)

                    self.errors_fixed += 1
                return fixed_content

                except Exception as e:
                    logger.error("Error processing %s: %s", file_path, e)
                return None

                def _fix_indentation(self, content: str) -> str:
                    """
                    Fix indentation issues in Python code.

                    Args:
                    content: Python code content to fix

                    Returns:
                    str: Fixed Python code content
                    """
                    # Fix unindented except blocks
                    content = re.sub(
                    r"(except\s+\w+\s*:)\n(\s*pass)?$",
                    r"\1\n    pass",
                    content,
                    flags=re.MULTILINE,
                    )

                    # Split into lines for indentation processing
                    lines = content.split("\n")
                    fixed_lines = []
                    indent_level = 0
                    in_class = False
                    in_function = False

                    for i, line in enumerate(lines):
                        stripped = line.strip()

                        # Skip empty lines
                        if not stripped:
                            fixed_lines.append("")
                        continue

                        # Check for indentation levels
                        if stripped.startswith(("class ", "def ")):
                            if stripped.startswith("class "):
                                in_class = True
                                if stripped.startswith("def "):
                                    in_function = True

                                    # Calculate indentation
                                    leading_spaces = len(
                                        line) - len(line.lstrip())
                                    current_indent = leading_spaces // 4

                                    # Handle indentation errors
                                    current_indent = min(
                                        current_indent,
                                        indent_level + 1)

                                    # Apply correct indentation
                                    fixed_line = " " * (
                                        4 * current_indent) + stripped
                                    fixed_lines.append(fixed_line)

                                    # Update indent level based on line content
                                    if stripped.endswith(":"):
                                        indent_level = current_indent + 1
                                        elif stripped in (
                                            "break",
                                            "continue",
                                            "pass",
                                            "return",
                                            "raise") or \
                                        stripped.startswith(
                                            ("return ", "raise ")):
                                        if indent_level > 0:
                                            indent_level -= 1

                                        return "\n".join(fixed_lines)

                                        def process_directory(
                                            self) -> Mapping[str, Sequence[str]]:
                                            """
                                                                                        Process all Python files in \
                                                the project directory.

                                            Returns:
                                            Mapping[str, Sequence[str]]: Dictionary mapping file paths to fixed content lines
                                            """
                                            results: dict[str, list[str]] = {}

                                            logger.info(
                                                "Processing Python files in %s",
                                                self.root_dir)
                                            for root, _, files in os.walk(
                                                self.root_dir):
                                                for file in files:
                                                    if file.endswith(".py"):
                                                        file_path = os.path.join(
                                                            root,
                                                            file)
                                                        self.files_processed += 1

                                                        logger.info(
                                                            "Processing file: %s",
                                                            file_path)
                                                        fixed_content = self.fix_syntax_errors(
                                                            file_path)
                                                        if fixed_content:
                                                            results[file_path] = fixed_content.split(
                                                                "\n")

                                                            logger.info("Processed %d files, fixed %d files",
                                                            self.files_processed, self.errors_fixed)
                                                        return results

                                                        def generate_report(
                                                            self,
                                                            results: Mapping[str, Sequence[str]]) -> None:
                                                            """
                                                            Generate a comprehensive syntax fixing report.

                                                            Args:
                                                            results: Dictionary mapping file paths to fixed content lines
                                                            """
                                                            report_path = os.path.join(
                                                            self.root_dir,
                                                            "logs",
                                                            "syntax_fix_report.md",
                                                            )
                                                            os.makedirs(
                                                                os.path.dirname(report_path),
                                                                exist_ok=True)

                                                            with open(
                                                                report_path,
                                                                "w",
                                                                encoding="utf-8") as f:
                                                            f.write(
                                                                "# Syntax Fixing Report\n\n")
                                                            f.write(
                                                                f"**Total Files Processed:** {self.files_processed}\n")
                                                            f.write(
                                                                f"**Errors Fixed:** {self.errors_fixed}\n\n")

                                                            for file_path, content in results.items():
                                                                f.write(
                                                                    f"## {file_path}\n")
                                                                f.write(
                                                                    "```python\n")
                                                                f.write(
                                                                    "\n".join(content[:20]))  # First 20 lines
                                                                f.write(
                                                                    "\n```\n\n")

                                                                logger.info(
                                                                    "Report generated at %s",
                                                                    report_path)


                                                                def main() -> None:
                                                                    """Main function to run the syntax fixer."""
                                                                    project_root = "/opt/sutazaiapp"
                                                                    fixer = ComprehensiveSyntaxFixer(
                                                                        project_root)
                                                                    results = fixer.process_directory()
                                                                    fixer.generate_report(
                                                                        results)


                                                                    if __name__ == "__main__":
                                                                        main()
