#!/usr/bin/env python3.11
"""
Comprehensive Syntax Fixer for SutazAI Project
"""

import ast
import logging
import os
import re
from typing import List, Optional

logging.basicConfig(
level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class SyntaxFixer:
    def __init__(self, base_path: str = "/opt/sutazaiapp"):
        """
        Initialize the syntax fixer.

        Args:
        base_path: Root directory to search for Python files
        """
        self.base_path = base_path
        self.ignored_dirs = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        }

        def find_python_files(self) -> List[str]:
            """
            Recursively find all Python files in the base path.

            Returns:
            List of Python file paths
            """
            python_files = []
            for root, dirs, files in os.walk(self.base_path):
            # Remove ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignored_dirs]

            for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
                return python_files

            def fix_imports(self, source: str) -> str:
                """Fix import statements and ensure proper formatting."""
                lines = source.split("\n")
                fixed_lines = []
                import_block = []
                in_import_block = False

                for line in lines:
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")):
                    # Fix common import formatting issues
                    if "as" in stripped:
                        # Fix 'from x import y as z' format
                        parts = stripped.split(" as ")
                        if len(parts) == 2:
                            base, alias = parts
                            if base.startswith("from"):
                                # Handle 'from x import y as z'
                                module_parts = base.split(" import ")
                                if len(module_parts) == 2:
                                    fixed_line = f"{module_parts[0]} import {module_parts[1].strip()} as {alias.strip()}"
                                    else:
                                        fixed_line = line
                                        else:
                                            # Handle 'import x as y'
                                            fixed_line = f"import {base.replace('import', '').strip()} as {alias.strip()}"
                                            else:
                                                fixed_line = line
                                                else:
                                                    fixed_line = line

                                                    in_import_block = True
                                                    import_block.append(fixed_line)
                                                    else:
                                                        if in_import_block:
                                                            # Sort and deduplicate imports
                                                            import_block.sort()
                                                            import_block = list(dict.fromkeys(import_block))
                                                            fixed_lines.extend(import_block)
                                                            import_block = []
                                                            in_import_block = False
                                                            fixed_lines.append(line)

                                                            if import_block:  # Handle trailing imports
                                                                import_block.sort()
                                                                import_block = list(dict.fromkeys(import_block))
                                                                fixed_lines.extend(import_block)

                                                                return "\n".join(fixed_lines)

                                                            def fix_indentation(self, source: str) -> str:
                                                                """Fix indentation issues using a more robust approach."""
                                                                lines = source.split("\n")
                                                                fixed_lines = []
                                                                indent_stack = [0]

                                                                # Convert tabs to spaces first
                                                                lines = [line.replace("\t", "    ") for line in lines]

                                                                for line in lines:
                                                                stripped = line.strip()
                                                                if not stripped:  # Empty line
                                                                    fixed_lines.append("")
                                                                    continue

                                                                # Calculate current indentation
                                                                current_indent = len(line) - len(line.lstrip())

                                                                # Normalize indentation to be a multiple of 4
                                                                if current_indent > 0:
                                                                    current_indent = ((current_indent + 3) // 4) * 4

                                                                    if stripped.startswith(
                                                                        (
                                                                        "def ",
                                                                        "class ",
                                                                        "if ",
                                                                        "elif ",
                                                                        "else:",
                                                                        "try:",
                                                                        "except ",
                                                                        "finally:",
                                                                        "for ",
                                                                        "while ",
                                                                        ),
                                                                        ):
                                                                        # Block starter - increase indentation for next line
                                                                        fixed_lines.append(" " * indent_stack[-1] + stripped)
                                                                        indent_stack.append(indent_stack[-1] + 4)
                                                                        elif (
                                                                            stripped == "pass"
                                                                            or stripped.startswith("return ")
                                                                            or stripped == "break"
                                                                            or stripped == "continue"
                                                                            ):
                                                                            # Block ender - decrease indentation after this line
                                                                            if len(indent_stack) > 1:
                                                                                indent_stack.pop()
                                                                                fixed_lines.append(" " * indent_stack[-1] + stripped)
                                                                                else:
                                                                                    # Regular line - maintain current indentation
                                                                                    fixed_lines.append(" " * indent_stack[-1] + stripped)

                                                                                    return "\n".join(fixed_lines)

                                                                                def fix_try_except(self, source: str) -> str:
                                                                                    """Fix try-except blocks to ensure proper structure."""
                                                                                    lines = source.split("\n")
                                                                                    fixed_lines = []
                                                                                    in_try_block = False
                                                                                    has_except = False

                                                                                    for i, line in enumerate(lines):
                                                                                    stripped = line.strip()

                                                                                    if stripped.startswith("try:"):
                                                                                        in_try_block = True
                                                                                        fixed_lines.append(line)
                                                                                        elif in_try_block and stripped.startswith(("except ", "except:")):
                                                                                            has_except = True
                                                                                            fixed_lines.append(line)
                                                                                            elif in_try_block and stripped.startswith("finally:"):
                                                                                                has_except = True  # finally is also valid without except
                                                                                                fixed_lines.append(line)
                                                                                                elif (
                                                                                                    in_try_block
                                                                                                    and not has_except
                                                                                                    and not stripped.startswith(("except ", "except:", "finally:"))
                                                                                                    ):
                                                                                                    # Add a generic except if missing
                                                                                                    indent = len(line) - len(line.lstrip())
                                                                                                    fixed_lines.append(" " * indent + "except Exception as e:")
                                                                                                    fixed_lines.append(" " * (indent + 4) + 'logger.error(f"Error: {e}")')
                                                                                                    fixed_lines.append(line)
                                                                                                    in_try_block = False
                                                                                                    else:
                                                                                                        if stripped and not any(
                                                                                                            stripped.startswith(x)
                                                                                                            for x in ["except ", "except:", "finally:", "else:"]
                                                                                                            ):
                                                                                                            in_try_block = False
                                                                                                            fixed_lines.append(line)

                                                                                                            return "\n".join(fixed_lines)

                                                                                                        def fix_syntax_errors(self, file_path: str) -> Optional[str]:
                                                                                                            """Fix syntax errors in a Python file."""
                                                                                                            try:
                                                                                                                with open(file_path, encoding="utf-8") as f:
                                                                                                                source = f.read()

                                                                                                                # Try parsing the original source
                                                                                                                try:
                                                                                                                    ast.parse(source)
                                                                                                                    return None  # No syntax errors
                                                                                                                except SyntaxError:
                                                                                                                    # Apply fixes in sequence
                                                                                                                    fixed_source = source
                                                                                                                    fixed_source = self.fix_imports(fixed_source)
                                                                                                                    fixed_source = self.fix_indentation(fixed_source)
                                                                                                                    fixed_source = self.fix_try_except(fixed_source)

                                                                                                                    # Verify the fixed source
                                                                                                                    try:
                                                                                                                        ast.parse(fixed_source)
                                                                                                                        return fixed_source
                                                                                                                    except SyntaxError as e:
                                                                                                                        logger.error(f"Could not fix all syntax errors in {file_path}: {e}")
                                                                                                                        return None

                                                                                                                    except Exception as e:
                                                                                                                        logger.error(f"Error fixing {file_path}: {e}")
                                                                                                                        return None

                                                                                                                    def fix_project_syntax(self):
                                                                                                                        """Fix syntax errors across the entire project."""
                                                                                                                        python_files = self.find_python_files()
                                                                                                                        fixed_count = 0
                                                                                                                        error_count = 0

                                                                                                                        for file_path in python_files:
                                                                                                                        try:
                                                                                                                            fixed_source = self.fix_syntax_errors(file_path)
                                                                                                                            if fixed_source is not None:
                                                                                                                                with open(file_path, "w", encoding="utf-8") as f:
                                                                                                                                f.write(fixed_source)
                                                                                                                                logger.info(f"Fixed syntax in {file_path}")
                                                                                                                                fixed_count += 1
                                                                                                                                except Exception as e:
                                                                                                                                    logger.error(f"Error processing {file_path}: {e}")
                                                                                                                                    error_count += 1

                                                                                                                                    logger.info(
                                                                                                                                    f"Completed syntax fixing: {fixed_count} files fixed, {error_count} errors",
                                                                                                                                    )


                                                                                                                                    def main():
                                                                                                                                        """Main function to run the syntax fixer."""
                                                                                                                                        fixer = SyntaxFixer()
                                                                                                                                        fixer.fix_project_syntax()


                                                                                                                                        if __name__ == "__main__":
                                                                                                                                            main()
