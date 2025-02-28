#!/opt/sutazaiapp/venv/bin/python3
import ast
import logging
import os
from typing import Any, Dict, List

import autopep8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ComprehensiveCodeCleaner:    def __init__(self, base_path: str):        self.base_path = base_path
        self.ignored_dirs = {".git", ".mypy_cache", "venv", "__pycache__"}

    def find_python_files(self) -> List[str]:        """Find all Python files in the project."""
        python_files = []
        for root, _, files in os.walk(self.base_path):            if not any(ignored in root for ignored in self.ignored_dirs):                python_files.extend([
                    os.path.join(root, file)
                    for file in files
                    if file.endswith(".py")
                ])
        return python_files

    def remove_unused_imports(self, file_path: str) -> bool:        """Remove unused imports from a Python file."""
        try:            with open(file_path) as f:                source = f.read()

            # Parse the source code
            module = ast.parse(source)

            # Analyze imports
            imports_to_remove = []
            for node in ast.walk(module):                if isinstance(node, ast.Import):                    for alias in node.names:                        if not self._is_import_used(module, alias.name):                            imports_to_remove.append(alias.name)
                elif isinstance(node, ast.ImportFrom):                    module_name = node.module
                    for alias in node.names:                        full_name = f"{module_name}.{alias.name}" if module_name else alias.name
                        if not self._is_import_used(module, full_name):                            imports_to_remove.append(full_name)

            # Remove unused imports
            lines = source.split("\n")
            cleaned_lines = [
                line for line in lines
                if not any(imp in line for imp in imports_to_remove)
            ]

            cleaned_source = "\n".join(cleaned_lines)

            # Apply PEP 8 formatting
            cleaned_source = autopep8.fix_code(cleaned_source)

            with open(file_path, "w") as f:                f.write(cleaned_source)

            return True
        except Exception as e:            logger.error(f"Error cleaning {file_path}: {e}")
            return False

    def _is_import_used(self, module: ast.Module, import_name: str) -> bool:        """Check if an import is used in the module."""
        for node in ast.walk(module):            if isinstance(node, (ast.Name, ast.Attribute)):                if import_name in str(
                        node.id if hasattr(node, "id") else node.attr):                    return True
        return False

    def fix_syntax_errors(self, file_path: str) -> bool:        """Attempt to fix syntax errors in a Python file."""
        try:            with open(file_path) as f:                source = f.read()

            # Use autopep8 to attempt syntax correction
            cleaned_source = autopep8.fix_code(
                source, options={"aggressive": 2})

            with open(file_path, "w") as f:                f.write(cleaned_source)

            return True
        except Exception as e:            logger.error(f"Error fixing syntax in {file_path}: {e}")
            return False

    def run_comprehensive_cleanup(self):        """Run comprehensive cleanup on all Python files."""
        python_files = self.find_python_files()

        for file_path in python_files:            logger.info(f"Processing {file_path}")

            # Fix syntax errors
            self.fix_syntax_errors(file_path)

            # Remove unused imports
            self.remove_unused_imports(file_path)


def main():    base_path = "/opt/sutazaiapp"
    cleaner = ComprehensiveCodeCleaner(base_path)
    cleaner.run_comprehensive_cleanup()
    logger.info("Comprehensive code cleanup completed.")


if __name__ == "__main__":    main()
