#!/usr/bin/env python3.11
"""
Comprehensive System Diagnostics for SutazAI Project

This script provides advanced system diagnostics, comprehensive review,
and validation capabilities across the project.
"""

import ast
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import pkg_resources

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

    class SystemDiagnostics:    """
    A comprehensive system diagnostics utility for Python projects.
    """

        def __init__(self, project_root: str):        """
        Initialize the system diagnostics manager.

        Args:        project_root: Root directory of the project
        """
        self.project_root = project_root
        self.diagnostics_data: Dict[str, Any] = {}
        self.ignored_directories: Set[str] = {
        ".git", ".venv", "venv", "__pycache__",
        ".mypy_cache", ".pytest_cache", "node_modules",
        }

            def analyze_project_structure(self) -> Dict[str, Any]:            """
            Analyze the overall project structure and file organization.

            Returns:            Dictionary containing project structure details
            """
            project_structure = {
            "total_files": 0,
            "total_directories": 0,
            "file_types": {},
            "directory_structure": {},
            "large_files": [],
            }

                for root, dirs, files in os.walk(self.project_root):                # Remove ignored directories
                dirs[:] = [d for d in dirs if d not in self.ignored_directories]

                project_structure["total_directories"] += len(dirs)

                    for file in files:                    file_path = os.path.join(root, file)
                    project_structure["total_files"] += 1

                    # File type tracking
                    file_ext = os.path.splitext(file)[1]
                    project_structure["file_types"][file_ext] = \
                    project_structure["file_types"].get(file_ext, 0) + 1

                    # Large file detection
                    file_size = os.path.getsize(file_path)
                        if file_size > 1024 * 100:  # 100 KB threshold
                        project_structure["large_files"].append({
                        "path": file_path,
                        "size_kb": file_size / 1024,
                        })

                        # Create nested directory structure
                        relative_path = os.path.relpath(
                            root, self.project_root)
                        current_level = project_structure["directory_structure"]
                            for part in relative_path.split(os.path.sep):                                if part != ".":                                current_level = current_level.setdefault(
                                    part, {})

                            return project_structure

                                def validate_python_files(
                                self, max_workers: int = 4) -> Dict[str, List[str]]:                                """
                                Validate Python files for syntax, structure, and potential issues.

                                Args:                                max_workers: Maximum number of concurrent workers for validation

                                Returns:                                Dictionary of validation results for each Python file
                                """
                                validation_results = {}

                                    def validate_file(
                                        file_path: str) -> Optional[Dict[str, Any]]:                                    """
                                    Validate a single Python file.

                                    Args:                                    file_path: Path to the Python file to validate

                                    Returns:                                    Validation results for the file
                                    """
                                        try:                                            with open(file_path, encoding="utf-8") as f:                                            content = f.read()

                                            # Syntax validation
                                                try:                                                ast.parse(content)
                                                    except SyntaxError as e:                                                return {
                                                "file_path": file_path,
                                                "errors": [f"Syntax Error: {e!s}"],
                                                }

                                                # Additional validations
                                                errors = []
                                                lines = content.split("\n")

                                                # Check for overly long lines
                                                    for i, line in enumerate(
                                                        lines, 1):                                                        if len(line) > 120:                                                        errors.append(
                                                        f"Line {i} exceeds recommended length (120 chars)")

                                                        # Check for missing
                                                        # docstrings
                                                            if not any(line.strip().startswith('"""') or line.strip(
                                                            ).startswith("'''") for line in lines[:10]):                                                            errors.append(
                                                                "Missing module docstring")

                                                            # Check for
                                                            # potential
                                                            # security issues
                                                                if re.search(
                                                                    r"(eval\(|exec\()", content):                                                                errors.append(
                                                                "Potential security risk: Using eval() or exec()")

                                                                # Check for
                                                                # TODO comments
                                                                todo_comments = [
                                                                line for line in lines
                                                                    if re.search(r"#\s*TODO", line, re.IGNORECASE)
                                                                    ]
                                                                        if todo_comments:                                                                        errors.append(
                                                                        f"Contains {len(todo_comments)} TODO comments")

                                                                    return {
                                                                    "file_path": file_path,
                                                                    "errors": errors,
                                                                    } if errors else None

                                                                        except Exception as e:                                                                        logger.error(f"Error validating {file_path}: {e}")
                                                                    return {
                                                                    "file_path": file_path,
                                                                    "errors": [f"Validation Error: {e!s}"],
                                                                    }

                                                                    # Use ThreadPoolExecutor for concurrent validation
                                                                        with ThreadPoolExecutor(max_workers=max_workers) as executor:                                                                        futures = {}
                                                                            for root, _, files in os.walk(self.project_root):                                                                                for file in files:                                                                                    if file.endswith(".py"):                                                                                    file_path = os.path.join(root, file)
                                                                                    futures[executor.submit(
                                                                                    validate_file, file_path)] = file_path

                                                                                        for future in as_completed(futures):                                                                                        result = future.result()
                                                                                            if result and result.get("errors"):                                                                                            validation_results[result["file_path"]] = result["errors"]

                                                                                        return validation_results

                                                                                            def check_dependencies(self) -> Dict[str, Any]:                                                                                            """
                                                                                            Check project dependencies and their compatibility.

                                                                                            Returns:                                                                                            Dictionary of dependency information
                                                                                            """
                                                                                                try:                                                                                                dependencies = {}
                                                                                                    for package in pkg_resources.working_set:                                                                                                    dependencies[package.key] = {
                                                                                                    "version": package.version,
                                                                                                    "location": package.location,
                                                                                                    }
                                                                                                return dependencies
                                                                                                    except Exception as e:                                                                                                    logger.error(f"Error checking dependencies: {e}")
                                                                                                return {}

                                                                                                    def generate_system_report(self) -> None:                                                                                                    """
                                                                                                    Generate a comprehensive system diagnostics report.
                                                                                                    """
                                                                                                    report_path = os.path.join(
                                                                                                    self.project_root,
                                                                                                    "logs",
                                                                                                    "system_diagnostics_report.md")
                                                                                                    os.makedirs(os.path.dirname(report_path), exist_ok=True)

                                                                                                        with open(report_path, "w", encoding="utf-8") as f:                                                                                                        f.write("# System Diagnostics Report\n\n")

                                                                                                        # Project Structure Section
                                                                                                        f.write("## Project Structure\n")
                                                                                                        project_structure = self.analyze_project_structure()
                                                                                                        f.write(f"**Total Files:** {project_structure['total_files']}\n")
                                                                                                        f.write(
                                                                                                        f"**Total Directories:** {project_structure['total_directories']}\n\n")

                                                                                                        f.write("### File Types\n")
                                                                                                            for ext, count in project_structure["file_types"].items():                                                                                                            f.write(f"- **{ext or 'No Extension'}:** {count}\n")

                                                                                                            # Large Files Section
                                                                                                                if project_structure["large_files"]:                                                                                                                f.write("\n### Large Files\n")
                                                                                                                    for large_file in project_structure["large_files"]:                                                                                                                    f.write(
                                                                                                                    f"- **{large_file['path']}**: {large_file['size_kb']:.2f} KB\n")

                                                                                                                    # Python File Validation Section
                                                                                                                    f.write("\n## Python File Validation\n")
                                                                                                                    validation_results = self.validate_python_files()
                                                                                                                        if validation_results:                                                                                                                            for file_path, errors in validation_results.items():                                                                                                                            f.write(f"### {file_path}\n")
                                                                                                                            f.write("**Errors:**\n")
                                                                                                                                for error in errors:                                                                                                                                f.write(f"- {error}\n")
                                                                                                                                else:                                                                                                                                f.write("**No validation errors found.**\n")

                                                                                                                                # Dependencies Section
                                                                                                                                f.write("\n## Project Dependencies\n")
                                                                                                                                dependencies = self.check_dependencies()
                                                                                                                                    for pkg, info in dependencies.items():                                                                                                                                    f.write(
                                                                                                                                    f"- **{pkg}**: {info['version']} (Location: {info['location']})\n")

                                                                                                                                    logger.info(f"System diagnostics report generated at {report_path}")


                                                                                                                                        def main() -> None:                                                                                                                                        """Main function to run the system diagnostics."""
                                                                                                                                        project_root = "/opt/sutazaiapp"
                                                                                                                                        sys_diagnostics = SystemDiagnostics(project_root)
                                                                                                                                        sys_diagnostics.generate_system_report()


                                                                                                                                            if __name__ == "__main__":                                                                                                                                            main()

