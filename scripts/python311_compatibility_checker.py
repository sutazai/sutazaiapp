#!/usr/bin/env python3.11
"""
Python 3.11 Compatibility Checker for SutazAI Project

This script identifies potential compatibility issues specific to Python 3.11
and provides recommendations for addressing these issues.
"""

import ast
import importlib
import logging
import os
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union, Any, TypedDict, Mapping
from typing import TypedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("python311_compatibility")

# Python 3.11 specific changes to review
PY311_CHANGES = {
    "removed_modules": [
        "distutils.command.bdist_msi",  # Removed in 3.11
        "tkinter.tix",  # Removed in 3.11
    ],
    "changed_behaviors": [
        "importlib.resources",  # API changed in 3.11
        "enum",  # Flag enum behavior changed in 3.11
        "typing",  # Generic syntax changed in 3.11
        "asyncio",  # Task cancellation behavior changed in 3.11
    ],
    "new_features": [
        "tomllib",  # New in 3.11
        "typing.Self",  # New in 3.11
        "asyncio.TaskGroup",  # New in 3.11
        "exception_group",  # New in 3.11
    ],
}

KNOWN_PACKAGE_ISSUES = {
    "wrapt": "Older versions have compatibility issues with Python 3.11. Ensure version >= 1.15.0",
    "numpy": "Requires version >= 1.23.5 for Python 3.11 compatibility",
    "pandas": "Requires version >= 1.5.0 for Python 3.11 compatibility",
    "setuptools": "Some older versions have compatibility issues with Python 3.11",
    "cryptography": "Requires version >= 39.0.0 for Python 3.11 compatibility",
    "psycopg2": "Use psycopg2-binary >= 2.9.6 for Python 3.11 compatibility",
}


class StatisticsType(TypedDict):
    files_scanned: int
    files_with_issues: int
    package_issues: int


class ReportType(TypedDict):
    file_issues: Dict[str, List[str]]
    package_issues: Dict[str, List[str]]
    statistics: StatisticsType


class Python311CompatibilityChecker:
    """
    Checks Python code for compatibility with Python 3.11 and provides 
    recommendations for addressing potential issues.
    """

    def __init__(self, project_root: str):
        """
        Initialize the compatibility checker.
        
        Args:
            project_root: Root directory of the project to check
        """
        self.project_root = os.path.abspath(project_root)
        self.python_files: List[str] = []
        self.issues_found: Dict[str, List[str]] = defaultdict(list)
        self.package_issues: Dict[str, List[str]] = defaultdict(list)
        
    def find_python_files(self) -> None:
        """Find all Python files in the project."""
        logger.info(f"Finding Python files in {self.project_root}")
        for root, _, files in os.walk(self.project_root):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.project_root)
                    
                    # Skip virtual environment and other non-project files
                    if any(p in rel_path for p in ["venv/", ".venv/", "__pycache__/"]):
                        continue
                        
                    self.python_files.append(file_path)
        
        logger.info(f"Found {len(self.python_files)} Python files")
    
    def check_import_compatibility(self, file_path: str) -> None:
        """
        Check imports for compatibility issues with Python 3.11.
        
        Args:
            file_path: Path to the Python file to check
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            
            tree = ast.parse(content)
            rel_path = os.path.relpath(file_path, self.project_root)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name in PY311_CHANGES["removed_modules"]:
                            self.issues_found[rel_path].append(
                                f"Import of removed module: {name.name}"
                            )
                        elif name.name in PY311_CHANGES["changed_behaviors"]:
                            self.issues_found[rel_path].append(
                                f"Import of module with changed behavior in 3.11: {name.name}"
                            )
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in PY311_CHANGES["removed_modules"]:
                        self.issues_found[rel_path].append(
                            f"Import from removed module: {node.module}"
                        )
                    elif node.module in PY311_CHANGES["changed_behaviors"]:
                        self.issues_found[rel_path].append(
                            f"Import from module with changed behavior in 3.11: {node.module}"
                        )
        
        except Exception as e:
            rel_path = os.path.relpath(file_path, self.project_root)
            logger.error(f"Error checking file {rel_path}: {str(e)}")
            self.issues_found[rel_path].append(f"Error analyzing file: {str(e)}")
    
    def check_installed_packages(self) -> None:
        """Check installed packages for known Python 3.11 compatibility issues."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True, 
                text=True, 
                check=True,
            )
            
            installed_packages = result.stdout.strip().split("\n")
            
            for package_line in installed_packages:
                if "==" in package_line:
                    package_name, version = package_line.split("==", 1)
                    package_name = package_name.lower()
                    
                    for known_pkg, issue in KNOWN_PACKAGE_ISSUES.items():
                        if package_name == known_pkg.lower():
                            self.package_issues[package_name].append(
                                f"Version {version} - {issue}"
                            )
        
        except Exception as e:
            logger.error(f"Error checking installed packages: {str(e)}")
    
    def check_compatibility(self) -> None:
        """Run all compatibility checks."""
        logger.info("Starting Python 3.11 compatibility check")
        
        self.find_python_files()
        
        for file_path in self.python_files:
            self.check_import_compatibility(file_path)
        
        self.check_installed_packages()
        
        logger.info("Compatibility check completed")
    
    def generate_report(self) -> ReportType:
        """
        Generate a report of compatibility issues.
        
        Returns:
            Dictionary containing compatibility issues and statistics
        """
        file_count = len(self.python_files)
        files_with_issues = len(self.issues_found)
        package_issues_count = sum(len(issues) for issues in self.package_issues.values())
        
        return {
            "file_issues": dict(self.issues_found),
            "package_issues": dict(self.package_issues),
            "statistics": {
                "files_scanned": file_count,
                "files_with_issues": files_with_issues,
                "package_issues": package_issues_count,
            }
        }


def print_report(report: ReportType) -> None:
    """
    Print a human-readable report of compatibility issues.
    
    Args:
        report: Report dictionary from the compatibility checker
    """
    print("\n" + "="*80)
    print(" "*30 + "PYTHON 3.11 COMPATIBILITY REPORT")
    print("="*80)
    
    stats = report["statistics"]
    print(f"\nFiles scanned: {stats['files_scanned']}")
    print(f"Files with potential issues: {stats['files_with_issues']}")
    print(f"Package compatibility issues: {stats['package_issues']}")
    
    if report["file_issues"]:
        print("\n" + "-"*80)
        print("FILE COMPATIBILITY ISSUES")
        print("-"*80)
        
        for file_path, issues in report["file_issues"].items():
            print(f"\n{file_path}:")
            for issue in issues:
                print(f"  - {issue}")
    
    if report["package_issues"]:
        print("\n" + "-"*80)
        print("PACKAGE COMPATIBILITY ISSUES")
        print("-"*80)
        
        for package, issues in report["package_issues"].items():
            print(f"\n{package}:")
            for issue in issues:
                print(f"  - {issue}")
    
    print("\n" + "="*80)
    print(" "*25 + "RECOMMENDATIONS FOR FIXING ISSUES")
    print("="*80)
    
    if report["file_issues"] or report["package_issues"]:
        print("\n1. Update packages with known issues to their Python 3.11 compatible versions")
        print("2. Review imports from modules with changed behavior in Python 3.11")
        print("3. Replace imports of removed modules with alternatives")
        print("4. Consider using new Python 3.11 features for better performance")
        
        # Specific recommendations for common issues
        if any("asyncio" in issue for issues in report["file_issues"].values() 
               for issue in issues):
            print("\nAsyncio-specific recommendations:")
            print("  - Review task cancellation handling")
            print("  - Consider using new TaskGroup API for concurrent tasks")
        
        if any("typing" in issue for issues in report["file_issues"].values() 
               for issue in issues):
            print("\nTyping-specific recommendations:")
            print("  - Update generic type syntax")
            print("  - Consider using the new Self type for better class method typing")
    else:
        print("\nGreat! No significant Python 3.11 compatibility issues were found.")
    
    print("\n" + "="*80 + "\n")


def main() -> None:
    """Main function to run the Python 3.11 compatibility checker."""
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
    else:
        project_path = os.getcwd()
    
    checker = Python311CompatibilityChecker(project_path)
    checker.check_compatibility()
    report = checker.generate_report()
    print_report(report)


if __name__ == "__main__":
    main() 