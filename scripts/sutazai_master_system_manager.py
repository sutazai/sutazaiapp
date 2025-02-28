#!/usr/bin/env python3.11
"""
SutazAI Master System Management Script

A comprehensive, all-in-one solution for system diagnostics,
optimization, and maintenance.
"""

import argparse
import ast
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from misc.utils.subprocess_utils import run_command, run_python_module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("/opt/sutazaiapp/logs/master_system_manager.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("SutazAI.MasterSystemManager")


class SutazAIMasterSystemManager:
    """
    Comprehensive system management class with multiple capabilities.
    """

    def __init__(self, base_path: str = "/opt/sutazaiapp"):
        """
        Initialize the master system manager.

        Args:
            base_path: Base directory of the SutazAI project
        """
        self.base_path = base_path
        self.log_dir = os.path.join(base_path, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def verify_python_version(self) -> bool:
        """
        Verify Python 3.11 compatibility.

        Returns:
            Whether the current Python version is 3.11
        """
        major, minor = sys.version_info.major, sys.version_info.minor
        if major != 3 or minor != 11:
            logger.error(f"Unsupported Python version. Required: 3.11, Current: {major}.{minor}")
            return False
        return True

    def find_python_files(self) -> List[str]:
        """
        Recursively find all Python files in the project.

        Returns:
            List of Python file paths
        """
        python_files = []
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files

    def detect_syntax_errors(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Detect syntax errors in a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            List of detected syntax errors
        """
        errors = []
        try:
            with open(file_path) as f:
                source = f.read()
            ast.parse(source)
        except SyntaxError as e:
            errors.append({"line": e.lineno, "offset": e.offset, "text": e.text, "msg": str(e)})
        return errors

    def fix_import_statements(self, file_path: str) -> bool:
        """
        Fix common import statement issues.

        Args:
            file_path: Path to the Python file

        Returns:
            Whether changes were made
        """
        try:
            with open(file_path) as f:
                content = f.read()

            # Fix common import issues
            content = re.sub(
                r"import\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*)",
                r"import \1\nimport \2",
                content,
            )

            # Remove duplicate imports
            imports = {}
            new_lines = []
            for line in content.split("\n"):
                if line.startswith("import ") or line.startswith("from "):
                    if line not in imports:
                        imports[line] = True
                        new_lines.append(line)
                else:
                    new_lines.append(line)

            new_content = "\n".join(new_lines)

            if new_content != content:
                with open(file_path, "w") as f:
                    f.write(new_content)
                return True
            return False

        except Exception as e:
            logger.error(f"Error fixing imports in {file_path}: {e}")
            return False

    def run_linters(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Run linters on a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            List of linter issues
        """
        try:
            # Run pylint
            pylint_result = run_python_module("pylint", [file_path], check=False)

            # Run mypy
            mypy_result = run_python_module("mypy", [file_path], check=False)

            issues = []

            # Parse pylint output
            for line in pylint_result.stdout.split("\n"):
                if ":" in line:
                    parts = line.split(":")
                    if len(parts) >= 3:
                        issues.append(
                            {
                                "type": "pylint",
                                "line": parts[1],
                                "message": ":".join(parts[2:]).strip(),
                            },
                        )

            # Parse mypy output
            for line in mypy_result.stdout.split("\n"):
                if ":" in line:
                    parts = line.split(":")
                    if len(parts) >= 3:
                        issues.append(
                            {
                                "type": "mypy",
                                "line": parts[1],
                                "message": ":".join(parts[2:]).strip(),
                            },
                        )

            return issues

        except Exception as e:
            logger.error(f"Error running linters on {file_path}: {e}")
            return []

    def optimize_dependencies(self) -> Dict[str, Any]:
        """
        Optimize and check project dependencies.

        Returns:
            Dependency optimization report
        """
        try:
            # Run pip list to get current dependencies
            pip_list = run_python_module("pip", ["list", "--format=json"], check=False)
            dependencies = json.loads(pip_list.stdout)

            # Run safety check for vulnerabilities
            safety_result = run_command(
                ["safety", "check", "--full-report"],
                check=False,
            )

            return {
                "installed_packages": dependencies,
                "vulnerabilities": safety_result.stdout,
            }

        except Exception as e:
            logger.error(f"Dependency optimization failed: {e}")
            return {}

    def performance_diagnostics(self) -> Dict[str, Any]:
        """
        Collect system performance diagnostics.

        Returns:
            Performance metrics dictionary
        """
        try:
            import psutil

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()

            # Memory metrics
            memory = psutil.virtual_memory()

            # Disk metrics
            disk = psutil.disk_usage("/")

            return {
                "cpu": {
                    "percent": cpu_percent,
                    "frequency": {
                        "current": cpu_freq.current,
                        "min": cpu_freq.min,
                        "max": cpu_freq.max,
                    },
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent,
                },
            }

        except Exception as e:
            logger.error(f"Performance diagnostics failed: {e}")
            return {}

    def comprehensive_system_check(self) -> Dict[str, Any]:
        """
        Run a comprehensive system check.

        Returns:
            System check report
        """
        if not self.verify_python_version():
            logger.warning("Python version check failed")

        report = {
            "timestamp": datetime.now().isoformat(),
            "python_files": [],
            "syntax_errors": [],
            "linter_issues": [],
            "dependencies": self.optimize_dependencies(),
            "performance": self.performance_diagnostics(),
        }

        python_files = self.find_python_files()
        report["python_files"] = python_files

        for file_path in python_files:
            # Check for syntax errors
            syntax_errors = self.detect_syntax_errors(file_path)
            if syntax_errors:
                report["syntax_errors"].extend([{"file": file_path, "error": error} for error in syntax_errors])

            # Fix import statements
            if self.fix_import_statements(file_path):
                logger.info(f"Fixed imports in {file_path}")

            # Run linters
            linter_issues = self.run_linters(file_path)
            if linter_issues:
                report["linter_issues"].extend([{"file": file_path, "issue": issue} for issue in linter_issues])

        return report

    def auto_repair(self, report: Optional[Dict[str, Any]] = None) -> None:
        """
        Attempt to automatically repair detected issues.

        Args:
            report: Optional system check report
        """
        if report is None:
            report = self.comprehensive_system_check()

        # Fix syntax errors
        for error in report["syntax_errors"]:
            file_path = error["file"]
            logger.info(f"Attempting to fix syntax errors in {file_path}")
            try:
                with open(file_path) as f:
                    content = f.read()

                # Basic syntax fixes
                content = re.sub(r"\t", "    ", content)  # Replace tabs with spaces
                content = re.sub(r"\r\n", "\n", content)  # Normalize line endings

                with open(file_path, "w") as f:
                    f.write(content)
            except Exception as e:
                logger.error(f"Error fixing {file_path}: {e}")

        # Fix linter issues
        for issue in report["linter_issues"]:
            file_path = issue["file"]
            logger.info(f"Attempting to fix linter issues in {file_path}")
            self.fix_import_statements(file_path)

    def main_menu(self):
        """Display interactive main menu"""
        while True:
            print("\nSutazAI Master System Manager")
            print("1. Run Comprehensive System Check")
            print("2. Auto-repair Issues")
            print("3. View Performance Diagnostics")
            print("4. Check Dependencies")
            print("5. Exit")

            choice = input("\nEnter your choice (1-5): ")

            if choice == "1":
                report = self.comprehensive_system_check()
                print("\nSystem Check Report:")
                print(json.dumps(report, indent=2))
            elif choice == "2":
                self.auto_repair()
                print("\nAuto-repair completed.")
            elif choice == "3":
                metrics = self.performance_diagnostics()
                print("\nPerformance Metrics:")
                print(json.dumps(metrics, indent=2))
            elif choice == "4":
                deps = self.optimize_dependencies()
                print("\nDependency Report:")
                print(json.dumps(deps, indent=2))
            elif choice == "5":
                print("\nExiting...")
                break
            else:
                print("\nInvalid choice. Please try again.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SutazAI Master System Manager")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with menu",
    )
    args = parser.parse_args()

    manager = SutazAIMasterSystemManager()

    if args.interactive:
        manager.main_menu()
    else:
        report = manager.comprehensive_system_check()
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
