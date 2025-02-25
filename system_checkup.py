#!/usr/bin/env python3
"""
SutazAI System Checkup Script
-----------------------------
This script performs a comprehensive health check of the SutazAI system,
identifying and fixing common issues.
"""

import ast
import importlib
import os
import pkgutil
import re
import subprocess
import sys
from pathlib import Path


class SystemCheckup:
    """System checkup and maintenance class for SutazAI."""

    def __init__(self):
        self.project_root = Path(os.path.dirname(os.path.abspath(__file__)))
        self.issues_found = []
        self.issues_fixed = []
        self.error_count = 0
        self.warning_count = 0
        self.info_count = 0

    def log_issue(self, level, message, file=None, line=None, fix_action=None):
        """Log an issue with the system."""
        issue = {
            "level": level,
            "message": message,
            "file": file,
            "line": line,
            "fix_action": fix_action,
        }
        self.issues_found.append(issue)

        if level == "ERROR":
            self.error_count += 1
            prefix = "❌ ERROR"
        elif level == "WARNING":
            self.warning_count += 1
            prefix = "⚠️ WARNING"
        else:
            self.info_count += 1
            prefix = "ℹ️ INFO"

        location = f" in {file}:{line}" if file else ""
        print(f"{prefix}{location}: {message}")

    def log_fix(self, message, file=None):
        """Log a fix that was applied."""
        fix = {"message": message, "file": file}
        self.issues_fixed.append(fix)
        location = f" in {file}" if file else ""
        print(f"✅ FIXED{location}: {message}")

    def check_syntax_errors(self):
        """Check for syntax errors in Python files."""
        print("\n🔍 Checking for syntax errors in Python files...")

        for path in self.project_root.rglob("*.py"):
            rel_path = path.relative_to(self.project_root)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                ast.parse(content)
            except SyntaxError as e:
                self.log_issue(
                    "ERROR",
                    f"Syntax error: {e.msg}",
                    file=str(rel_path),
                    line=e.lineno,
                    fix_action=f"Run 'python scripts/syntax_fixer.py {rel_path}'",
                )

    def check_import_errors(self):
        """Check for import errors in Python files."""
        print("\n🔍 Checking for import errors in Python files...")

        for path in self.project_root.rglob("*.py"):
            rel_path = path.relative_to(self.project_root)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                # Find all import statements
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        if isinstance(node, ast.Import):
                            for name in node.names:
                                module_name = name.name
                                try:
                                    importlib.import_module(module_name)
                                except ImportError:
                                    self.log_issue(
                                        "ERROR",
                                        f"Cannot import module: {module_name}",
                                        file=str(rel_path),
                                        line=node.lineno,
                                        fix_action="Run 'pip install -r requirements.txt'",
                                    )
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            try:
                                importlib.import_module(node.module)
                            except ImportError:
                                self.log_issue(
                                    "ERROR",
                                    f"Cannot import module: {node.module}",
                                    file=str(rel_path),
                                    line=node.lineno,
                                    fix_action="Run 'pip install -r requirements.txt'",
                                )
            except SyntaxError:
                # Skip files with syntax errors as they're reported by check_syntax_errors
                pass
            except Exception as e:
                self.log_issue(
                    "WARNING",
                    f"Error checking imports: {str(e)}",
                    file=str(rel_path),
                )

    def check_requirements(self):
        """Check if all required packages are installed."""
        print("\n🔍 Checking requirements installation...")

        req_files = [
            "requirements.txt",
            "requirements-integration.txt",
            "requirements-documentation.txt",
            "requirements-analysis.txt",
        ]

        for req_file in req_files:
            if not os.path.exists(os.path.join(self.project_root, req_file)):
                continue

            with open(os.path.join(self.project_root, req_file), "r") as f:
                requirements = f.readlines()

            for req in requirements:
                req = req.strip()
                if not req or req.startswith("#"):
                    continue

                # Extract package name (without version specifiers)
                package_name = re.split(r"[<>=!~]", req)[0].strip()

                try:
                    importlib.import_module(package_name)
                except ImportError:
                    self.log_issue(
                        "WARNING",
                        f"Package {package_name} from {req_file} is not installed",
                        fix_action=f"Run 'pip install -r {req_file}'",
                    )

    def check_empty_files(self):
        """Check for empty Python files."""
        print("\n🔍 Checking for empty Python files...")

        for path in self.project_root.rglob("*.py"):
            rel_path = path.relative_to(self.project_root)
            if path.stat().st_size == 0:
                self.log_issue(
                    "WARNING",
                    "File is empty",
                    file=str(rel_path),
                    fix_action="Implement the file or remove it",
                )

    def check_duplicate_files(self):
        """Check for duplicate Python files."""
        print("\n🔍 Checking for duplicate Python files...")

        file_contents = {}
        for path in self.project_root.rglob("*.py"):
            rel_path = path.relative_to(self.project_root)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                if content in file_contents:
                    self.log_issue(
                        "INFO",
                        f"Duplicate content found in: {file_contents[content]}",
                        file=str(rel_path),
                        fix_action="Consider consolidating duplicate files",
                    )
                else:
                    file_contents[content] = str(rel_path)
            except Exception:
                pass

    def fix_issues(self):
        """Attempt to fix identified issues."""
        print("\n🔧 Attempting to fix issues...")

        # Fix syntax errors using the syntax_fixer.py script
        syntax_fixer_path = os.path.join(
            self.project_root, "scripts", "syntax_fixer.py"
        )
        if os.path.exists(syntax_fixer_path):
            syntax_issues = [
                issue
                for issue in self.issues_found
                if issue["level"] == "ERROR"
                and "Syntax error" in issue["message"]
                and issue["file"]
            ]

            if syntax_issues:
                print(f"Fixing {len(syntax_issues)} syntax issues...")
                for issue in syntax_issues:
                    try:
                        file_path = os.path.join(
                            self.project_root, issue["file"]
                        )
                        subprocess.run(
                            [sys.executable, syntax_fixer_path, file_path],
                            check=True,
                            capture_output=True,
                        )
                        self.log_fix("Fixed syntax issue", file=issue["file"])
                    except subprocess.CalledProcessError:
                        print(f"Failed to fix syntax issue in {issue['file']}")

        # Fix missing dependencies
        missing_deps = [
            issue
            for issue in self.issues_found
            if issue["level"] in ("ERROR", "WARNING")
            and "Cannot import module" in issue["message"]
        ]

        if missing_deps:
            # Get unique packages
            packages = set()
            for issue in missing_deps:
                match = re.search(
                    r"Cannot import module: ([\w\.]+)", issue["message"]
                )
                if match:
                    packages.add(match.group(1))

            if packages:
                print(f"Installing {len(packages)} missing packages...")
                for package in packages:
                    try:
                        subprocess.run(
                            [sys.executable, "-m", "pip", "install", package],
                            check=True,
                            capture_output=True,
                        )
                        self.log_fix(f"Installed package: {package}")
                    except subprocess.CalledProcessError:
                        print(f"Failed to install package: {package}")

    def run_full_checkup(self):
        """Run a full system checkup."""
        print("🚀 Starting SutazAI system checkup...")

        self.check_syntax_errors()
        self.check_import_errors()
        self.check_requirements()
        self.check_empty_files()
        self.check_duplicate_files()

        print("\n📊 Checkup Summary:")
        print(f"- Errors: {self.error_count}")
        print(f"- Warnings: {self.warning_count}")
        print(f"- Info: {self.info_count}")

        if self.error_count > 0 or self.warning_count > 0:
            self.fix_issues()

            print("\n📝 Final Report:")
            print(f"- Issues found: {len(self.issues_found)}")
            print(f"- Issues fixed: {len(self.issues_fixed)}")

            if self.issues_fixed:
                print("\nFixed issues:")
                for fix in self.issues_fixed:
                    file_info = f" in {fix['file']}" if fix.get("file") else ""
                    print(f"- {fix['message']}{file_info}")

            if len(self.issues_found) > len(self.issues_fixed):
                print("\nRemaining issues that need manual intervention:")
                for issue in self.issues_found:
                    if not any(
                        fix.get("file") == issue.get("file")
                        for fix in self.issues_fixed
                    ):
                        file_info = (
                            f" in {issue['file']}" if issue.get("file") else ""
                        )
                        line_info = (
                            f" at line {issue['line']}"
                            if issue.get("line")
                            else ""
                        )
                        print(
                            f"- {issue['level']}: {issue['message']}{file_info}{line_info}"
                        )
                        if issue.get("fix_action"):
                            print(f"  Fix: {issue['fix_action']}")
        else:
            print("\n✨ No issues found. System is in good health!")

        return self.error_count == 0 and self.warning_count == 0


if __name__ == "__main__":
    checkup = SystemCheckup()
    success = checkup.run_full_checkup()
    sys.exit(0 if success else 1)
