#!/usr/bin/env python3.11
"""
SutazAI Comprehensive System Review Script

This script performs an in-depth analysis of the entire SutazAI application,
examining code quality, performance, security, and system architecture.
"""

import ast
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from misc.utils.subprocess_utils import run_command, validate_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(
            "/opt/sutazaiapp/logs/comprehensive_system_review.log",
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("SutazAI.SystemReview")


class ComprehensiveSystemReviewer:
    """Comprehensive system review framework"""

    def __init__(self, base_path: str = "/opt/sutazaiapp"):
        self.base_path = base_path
        self.log_dir = os.path.join(base_path, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def find_python_files(self) -> List[str]:
        """Find all Python files in project"""
        python_files = []
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    if validate_path(full_path):
                        python_files.append(full_path)
        return python_files

    def analyze_code_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Perform a detailed analysis of a Python file's structure.

        Args:
            file_path: Path to the Python file

        Returns:
            Dictionary containing code structure details
        """
        if not validate_path(file_path):
            logger.error(f"Invalid file path: {file_path}")
            return {}

        try:
            with open(file_path) as f:
                source = f.read()

            # Parse the source code
            tree = ast.parse(source)

            # Analyze code structure
            structure = {
                "filename": os.path.basename(file_path),
                "classes": [],
                "functions": [],
                "imports": [],
                "complexity": {
                    "total_lines": len(source.splitlines()),
                    "class_count": 0,
                    "function_count": 0,
                    "import_count": 0,
                },
            }

            # Traverse the AST
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "methods": [method.name for method in node.body if isinstance(method, ast.FunctionDef)],
                    }
                    structure["classes"].append(class_info)
                    structure["complexity"]["class_count"] += 1

                elif isinstance(node, ast.FunctionDef):
                    structure["functions"].append(
                        {
                            "name": node.name,
                            "args": [arg.arg for arg in node.args.args],
                            "line_number": node.lineno,
                        },
                    )
                    structure["complexity"]["function_count"] += 1

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        imports = [alias.name for alias in node.names]
                    else:
                        imports = [f"{node.module}.{alias.name}" for alias in node.names]
                    structure["imports"].extend(imports)
                    structure["complexity"]["import_count"] += len(imports)

            return structure

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {}

    def detect_code_smells(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Detect potential code smells and anti-patterns.

        Args:
            file_path: Path to the Python file

        Returns:
            List of detected code smells
        """
        if not validate_path(file_path):
            logger.error(f"Invalid file path: {file_path}")
            return []

        code_smells = []

        try:
            with open(file_path) as f:
                source_lines = f.readlines()

            # Detect long methods/functions
            for i, line in enumerate(source_lines):
                if "def " in line:
                    method_lines = 0
                    for j in range(i + 1, len(source_lines)):
                        if source_lines[j].strip() and not source_lines[j].startswith(" "):
                            break
                        method_lines += 1

                    if method_lines > 50:  # Arbitrary threshold
                        code_smells.append(
                            {
                                "type": "long_method",
                                "line": i + 1,
                                "description": (f"Method is too long ({method_lines} lines)"),
                            },
                        )

            # Detect duplicate code
            source_code = "".join(source_lines)
            duplicates = self._find_duplicate_code(source_code)
            code_smells.extend(duplicates)

            # Detect overly complex conditionals
            conditionals = self._detect_complex_conditionals(source_code)
            code_smells.extend(conditionals)

        except Exception as e:
            logger.error(f"Error detecting code smells in {file_path}: {e}")

        return code_smells

    def _find_duplicate_code(self, source_code: str) -> List[Dict[str, Any]]:
        """
        Find duplicate code segments.

        Args:
            source_code: Full source code as a string

        Returns:
            List of duplicate code segments
        """
        duplicates = []
        lines = source_code.split("\n")
        line_count = len(lines)

        for i in range(line_count):
            # Look for duplicates at least 5 lines apart
            for j in range(i + 5, line_count):
                # Compare 3-line segments
                segment_length = 3
                if j + segment_length > line_count:
                    break

                segment1 = lines[i : i + segment_length]
                segment2 = lines[j : j + segment_length]

                # Remove whitespace and comments for comparison
                clean_segment1 = [re.sub(r"#.*$", "", line).strip() for line in segment1]
                clean_segment2 = [re.sub(r"#.*$", "", line).strip() for line in segment2]

                if clean_segment1 == clean_segment2 and any(clean_segment1):
                    duplicates.append(
                        {
                            "type": "duplicate_code",
                            "first_occurrence": i + 1,
                            "second_occurrence": j + 1,
                            "segment": segment1,
                        },
                    )

        return duplicates

    def _detect_complex_conditionals(self, source_code: str) -> List[Dict[str, Any]]:
        """
        Detect overly complex conditional statements.

        Args:
            source_code: Full source code as a string

        Returns:
            List of complex conditional statements
        """
        complex_conditionals = []

        # Regex to find complex if/elif/else statements
        complex_if_pattern = re.compile(
            r"(if|elif)\s*\(.*\band\b.*\bor\b.*\):|(if|elif)\s*\(.*\bor\b.*\band\b.*\):",
        )

        for match in complex_if_pattern.finditer(source_code):
            complex_conditionals.append(
                {
                    "type": "complex_conditional",
                    "line": source_code[: match.start()].count("\n") + 1,
                    "description": ("Complex conditional with multiple AND/OR operators"),
                },
            )

        return complex_conditionals

    def run_linters(self, file_path: str) -> Dict[str, Any]:
        """
        Run various linters on a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            Dictionary containing linter results
        """
        if not validate_path(file_path):
            logger.error(f"Invalid file path: {file_path}")
            return {"error": "Invalid file path"}

        linter_results = {
            "flake8": [],
            "pylint": [],
            "mypy": [],
        }

        try:
            # Run flake8
            flake8_result = run_command(
                ["flake8", "--format=json", file_path],
                cwd=self.base_path,
            )
            if flake8_result.stdout:
                try:
                    linter_results["flake8"] = json.loads(flake8_result.stdout)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse flake8 output: {e}")

            # Run pylint
            pylint_result = run_command(
                ["pylint", "--output-format=json", file_path],
                cwd=self.base_path,
            )
            if pylint_result.stdout:
                try:
                    linter_results["pylint"] = json.loads(pylint_result.stdout)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse pylint output: {e}")

            # Run mypy
            mypy_result = run_command(
                ["mypy", "--show-error-codes", "--no-error-summary", file_path],
                cwd=self.base_path,
            )
            if mypy_result.stdout:
                linter_results["mypy"] = mypy_result.stdout.splitlines()

        except Exception as e:
            logger.error(f"Error running linters on {file_path}: {e}")

        return linter_results

    def comprehensive_system_review(self) -> Dict[str, Any]:
        """
        Perform a comprehensive system review.

        Returns:
            Dictionary containing review results
        """
        review_report = {
            "timestamp": datetime.now().isoformat(),
            "files_analyzed": 0,
            "total_issues": 0,
            "code_structure": {},
            "code_smells": {},
            "linter_results": {},
        }

        python_files = self.find_python_files()
        review_report["files_analyzed"] = len(python_files)

        for file_path in python_files:
            logger.info(f"Analyzing {file_path}...")

            # Code structure analysis
            structure = self.analyze_code_structure(file_path)
            review_report["code_structure"][file_path] = structure

            # Code smell detection
            smells = self.detect_code_smells(file_path)
            review_report["code_smells"][file_path] = smells
            review_report["total_issues"] += len(smells)

            # Linter analysis
            linter_results = self.run_linters(file_path)
            review_report["linter_results"][file_path] = linter_results
            review_report["total_issues"] += sum(len(results) for results in linter_results.values())

        return review_report

    def generate_summary_report(self, review_report: Dict[str, Any]) -> None:
        """
        Generate a summary report of the system review.

        Args:
            review_report: Dictionary containing review results
        """
        summary = {
            "timestamp": review_report["timestamp"],
            "files_analyzed": review_report["files_analyzed"],
            "total_issues": review_report["total_issues"],
            "summary_by_category": {
                "code_structure": {
                    "total_classes": sum(
                        len(structure["classes"]) for structure in review_report["code_structure"].values()
                    ),
                    "total_functions": sum(
                        len(structure["functions"]) for structure in review_report["code_structure"].values()
                    ),
                },
                "code_smells": {
                    "total_smells": sum(len(smells) for smells in review_report["code_smells"].values()),
                    "by_type": {},
                },
                "linter_results": {
                    "total_linter_issues": sum(
                        sum(len(results) for results in file_results.values())
                        for file_results in review_report["linter_results"].values()
                    ),
                    "by_linter": {
                        "flake8": 0,
                        "pylint": 0,
                        "mypy": 0,
                    },
                },
            },
        }

        # Count code smells by type
        for file_smells in review_report["code_smells"].values():
            for smell in file_smells:
                smell_type = smell["type"]
                if smell_type not in summary["summary_by_category"]["code_smells"]["by_type"]:
                    summary["summary_by_category"]["code_smells"]["by_type"][smell_type] = 0
                summary["summary_by_category"]["code_smells"]["by_type"][smell_type] += 1

        # Count linter issues by type
        for file_results in review_report["linter_results"].values():
            for linter, issues in file_results.items():
                summary["summary_by_category"]["linter_results"]["by_linter"][linter] += len(issues)

        # Save summary report
        summary_path = os.path.join(
            self.log_dir,
            f"system_review_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary report saved to {summary_path}")

    def main(self) -> None:
        """
        Main execution method for the system review.
        """
        logger.info("Starting comprehensive system review...")
        review_report = self.comprehensive_system_review()
        self.generate_summary_report(review_report)
        logger.info("System review completed.")


def main():
    """
    Main entry point for the script.
    """
    reviewer = ComprehensiveSystemReviewer()
    reviewer.main()


if __name__ == "__main__":
    main()
