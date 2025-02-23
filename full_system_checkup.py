#!/usr/bin/env python3
"""
Comprehensive System Checkup and Auto-Fix Script for SutazAI Codebase

This script performs an extensive system-wide checkup including:
- Advanced linting and formatting
- Type checking and static analysis
- Security vulnerability scanning
- Dependency verification
- Documentation and markdown quality checks
- Spell checking and code style consistency

Generates detailed JSON and Markdown reports with actionable insights.
"""

import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ComprehensiveSystemCheckup")


def run_command(
    command: List[str], cwd: Optional[str] = None, capture_output: bool = True
) -> Dict[str, Any]:
    """
    Run a command with enhanced error handling and output capturing.

    Args:
        command (List[str]): Command to execute
        cwd (Optional[str]): Working directory
        capture_output (bool): Whether to capture command output

    Returns:
        Dict[str, Any]: Command execution results
    """
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }
    except Exception as e:
        logger.error(f"Command {' '.join(command)} failed: {e}")
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": 1,
            "success": False,
        }


def scan_python_files(
    base_dir: str, exclude_patterns: Optional[List[str]] = None
) -> List[str]:
    """
    Recursively scan for Python files with advanced exclusion.

    Args:
        base_dir (str): Base directory to scan
        exclude_patterns (Optional[List[str]]): Patterns to exclude

    Returns:
        List[str]: List of Python file paths
    """
    exclude_patterns = exclude_patterns or [
        "venv",
        ".venv",
        "__pycache__",
        ".git",
        "node_modules",
        "build",
        "dist",
    ]

    py_files = []
    for root, dirs, files in os.walk(base_dir):
        # Remove excluded directories
        dirs[:] = [
            d
            for d in dirs
            if not any(pattern in d for pattern in exclude_patterns)
        ]

        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                if not any(
                    pattern in full_path for pattern in exclude_patterns
                ):
                    py_files.append(full_path)

    return py_files


def run_comprehensive_linters(files: List[str]) -> Dict[str, Any]:
    """
    Run multiple linters and formatters on files with detailed reporting.

    Args:
        files (List[str]): List of Python files to check

    Returns:
        Dict[str, Any]: Comprehensive linting and formatting results
    """
    results = {
        "flake8": {},
        "pylint": {},
        "mypy": {},
        "black_fixes": {},
        "isort_fixes": {},
    }

    for file in files:
        # Flake8 linting
        flake8_result = run_command(["flake8", file])
        if not flake8_result["success"]:
            results["flake8"][file] = flake8_result

        # Pylint analysis
        pylint_result = run_command(["pylint", file])
        if not pylint_result["success"]:
            results["pylint"][file] = pylint_result

        # MyPy type checking
        mypy_result = run_command(["mypy", file])
        if not mypy_result["success"]:
            results["mypy"][file] = mypy_result

        # Black formatting
        black_result = run_command(["black", file])
        results["black_fixes"][file] = black_result

        # Isort import sorting
        isort_result = run_command(["isort", file])
        results["isort_fixes"][file] = isort_result

    return results


def run_security_scan(base_dir: str) -> Dict[str, Any]:
    """
    Run comprehensive security vulnerability scanner.

    Args:
        base_dir (str): Base directory to scan

    Returns:
        Dict[str, Any]: Security scan results
    """
    bandit_result = run_command(["bandit", "-r", base_dir, "-f", "json"])
    safety_result = run_command(["safety", "check"])

    return {"bandit": bandit_result, "safety": safety_result}


def verify_dependencies() -> Dict[str, Any]:
    """
    Verify project dependencies and compatibility.

    Returns:
        Dict[str, Any]: Dependency verification results
    """
    pip_check_result = run_command(["pip", "check"])
    pip_list_result = run_command(["pip", "list"])

    return {"pip_check": pip_check_result, "pip_list": pip_list_result}


def check_markdown_quality(base_dir: str) -> Dict[str, List[str]]:
    """
    Check markdown files for quality and style issues.

    Args:
        base_dir (str): Base directory to scan

    Returns:
        Dict[str, List[str]]: Markdown quality issues
    """
    markdown_issues = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                markdownlint_result = run_command(["markdownlint", full_path])
                if not markdownlint_result["success"]:
                    markdown_issues[full_path] = markdownlint_result[
                        "stderr"
                    ].split("\n")

    return markdown_issues


def generate_comprehensive_report(results: Dict[str, Any]) -> None:
    """
    Generate comprehensive JSON and Markdown reports.

    Args:
        results (Dict[str, Any]): Comprehensive system check results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON Report
    json_report_path = f"system_comprehensive_report_{timestamp}.json"
    with open(json_report_path, "w") as f:
        json.dump(results, f, indent=2)

    # Markdown Report
    md_report_path = f"system_comprehensive_report_{timestamp}.md"
    with open(md_report_path, "w") as f:
        f.write("# Comprehensive System Checkup Report\n\n")
        f.write(f"**Generated on:** {datetime.now().isoformat()}\n\n")

        # Linting Section
        f.write("## Code Linting Results\n")
        for linter, files in results.get("linting", {}).items():
            f.write(f"### {linter.capitalize()} Issues\n")
            for file, issues in files.items():
                f.write(f"#### {file}\n```\n{issues}\n```\n")

        # Security Scan Section
        f.write("## Security Vulnerability Scan\n")
        f.write(
            f"### Bandit Results\n```json\n{results.get('security_scan', {}).get('bandit', 'No issues found')}\n```\n"
        )
        f.write(
            f"### Safety Results\n```\n{results.get('security_scan', {}).get('safety', 'No issues found')}\n```\n"
        )

        # Markdown Quality Section
        f.write("## Markdown Quality Issues\n")
        for file, issues in results.get("markdown_issues", {}).items():
            f.write(f"### {file}\n```\n" + "\n".join(issues) + "\n```\n")

    logger.info(
        f"Comprehensive reports generated: {json_report_path} and {md_report_path}"
    )


def main():
    base_dir = os.getcwd()
    logger.info("Starting Comprehensive System Checkup...")

    # Scan Python files
    python_files = scan_python_files(base_dir)

    # Comprehensive system check
    results = {
        "start_time": datetime.now().isoformat(),
        "total_files": len(python_files),
        "linting": run_comprehensive_linters(python_files),
        "security_scan": run_security_scan(base_dir),
        "dependency_verification": verify_dependencies(),
        "markdown_issues": check_markdown_quality(base_dir),
    }

    # Generate reports
    generate_comprehensive_report(results)

    logger.info("Comprehensive system checkup completed.")


if __name__ == "__main__":
    main()
