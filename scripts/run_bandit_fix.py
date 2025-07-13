#!/usr/bin/env python3.11
"""
Automated Bandit Security Scanner and Fixer

This script:
    1. Runs the bandit scanner on the codebase
2. Saves the output to a file
3. Runs the bandit issue fixer on the output
"""

import os
import subprocess
import sys
from pathlib import Path

# ANSI color codes

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(message: str) -> None:
    """Print a formatted header message."""
    print(f"\n{Colors.HEADER}{'='*80}\n{message}\n{'='*80}{Colors.ENDC}")

def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}{message}{Colors.ENDC}")

def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}{message}{Colors.ENDC}")

def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}{message}{Colors.ENDC}")

def run_command(command: list) -> tuple:
    """Run a command and return its output and return code."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def main() -> None:
    """Main function."""
    print_header("Bandit Security Scanner and Fixer")

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Define output file
    bandit_output_file = logs_dir / "bandit_output.txt"

    # Run bandit scanner
    print("Running bandit security scanner...")

    bandit_command = [
        "bandit",
        "-r",  # Recursive
        ".",   # Current directory
        "-f",  # Format
        "txt", # Text format
        "--exclude", ".venv,venv,node_modules,build,dist",
        "--skip", "B101"  # Skip assert warnings in tests
    ]

    stdout, stderr, return_code = run_command(bandit_command)

    if return_code != 0:
        print_warning("Bandit scanner found issues. Saving output...")
    else:
        print_success("No security issues found by bandit scanner.")
        return

    # Save bandit output to file
    with open(bandit_output_file, "w", encoding="utf-8") as f:
        f.write(stdout)

    print_success(f"Saved bandit output to {bandit_output_file}")

    # Count the number of issues
    issue_count = stdout.count(">> Issue:")
    print(f"Found {issue_count} security issues.")

    # Run bandit issue fixer
    print("\nRunning bandit issue fixer...")

    fixer_command = [
        "python",
        "scripts/fix_bandit_issues.py",
        str(bandit_output_file)
    ]

    stdout, stderr, return_code = run_command(fixer_command)

    print(stdout)

    if stderr:
        print_error("Errors encountered:")
        print(stderr)

    if return_code != 0:
        print_error("Bandit issue fixer encountered errors.")
    else:
        print_success("Bandit issue fixer completed successfully.")

if __name__ == "__main__":
    main()
