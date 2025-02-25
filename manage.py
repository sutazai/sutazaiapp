#!/usr/bin/env python3
# cSpell:ignore Sutaz maxfail

"""
Manage file for SutazAI project
Provides CLI commands for linting, formatting, testing, and running the application.
Usage: python manage.py [command]

Commands:
  lint        Run code linters (flake8, mypy, pylint)
  format      Run code formatter (black)
  test        Run unit tests
  run         Run the application (for development)
"""

import os
import subprocess
import sys


def run_lint():
    """
    Execute code linting using multiple tools.
    Checks code quality and potential errors.
    """
    commands = [
        ["flake8", "."],
        ["pylint", "SutazAI"],
        ["mypy", "SutazAI"],
    ]
    for cmd in commands:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def run_format():
    """
    Apply code formatting using Black.
    Ensures consistent code style across the project.
    """
    subprocess.run(["black", "."], check=True)


def run_tests():
    """
    Execute unit tests with specific configuration.
    Limits test failures and suppresses unnecessary warnings.
    """
    subprocess.run(["pytest", "--maxfail=1", "--disable-warnings", "-q"], check=True)


def run_app():
    """
    Run the primary application module for development.
    Focuses on the self-improvement core system.
    """
    subprocess.run(["python3", "SutazAI/core_system/self_improvement.py"], check=True)


def main():
    """
    Main entry point for the management script.
    Handles command-line arguments and dispatches appropriate actions.
    """
    if len(sys.argv) < 2:
        print("Usage: python manage.py [lint|format|test|run]")
        sys.exit(1)

    command = sys.argv[1]
    command_map = {
        "lint": run_lint,
        "format": run_format,
        "test": run_tests,
        "run": run_app,
    }

    action = command_map.get(command)
    if action:
        action()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
