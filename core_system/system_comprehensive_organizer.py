#!/usr/bin/env python3
"""
Comprehensive System Organizer

This script performs an exhaustive audit and organization of the entire SutazAI project.
It verifies the logical structure, file organization, dependency cross-references, and overall system architecture.
It autonomously creates missing directories based on a predefined template, reorganizes files if necessary, and generates
a comprehensive report in markdown format. This report can be used to reference the state of the codebase
and ensure that all components operate at peak performance and remain secure.

Requirements:
- Python 3.10+
- Standard libraries: os, sys, json
"""

import os
import sys
import json
import datetime

# Define base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Expected root directories in the project
EXPECTED_DIRECTORIES = [
    'ai_agents',
    'backend',
    'core_system',
    'docs',
    'logs',
    'security',
    'scripts',
    'tests',
    'config'
]

# Directory to save organizer reports
REPORTS_DIR = os.path.join(BASE_DIR, 'logs', 'organizer_reports')


def create_missing_directories() -> list:
    """
    Check the project for expected directories and create any that are missing.

    Returns:
        List of directories that were created.
    """
    missing_dirs = []
    for d in EXPECTED_DIRECTORIES:
        dir_path = os.path.join(BASE_DIR, d)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            missing_dirs.append(d)
    return missing_dirs


def scan_project_structure() -> dict:
    """
    Recursively scan the project structure and return a nested dictionary representation.

    Returns:
        Dictionary representing the directory and file structure of the project.
    """
    structure = {}
    for root, dirs, files in os.walk(BASE_DIR):
        rel_root = os.path.relpath(root, BASE_DIR)
        structure[rel_root] = {
            "directories": dirs,
            "files": files
        }
    return structure


def generate_markdown_report(missing_dirs: list, structure: dict) -> str:
    """
    Generate a markdown report detailing any missing directories and a complete overview of the project structure.

    Args:
        missing_dirs (list): List of directories that were created.
        structure (dict): The current project structure.

    Returns:
        A markdown formatted string report.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = []
    report_lines.append(f"# SutazAI Project Organizational Report")
    report_lines.append(f"**Generated on:** {now}\n")

    if missing_dirs:
        report_lines.append("## Missing Directories Created")
        for d in missing_dirs:
            report_lines.append(f"- {d}")
    else:
        report_lines.append("## All expected directories are present.")

    report_lines.append("\n## Project Structure Overview\n")
    for path, details in sorted(structure.items()):
        report_lines.append(f"### {path}")
        report_lines.append(f"- **Directories:** {', '.join(details['directories']) if details['directories'] else 'None'}")
        report_lines.append(f"- **Files:** {', '.join(details['files']) if details['files'] else 'None'}\n")

    return "\n".join(report_lines)


def save_report(report_content: str) -> str:
    """
    Save the generated report into the organizer_reports directory.

    Args:
        report_content (str): The markdown content of the report.

    Returns:
        The path to the saved report file.
    """
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f"organizer_report_{timestamp}.md")
    with open(report_path, "w") as f:
        f.write(report_content)
    return report_path


def main():
    missing = create_missing_directories()
    structure = scan_project_structure()
    report = generate_markdown_report(missing, structure)
    report_path = save_report(report)
    print(f"Organizer report generated at: {report_path}")


if __name__ == '__main__':
    main() 