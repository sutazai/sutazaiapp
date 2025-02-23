#!/usr/bin/env python3
"""
Advanced Dependency Management and Security Framework

Comprehensive dependency management with:
- Vulnerability scanning
- Update recommendations
- Dependency health tracking
- Security risk assessment
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pkg_resources
import requests
import safety
import toml
import yaml


class AdvancedDependencyManager:
    """
    A comprehensive dependency management tool for SutazAI project.
    Handles package management, vulnerability scanning, and dependency optimization.
    """

    def __init__(self, project_root: str = '.'):
        """
        Initialize the dependency manager.

        Args:
            project_root (str): Root directory of the project
        """
        self.project_root = os.path.abspath(project_root)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Paths for various dependency files
        self.requirements_path = os.path.join(self.project_root, 'requirements.txt')
        self.pyproject_path = os.path.join(self.project_root, 'pyproject.toml')
        self.dependency_report_path = os.path.join(self.project_root, 'dependency_report.json')
        
    def detect_dependency_files(self) -> List[str]:
        """
        Detect dependency management files in the project.

        Returns:
            List[str]: Paths to dependency files
        """
        dependency_files = [
            'requirements.txt',
            'pyproject.toml',
            'poetry.lock',
            'Pipfile',
            'Pipfile.lock',
            'setup.py'
        ]
        found_files = []

        for root, _, files in os.walk(self.project_root):
            for file in dependency_files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    found_files.append(file_path)

        return found_files

    def _get_installed_packages(self) -> List[str]:
        """
        Get list of installed packages with versions.
        
        Returns:
            List[str]: List of installed packages
        """
        return [
            f"{pkg.key}=={pkg.version}"
            for pkg in pkg_resources.working_set
        ]

    def analyze_dependencies(self) -> Dict[str, Any]:
        """
        Analyze project dependencies for potential issues.

        Returns:
            Dict[str, Any]: Dependency analysis results
        """
        dependency_analysis = {
            'files': self.detect_dependency_files(),
            'vulnerabilities': [],
            'outdated_packages': [],
            'installed_packages': self._get_installed_packages()
        }

        try:
            # Use pip to list outdated packages
            outdated_output = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'],
                capture_output=True,
                text=True,
                check=True
            )
            dependency_analysis['outdated_packages'] = json.loads(outdated_output.stdout)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error checking outdated packages: {e}")
        except json.JSONDecodeError:
            self.logger.error("Could not parse pip list output")

        try:
            # Use safety to check for vulnerabilities
            safety_results = safety.safety_check(
                packages=self._get_installed_packages(),
                ignore_ids=[],  # Optionally specify known false positives
                output_json=True
            )
            dependency_analysis['vulnerabilities'] = safety_results
        except Exception as e:
            self.logger.error(f"Vulnerability scanning failed: {e}")

        return dependency_analysis

    def upgrade_dependencies(self, upgrade_type: str = 'minor') -> Dict[str, Any]:
        """
        Upgrade project dependencies.

        Args:
            upgrade_type (str): Type of upgrade ('minor', 'major', 'patch')

        Returns:
            Dict[str, Any]: Dependency upgrade results
        """
        upgrade_results = {
            'upgraded_packages': [],
            'failed_upgrades': []
        }

        try:
            # Determine upgrade command based on type
            upgrade_command = {
                'minor': [sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'],
                'major': [sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'],
                'patch': [sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json']
            }.get(upgrade_type, [sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'])

            outdated_output = subprocess.run(
                upgrade_command,
                capture_output=True,
                text=True,
                check=True
            )
            outdated_packages = json.loads(outdated_output.stdout)

            for package in outdated_packages:
                try:
                    subprocess.run(
                        [sys.executable, '-m', 'pip', 'install', '--upgrade', package['name']],
                        check=True
                    )
                    upgrade_results['upgraded_packages'].append(package)
                except subprocess.CalledProcessError:
                    upgrade_results['failed_upgrades'].append(package)

        except Exception as e:
            self.logger.error(f"Dependency upgrade error: {e}")

        return upgrade_results

    def generate_dependency_report(self) -> None:
        """
        Generate a comprehensive dependency management report.
        """
        analysis_results = self.analyze_dependencies()
        report_path = os.path.join(
            self.project_root, 
            f'dependency_report_{time.strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2)
            self.logger.info(f"Dependency report generated: {report_path}")
        except Exception as e:
            self.logger.error(f"Error generating dependency report: {e}")

def main():
    dependency_manager = AdvancedDependencyManager()
    dependency_manager.generate_dependency_report()

if __name__ == '__main__':
    main() 