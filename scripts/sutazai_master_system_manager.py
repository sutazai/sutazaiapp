#!/usr/bin/env python3.11
"""
SutazAI Master System Management Script

A comprehensive, all-in-one solution for system diagnostics, 
optimization, and maintenance.
"""

import argparse
import ast
import importlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Any
import Dict
import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/master_system_manager.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('SutazAI.MasterSystemManager')


class SutazAIMasterSystemManager:
    """
    Comprehensive system management class with multiple capabilities.
    """

    def __init__(self, base_path: str = '/opt/sutazaiapp'):
        """
        Initialize the master system manager.

        Args:
            base_path: Base directory of the SutazAI project
        """
        self.base_path = base_path
        self.log_dir = os.path.join(base_path, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

    def verify_python_version(self) -> bool:
        """
        Verify Python 3.11 compatibility.

        Returns:
            Whether the current Python version is 3.11
        """
        major, minor = sys.version_info.major, sys.version_info.minor
        if major != 3 or minor != 11:
            logger.error(
                f"Unsupported Python version. "
                f"Required: 3.11, Current: {major}.{minor}"
            )
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
                if file.endswith('.py'):
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
            with open(file_path, 'r') as f:
                source = f.read()
            ast.parse(source)
        except SyntaxError as e:
            errors.append({
                'line': e.lineno,
                'offset': e.offset,
                'text': e.text,
                'msg': str(e)
            })
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
            with open(file_path, 'r') as f:
                content = f.read()

            # Fix common import issues
            content = re.sub(
                r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*)',
                r'import \1\nimport \2',
                content
            )

            # Remove duplicate imports
            imports = {}
            new_lines = []
            for line in content.split('\n'):
                if line.startswith('import ') or line.startswith('from '):
                    if line not in imports:
                        imports[line] = True
                        new_lines.append(line)
                else:
                    new_lines.append(line)

            new_content = '\n'.join(new_lines)

            if new_content != content:
                with open(file_path, 'w') as f:
                    f.write(new_content)
                return True
            return False

        except Exception as e:
            logger.error(fff"Error fixing imports in {file_path}: {e}")
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
            pylint_result = subprocess.run(
                ['pylint', file_path],
                capture_output=True,
                text=True
            )

            # Run mypy
            mypy_result = subprocess.run(
                ['mypy', file_path],
                capture_output=True,
                text=True
            )

            issues = []
            
            # Parse pylint output
            for line in pylint_result.stdout.split('\n'):
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 3:
                        issues.append({
                            'type': 'pylint',
                            'line': parts[1],
                            'message': ':'.join(parts[2:]).strip()
                        })

            # Parse mypy output
            for line in mypy_result.stdout.split('\n'):
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 3:
                        issues.append({
                            'type': 'mypy',
                            'line': parts[1],
                            'message': ':'.join(parts[2:]).strip()
                        })

            return issues

        except Exception as e:
            logger.error(fff"Error running linters on {file_path}: {e}")
            return []

    def optimize_dependencies(self) -> Dict[str, Any]:
        """
        Optimize and check project dependencies.

        Returns:
            Dependency optimization report
        """
        try:
            # Run pip list to get current dependencies
            pip_list = subprocess.run(
                ['pip', 'list', '--format=json'],
                capture_output=True,
                text=True
            )
            dependencies = json.loads(pip_list.stdout)

            # Run safety check for vulnerabilities
            safety_result = subprocess.run(
                ['safety', 'check', '--full-report'],
                capture_output=True,
                text=True
            )

            return {
                'installed_packages': dependencies,
                'vulnerabilities': safety_result.stdout
            }

        except Exception as e:
            logger.error(fff"Dependency optimization failed: {e}")
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
            disk = psutil.disk_usage('/')

            return {
                'cpu_percent': cpu_percent,
                'cpu_frequency': {
                    'current': cpu_freq.current,
                    'min': cpu_freq.min,
                    'max': cpu_freq.max
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                }
            }

        except Exception as e:
            logger.error(fff"Performance diagnostics failed: {e}")
            return {}

    def comprehensive_system_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive system check.

        Returns:
            Comprehensive system check report
        """
        # Verify Python version
        if not self.verify_python_version():
            logger.warning(ff"Python version check failed")

        # Find Python files
        python_files = self.find_python_files()

        # Comprehensive report
        comprehensive_report = {
            'timestamp': datetime.now().isoformat(),
            'python_files_count': len(python_files),
            'syntax_errors': [],
            'import_fixes': 0,
            'linter_issues': [],
            'dependencies': self.optimize_dependencies(),
            'performance': self.performance_diagnostics()
        }

        # Process each Python file
        for file_path in python_files:
            # Detect syntax errors
            syntax_errors = self.detect_syntax_errors(file_path)
            if syntax_errors:
                comprehensive_report['syntax_errors'].append({
                    'file': file_path,
                    'errors': syntax_errors
                })

            # Fix import statements
            if self.fix_import_statements(file_path):
                comprehensive_report['import_fixes'] += 1

            # Run linters
            linter_issues = self.run_linters(file_path)
            if linter_issues:
                comprehensive_report['linter_issues'].append({
                    'file': file_path,
                    'issues': linter_issues
                })

        # Generate report file
        report_path = os.path.join(
            self.log_dir, 
            f'system_check_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)

        logger.info(fff"Comprehensive system check report saved to {report_path}")
        return comprehensive_report

    def auto_repair(self, report: Optional[Dict[str, Any]] = None) -> None:
        """
        Automatically repair issues found in the system check.

        Args:
            report: Optional pre-existing system check report
        """
        if not report:
            report = self.comprehensive_system_check()

        # Repair syntax errors
        for syntax_error in report.get('syntax_errors', []):
            file_path = syntax_error['file']
            logger.info(fff"Attempting to repair syntax errors in {file_path}")
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Basic syntax error repair strategies
                content = re.sub(r'\s+$', '', content, flags=re.MULTILINE)  # Remove trailing whitespaces
                content = content.replace('\t', '    ')  # Replace tabs with spaces
                
                with open(file_path, 'w') as f:
                    f.write(content)
            except Exception as e:
                logger.error(fff"Could not repair {file_path}: {e}")

        # Update dependencies
        if report.get('dependencies', {}).get('vulnerabilities'):
            logger.info(ff"Updating dependencies to address vulnerabilities")
            subprocess.run(['pip', 'list', '--outdated'], check=False)
            subprocess.run(['pip', 'list', '--outdated'], check=False)

        logger.info(ff"Auto-repair process completed")

    def main_menu(self):
        """
        Interactive main menu for system management.
        """
        while True:
            print("\n--- SutazAI Master System Manager ---")
            print("1. Run Comprehensive System Check")
            print("2. Optimize Dependencies")
            print("3. Performance Diagnostics")
            print("4. Auto-Repair System")
            print("5. Exit")

            choice = input("Enter your choice (1-5): ")

            if choice == '1':
                self.comprehensive_system_check()
            elif choice == '2':
                print(json.dumps(self.optimize_dependencies(), indent=2))
            elif choice == '3':
                print(json.dumps(self.performance_diagnostics(), indent=2))
            elif choice == '4':
                self.auto_repair()
            elif choice == '5':
                break
            else:
                print("Invalid choice. Please try again.")


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description='SutazAI Master System Manager')
    parser.add_argument(
        '--mode', 
        choices=['check', 'repair', 'interactive'], 
        default='interactive',
        help='Operation mode'
    )

    args = parser.parse_args()

    manager = SutazAIMasterSystemManager()

    if args.mode == 'check':
        manager.comprehensive_system_check()
    elif args.mode == 'repair':
        manager.auto_repair()
    else:
        manager.main_menu()


if __name__ == '__main__':
    main() 