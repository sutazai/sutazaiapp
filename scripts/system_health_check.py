#!/usr/bin/env python3
"""
SutazAI System Health Check and Diagnostic Tool

Provides comprehensive system health assessment, 
dependency verification, and performance analysis.
"""

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    filename='/opt/sutazai/logs/system_health.log'
)
logger = logging.getLogger(__name__)

class SystemHealthChecker:
    def __init__(self, project_root: str = '/opt/sutazai_project/SutazAI'):
        """
        Initialize system health checker.
        
        Args:
            project_root (str): Root directory of the project
        """
        self.project_root = project_root
        self.health_report_dir = '/opt/sutazai/logs/health_reports'
        os.makedirs(self.health_report_dir, exist_ok=True)

    def check_system_dependencies(self) -> Dict[str, Any]:
        """
        Check critical system dependencies.
        
        Returns:
            Dict of dependency check results
        """
        dependencies = {
            'python_version': self._check_python_version(),
            'pip_version': self._check_pip_version(),
            'venv_status': self._check_virtual_environment(),
            'required_packages': self._check_required_packages()
        }
        return dependencies

    def _check_python_version(self) -> Dict[str, Any]:
        """
        Check Python version compatibility.
        
        Returns:
            Dict with Python version details
        """
        try:
            version_output = subprocess.check_output(
                ['python3', '--version'], 
                universal_newlines=True
            ).strip()
            
            version_parts = version_output.split()[1].split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            return {
                'version': version_output,
                'is_compatible': (major == 3 and minor >= 8),
                'status': 'OK' if (major == 3 and minor >= 8) else 'WARNING'
            }
        except Exception as e:
            logger.error(f"Python version check failed: {e}")
            return {'error': str(e), 'status': 'ERROR'}

    def _check_pip_version(self) -> Dict[str, Any]:
        """
        Check pip version and installation.
        
        Returns:
            Dict with pip version details
        """
        try:
            version_output = subprocess.check_output(
                ['python3', '-m', 'pip', '--version'], 
                universal_newlines=True
            ).strip()
            
            return {
                'version': version_output,
                'status': 'OK'
            }
        except Exception as e:
            logger.error(f"Pip version check failed: {e}")
            return {'error': str(e), 'status': 'ERROR'}

    def _check_virtual_environment(self) -> Dict[str, Any]:
        """
        Check virtual environment status.
        
        Returns:
            Dict with virtual environment details
        """
        venv_path = os.path.join(self.project_root, 'venv')
        
        if not os.path.exists(venv_path):
            return {
                'exists': False,
                'status': 'WARNING',
                'message': 'Virtual environment not found'
            }
        
        try:
            # Check if venv is properly configured
            activate_path = os.path.join(venv_path, 'bin', 'activate')
            python_path = os.path.join(venv_path, 'bin', 'python')
            
            return {
                'exists': True,
                'activate_script_exists': os.path.exists(activate_path),
                'python_executable_exists': os.path.exists(python_path),
                'status': 'OK'
            }
        except Exception as e:
            logger.error(f"Virtual environment check failed: {e}")
            return {'error': str(e), 'status': 'ERROR'}

    def _check_required_packages(self) -> Dict[str, Any]:
        """
        Check required package installations.
        
        Returns:
            Dict with package installation status
        """
        try:
            # Read requirements from requirements.txt
            requirements_path = os.path.join(self.project_root, 'requirements.txt')
            
            with open(requirements_path, 'r') as f:
                required_packages = [
                    line.split('==')[0].strip() 
                    for line in f 
                    if line.strip() and not line.startswith('#')
                ]
            
            # Check package installations
            installed_packages = subprocess.check_output(
                ['python3', '-m', 'pip', 'list'], 
                universal_newlines=True
            ).splitlines()[2:]  # Skip header lines
            
            installed_package_names = [
                line.split()[0] for line in installed_packages
            ]
            
            missing_packages = [
                pkg for pkg in required_packages 
                if not any(pkg.lower() in installed.lower() for installed in installed_package_names)
            ]
            
            return {
                'total_required': len(required_packages),
                'installed': len(required_packages) - len(missing_packages),
                'missing_packages': missing_packages,
                'status': 'OK' if not missing_packages else 'WARNING'
            }
        except Exception as e:
            logger.error(f"Package check failed: {e}")
            return {'error': str(e), 'status': 'ERROR'}

    def generate_health_report(self) -> str:
        """
        Generate comprehensive system health report.
        
        Returns:
            Path to generated health report
        """
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'os': platform.platform(),
                'python_version': platform.python_version()
            },
            'dependencies': self.check_system_dependencies()
        }
        
        # Generate report filename
        report_filename = f'health_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_path = os.path.join(self.health_report_dir, report_filename)
        
        # Write report
        with open(report_path, 'w') as f:
            json.dump(health_report, f, indent=4)
        
        logger.info(f"Health report generated: {report_path}")
        return report_path

def main():
    """
    Main execution function for system health check.
    """
    try:
        health_checker = SystemHealthChecker()
        report_path = health_checker.generate_health_report()
        print(f"Health report generated: {report_path}")
        
        # Check for any critical issues
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        # Example of checking for critical issues
        if any(dep.get('status') == 'ERROR' for dep in report['dependencies'].values()):
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 