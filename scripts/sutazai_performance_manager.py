#!/usr/bin/env python3
"""
SutazAI Performance and Dependency Management System

Provides comprehensive system performance monitoring, 
dependency management, and optimization capabilities.
"""

import os
import sys
import subprocess
import logging
import platform
import psutil
import json
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    filename='/opt/sutazai/logs/performance_management.log'
)
logger = logging.getLogger(__name__)

class SutazAIPerformanceManager:
    def __init__(self, project_root: str = '/opt/sutazai_project/SutazAI'):
        """
        Initialize performance manager with project root.
        
        Args:
            project_root (str): Root directory of the SutazAI project
        """
        self.project_root = project_root
        self.log_dir = '/opt/sutazai/logs/performance_reports'
        os.makedirs(self.log_dir, exist_ok=True)

    def run_system_diagnostics(self) -> Dict[str, Any]:
        """
        Perform comprehensive system diagnostics.
        
        Returns:
            Dict containing system diagnostic information
        """
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'os': platform.platform(),
                'python_version': platform.python_version(),
                'architecture': platform.machine()
            },
            'cpu': {
                'physical_cores': psutil.cpu_count(logical=False),
                'total_cores': psutil.cpu_count(logical=True),
                'current_frequency': psutil.cpu_freq().current,
                'usage_percent': psutil.cpu_percent(interval=1)
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'used_percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'free': psutil.disk_usage('/').free,
                'used_percent': psutil.disk_usage('/').percent
            },
            'network': self._get_network_info()
        }

        # Save diagnostics to file
        report_path = os.path.join(
            self.log_dir, 
            f'system_diagnostics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(report_path, 'w') as f:
            json.dump(diagnostics, f, indent=4)

        logger.info(f"System diagnostics saved to {report_path}")
        return diagnostics

    def _get_network_info(self) -> Dict[str, Any]:
        """
        Retrieve network interface information.
        
        Returns:
            Dict containing network interface details
        """
        network_info = {}
        try:
            for interface, addresses in psutil.net_if_addrs().items():
                network_info[interface] = {
                    'ip_addresses': [
                        addr.address for addr in addresses 
                        if addr.family == 2  # IPv4
                    ]
                }
        except Exception as e:
            logger.error(f"Error retrieving network information: {e}")
        return network_info

    def optimize_python_environment(self) -> bool:
        """
        Optimize Python environment for performance.
        
        Returns:
            bool: True if optimization successful, False otherwise
        """
        try:
            # Upgrade pip and setuptools
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                '--upgrade', '--break-system-packages', 'pip', 'setuptools', 'wheel'
            ], check=True)

            # Install performance packages
            performance_packages = [
                'cython', 'numpy', 'numba', 
                'psutil', 'py-spy', 'memory_profiler',
                'pylint', 'black', 'isort', 'flake8', 'mypy'
            ]
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                '--upgrade', '--break-system-packages'
            ] + performance_packages, check=True)

            # Compile Python bytecode with graceful error handling
            try:
                subprocess.run([
                    sys.executable, '-m', 'compileall', 
                    self.project_root
                ], check=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Compileall failed: {e}. Proceeding without compiled bytecode.")

            logger.info("Python environment optimization completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Python environment optimization failed: {e}")
            return False

    def manage_dependencies(self) -> bool:
        """
        Manage project dependencies.
        
        Returns:
            bool: True if dependency management successful, False otherwise
        """
        try:
            # Update requirements
            requirements_path = os.path.join(self.project_root, 'requirements.txt')
            if (not os.path.exists(requirements_path)) or (os.path.getsize(requirements_path) == 0):
                unified_path = os.path.join(self.project_root, 'requirements_unified.txt')
                if os.path.exists(unified_path) and os.path.getsize(unified_path) > 0:
                    logger.info("requirements.txt is empty, installing dependencies from requirements_unified.txt")
                    subprocess.run([
                        sys.executable, '-m', 'pip', 'install', '--break-system-packages', '-r', unified_path
                    ], check=True)
                else:
                    logger.info("No valid requirements file found, skipping dependency installation.")
            else:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '--break-system-packages', '-r', requirements_path
                ], check=True)

            # Check for outdated packages
            outdated = subprocess.run([
                sys.executable, '-m', 'pip', 'list', '--outdated'
            ], capture_output=True, text=True)
            
            if outdated.stdout:
                logger.warning(f"Outdated packages found:\n{outdated.stdout}")

            logger.info("Dependency management completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Dependency management failed: {e}")
            return False

    def run_comprehensive_optimization(self):
        """
        Run a comprehensive system optimization process.
        """
        logger.info("Starting comprehensive system optimization")
        
        # Run diagnostics
        diagnostics = self.run_system_diagnostics()
        
        # Optimize Python environment
        python_optimization_result = self.optimize_python_environment()
        
        # Manage dependencies
        dependency_management_result = self.manage_dependencies()
        
        # Log overall results
        optimization_report = {
            'timestamp': datetime.now().isoformat(),
            'system_diagnostics': diagnostics,
            'python_optimization': python_optimization_result,
            'dependency_management': dependency_management_result
        }
        
        report_path = os.path.join(
            self.log_dir, 
            f'optimization_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(report_path, 'w') as f:
            json.dump(optimization_report, f, indent=4)
        
        logger.info(f"Comprehensive optimization report saved to {report_path}")

def main():
    """
    Main execution function for performance management.
    """
    try:
        performance_manager = SutazAIPerformanceManager()
        performance_manager.run_comprehensive_optimization()
    except Exception as e:
        logger.error(f"Performance management failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 