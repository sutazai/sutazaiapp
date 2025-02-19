#!/usr/bin/env python3
"""
SutazAI Autonomous System Initialization Script

Comprehensive initialization framework providing:
- Automatic system configuration
- Dependency validation
- Environment setup
- Security hardening
- Performance optimization
"""

import os
import sys
import subprocess
import logging
from typing import Dict, Any, List
import json
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Internal system imports
from config.config_manager import ConfigurationManager
from scripts.dependency_manager import DependencyManager
from security.security_manager import SecurityManager
from core_system.system_optimizer import SystemOptimizer
from core_system.monitoring.advanced_logger import AdvancedLogger

class SystemInitializer:
    """
    Comprehensive autonomous system initialization framework
    """
    
    def __init__(
        self, 
        base_dir: str = '/opt/sutazai_project/SutazAI',
        config_env: str = 'development'
    ):
        """
        Initialize system initialization framework
        
        Args:
            base_dir (str): Base directory of the project
            config_env (str): Configuration environment
        """
        self.base_dir = base_dir
        self.config_env = config_env
        
        # System components
        self.config_manager = ConfigurationManager(environment=config_env)
        self.dependency_manager = DependencyManager()
        self.security_manager = SecurityManager()
        self.system_optimizer = SystemOptimizer()
        self.logger = AdvancedLogger()
        
        # Initialization configuration
        self.initialization_steps = [
            self._validate_environment,
            self._check_dependencies,
            self._configure_security,
            self._optimize_system,
            self._generate_initialization_report
        ]
    
    def _validate_environment(self) -> Dict[str, Any]:
        """
        Validate system environment and prerequisites
        
        Returns:
            Environment validation results
        """
        validation_results = {
            'python_version': sys.version_info,
            'os_details': os.uname(),
            'required_directories': {}
        }
        
        # Check required directories
        required_dirs = [
            'ai_agents', 'backend', 'core_system', 'logs', 
            'security', 'config', 'scripts'
        ]
        
        for directory in required_dirs:
            full_path = os.path.join(self.base_dir, directory)
            validation_results['required_directories'][directory] = {
                'exists': os.path.exists(full_path),
                'is_dir': os.path.isdir(full_path) if os.path.exists(full_path) else False
            }
        
        # Validate Python version
        if sys.version_info < (3, 10):
            validation_results['python_version_warning'] = "Python 3.10+ recommended"
        
        return validation_results
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """
        Check and validate project dependencies
        
        Returns:
            Dependency validation results
        """
        return self.dependency_manager.comprehensive_dependency_analysis()
    
    def _configure_security(self) -> Dict[str, Any]:
        """
        Configure and harden system security
        
        Returns:
            Security configuration results
        """
        return self.security_manager.comprehensive_security_scan()
    
    def _optimize_system(self) -> Dict[str, Any]:
        """
        Perform system-wide performance optimization
        
        Returns:
            System optimization results
        """
        return self.system_optimizer.generate_comprehensive_report()
    
    def _generate_initialization_report(
        self, 
        validation_results: Dict[str, Any], 
        dependency_results: Dict[str, Any],
        security_results: Dict[str, Any],
        optimization_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive initialization report
        
        Args:
            validation_results (Dict): Environment validation results
            dependency_results (Dict): Dependency analysis results
            security_results (Dict): Security configuration results
            optimization_results (Dict): System optimization results
        
        Returns:
            Comprehensive initialization report
        """
        initialization_report = {
            'timestamp': datetime.now().isoformat(),
            'environment_validation': validation_results,
            'dependency_analysis': dependency_results,
            'security_configuration': security_results,
            'system_optimization': optimization_results
        }
        
        # Persist report
        report_path = os.path.join(
            self.base_dir, 
            f'logs/system_initialization_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(report_path, 'w') as f:
            json.dump(initialization_report, f, indent=2)
        
        return initialization_report
    
    def initialize_system(self) -> Dict[str, Any]:
        """
        Execute comprehensive system initialization
        
        Returns:
            Comprehensive initialization report
        """
        try:
            # Execute initialization steps
            validation_results = self._validate_environment()
            dependency_results = self._check_dependencies()
            security_results = self._configure_security()
            optimization_results = self._optimize_system()
            
            # Generate final initialization report
            initialization_report = self._generate_initialization_report(
                validation_results,
                dependency_results,
                security_results,
                optimization_results
            )
            
            # Log initialization
            self.logger.log(
                "System initialization completed successfully", 
                level='info', 
                context=initialization_report
            )
            
            return initialization_report
        
        except Exception as e:
            self.logger.log(
                f"System initialization failed: {e}", 
                level='error'
            )
            raise

def main():
    """
    Main execution for system initialization
    """
    try:
        initializer = SystemInitializer()
        initialization_report = initializer.initialize_system()
        
        print("System Initialization Report:")
        print(json.dumps(initialization_report, indent=2))
    
    except Exception as e:
        print(f"System initialization failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 