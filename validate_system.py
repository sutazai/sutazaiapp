#!/usr/bin/env python3
"""
SutazAI System Validation Script
Validates the complete system installation and functionality
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemValidator:
    """Validates the complete SutazAI system"""
    
    def __init__(self):
        self.base_dir = Path("/opt/sutazaiapp")
        self.results = []
        self.start_time = time.time()
        
    def log_result(self, test_name: str, status: str, message: str = "", details: Any = None):
        """Log a test result"""
        result = {
            "test": test_name,
            "status": status,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        
        status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        logger.info(f"{status_symbol} {test_name}: {message}")
    
    def check_file_structure(self):
        """Check if all required files exist"""
        logger.info("Checking file structure...")
        
        required_files = [
            "main_agi.py",
            "core/agi_system.py",
            "core/security.py",
            "core/exceptions.py",
            "api/agi_api.py",
            "models/local_model_manager.py",
            "monitoring/observability.py",
            "deployment/docker_deployment.py",
            "deployment/kubernetes_deployment.py",
            "setup_enterprise.py",
            "README_ENTERPRISE.md",
            "ENTERPRISE_DEPLOYMENT_STATUS.md"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.base_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.log_result("File Structure", "FAIL", f"Missing files: {missing_files}")
            return False
        else:
            self.log_result("File Structure", "PASS", f"All {len(required_files)} required files present")
            return True
    
    def check_imports(self):
        """Check if all modules can be imported"""
        logger.info("Checking module imports...")
        
        modules_to_test = [
            ("core.agi_system", "IntegratedAGISystem"),
            ("core.security", "SecurityManager"),
            ("core.exceptions", "SutazaiException"),
            ("api.agi_api_simple", "AGIAPISystem"),
            ("models.local_model_manager_simple", "LocalModelManager"),
            ("monitoring.observability_simple", "ObservabilitySystem"),
            ("deployment.docker_deployment_simple", "DockerDeploymentManager"),
            ("deployment.kubernetes_deployment_simple", "KubernetesDeploymentManager")
        ]
        
        failed_imports = []
        
        for module_name, class_name in modules_to_test:
            try:
                # Add base directory to Python path
                sys.path.insert(0, str(self.base_dir))
                
                # Try to import the module
                module = importlib.import_module(module_name)
                
                # Try to get the class
                if hasattr(module, class_name):
                    self.log_result(f"Import {module_name}", "PASS", f"Successfully imported {class_name}")
                else:
                    self.log_result(f"Import {module_name}", "FAIL", f"Class {class_name} not found")
                    failed_imports.append(f"{module_name}.{class_name}")
                    
            except Exception as e:
                self.log_result(f"Import {module_name}", "FAIL", f"Import failed: {str(e)}")
                failed_imports.append(f"{module_name}: {str(e)}")
        
        if failed_imports:
            return False
        else:
            return True
    
    def check_configuration(self):
        """Check system configuration"""
        logger.info("Checking configuration...")
        
        config_files = [
            "config/settings.json",
            "data/model_registry.json",
            "data/neural_network.json"
        ]
        
        config_status = True
        
        for config_file in config_files:
            config_path = self.base_dir / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    self.log_result(f"Config {config_file}", "PASS", f"Valid JSON with {len(config_data)} keys")
                except json.JSONDecodeError as e:
                    self.log_result(f"Config {config_file}", "FAIL", f"Invalid JSON: {str(e)}")
                    config_status = False
            else:
                self.log_result(f"Config {config_file}", "WARN", "File not found (may be created on first run)")
        
        return config_status
    
    def check_directories(self):
        """Check required directories exist"""
        logger.info("Checking directories...")
        
        required_dirs = [
            "core", "api", "models", "monitoring", "deployment",
            "nln", "tests", "data", "logs", "config"
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = self.base_dir / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            self.log_result("Directories", "FAIL", f"Missing directories: {missing_dirs}")
            return False
        else:
            self.log_result("Directories", "PASS", f"All {len(required_dirs)} required directories present")
            return True
    
    def check_python_version(self):
        """Check Python version compatibility"""
        logger.info("Checking Python version...")
        
        version_info = sys.version_info
        python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
        
        if version_info.major == 3 and version_info.minor >= 8:
            self.log_result("Python Version", "PASS", f"Python {python_version} is compatible")
            return True
        else:
            self.log_result("Python Version", "FAIL", f"Python {python_version} is not supported (requires 3.8+)")
            return False
    
    def check_system_resources(self):
        """Check system resources"""
        logger.info("Checking system resources...")
        
        try:
            # Check available disk space
            disk_usage = os.statvfs(str(self.base_dir))
            free_space_gb = (disk_usage.f_bavail * disk_usage.f_frsize) / (1024**3)
            
            if free_space_gb > 1.0:
                self.log_result("Disk Space", "PASS", f"{free_space_gb:.2f}GB available")
            else:
                self.log_result("Disk Space", "WARN", f"Low disk space: {free_space_gb:.2f}GB")
            
            # Check if we can write to directories
            test_file = self.base_dir / "test_write.tmp"
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                test_file.unlink()
                self.log_result("Write Permissions", "PASS", "Can write to installation directory")
            except Exception as e:
                self.log_result("Write Permissions", "FAIL", f"Cannot write to directory: {str(e)}")
                return False
            
            return True
            
        except Exception as e:
            self.log_result("System Resources", "FAIL", f"Resource check failed: {str(e)}")
            return False
    
    def check_core_functionality(self):
        """Test core system functionality"""
        logger.info("Checking core functionality...")
        
        try:
            # Add to Python path
            sys.path.insert(0, str(self.base_dir))
            
            # Test AGI system basic functionality
            try:
                from core.agi_system import create_agi_task, TaskPriority
                
                # Create a test task
                task = create_agi_task(
                    name="validation_test",
                    priority=TaskPriority.HIGH,
                    data={"test": "validation"}
                )
                
                if task and task.name == "validation_test":
                    self.log_result("AGI Task Creation", "PASS", "Can create AGI tasks")
                else:
                    self.log_result("AGI Task Creation", "FAIL", "Task creation failed")
                    
            except Exception as e:
                self.log_result("AGI Task Creation", "FAIL", f"Error: {str(e)}")
            
            # Test security manager
            try:
                from core.security import SecurityManager
                
                security_manager = SecurityManager()
                test_input = {"test": "data"}
                
                # Test input validation
                is_valid = security_manager.validate_input(test_input)
                if isinstance(is_valid, bool):
                    self.log_result("Security Validation", "PASS", "Security manager functional")
                else:
                    self.log_result("Security Validation", "FAIL", "Security validation failed")
                    
            except Exception as e:
                self.log_result("Security Validation", "FAIL", f"Error: {str(e)}")
            
            # Test exception handling
            try:
                from core.exceptions import SutazaiException
                
                # Test exception creation
                test_exception = SutazaiException("Test exception")
                if isinstance(test_exception, Exception):
                    self.log_result("Exception Handling", "PASS", "Exception system functional")
                else:
                    self.log_result("Exception Handling", "FAIL", "Exception creation failed")
                    
            except Exception as e:
                self.log_result("Exception Handling", "FAIL", f"Error: {str(e)}")
            
            return True
            
        except Exception as e:
            self.log_result("Core Functionality", "FAIL", f"Core test failed: {str(e)}")
            return False
    
    def generate_report(self):
        """Generate validation report"""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r["status"] == "PASS"])
        failed_tests = len([r for r in self.results if r["status"] == "FAIL"])
        warning_tests = len([r for r in self.results if r["status"] == "WARN"])
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        total_time = time.time() - self.start_time
        
        report = {
            "validation_report": {
                "timestamp": datetime.now().isoformat(),
                "total_time": total_time,
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "failed": failed_tests,
                    "warnings": warning_tests,
                    "success_rate": success_rate,
                    "status": "PASS" if success_rate >= 80 and failed_tests == 0 else "FAIL"
                },
                "detailed_results": self.results
            }
        }
        
        # Save report
        report_file = self.base_dir / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def run_validation(self):
        """Run complete system validation"""
        logger.info("=== Starting SutazAI System Validation ===")
        
        # Run all validation checks
        self.check_python_version()
        self.check_directories()
        self.check_file_structure()
        self.check_system_resources()
        self.check_configuration()
        self.check_imports()
        self.check_core_functionality()
        
        # Generate and display report
        report = self.generate_report()
        
        logger.info("=== Validation Complete ===")
        
        summary = report["validation_report"]["summary"]
        print(f"\n{'='*60}")
        print("SUTAZAI SYSTEM VALIDATION REPORT")
        print(f"{'='*60}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Warnings: {summary['warnings']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Overall Status: {summary['status']}")
        print(f"Validation Time: {report['validation_report']['total_time']:.2f}s")
        print(f"Report saved to: validation_report.json")
        
        if summary['status'] == 'PASS':
            print(f"\n✅ SYSTEM VALIDATION PASSED")
            print("The SutazAI system is ready for deployment!")
        else:
            print(f"\n❌ SYSTEM VALIDATION FAILED")
            print("Please review the failed tests and fix issues before deployment.")
        
        return summary['status'] == 'PASS'

def main():
    """Main validation entry point"""
    validator = SystemValidator()
    success = validator.run_validation()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())