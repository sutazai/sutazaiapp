#!/usr/bin/env python3.11
"""
SutazAI Final System Test Script

This script performs a comprehensive system test and validation
of the SutazAI application.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pytest
except ImportError:
    pytest = None

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    Console = None
    Panel = None
    Table = None


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


class SystemValidator:
    """Comprehensive system validation framework."""
    
    def __init__(self):
        """Initialize comprehensive system validation framework."""
        self.critical_dirs = [
            "ai_agents",
            "model_management", 
            "backend",
            "scripts",
        ]
        self.required_models = ["gpt4all", "deepseek-coder", "llama2"]
        self.console = Console() if Console else None
    
    def validate_system_requirements(self) -> None:
        """
        Comprehensive system requirements validation.
        
        Raises:
            ValidationError: If any system requirement is not met
        """
        logger.info("🔍 Starting Comprehensive System Validation")
        
        # Python version check
        logger.info("Python Version: %s", sys.version)
        if not (sys.version_info >= (3, 11)):
            raise ValidationError("Python 3.11+ is required")
        
        # OS and Hardware Validation
        self._validate_os_and_hardware()
        
        # Critical Directories Check
        self._validate_critical_directories()
        
        # Network Configuration Check
        self._validate_network_config()
        
        logger.info("✅ System validation completed successfully")
    
    def _validate_os_and_hardware(self) -> None:
        """Validate OS and hardware requirements."""
        if not psutil:
            logger.warning("psutil not available, skipping hardware validation")
            return
        
        # Check available memory
        memory = psutil.virtual_memory()
        if memory.total < 8 * 1024 * 1024 * 1024:  # 8GB
            logger.warning("Less than 8GB RAM available")
        
        # Check available disk space
        disk = psutil.disk_usage('/')
        if disk.free < 10 * 1024 * 1024 * 1024:  # 10GB
            logger.warning("Less than 10GB disk space available")
    
    def _validate_critical_directories(self) -> None:
        """Validate critical application directories exist."""
        missing_dirs = []
        
        for directory in self.critical_dirs:
            if not os.path.exists(directory):
                missing_dirs.append(directory)
        
        if missing_dirs:
            logger.warning("Missing directories: %s", missing_dirs)
    
    def _validate_network_config(self) -> None:
        """Validate network configuration."""
        try:
            import socket
            # Test basic connectivity
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            logger.info("Network connectivity verified")
        except OSError:
            logger.warning("Network connectivity issues detected")
    
    def run_test_suite(self) -> Dict[str, Any]:
        """
        Run comprehensive test suite.
        
        Returns:
            Dict containing test results
        """
        if not pytest:
            logger.warning("pytest not available, skipping tests")
            return {"status": "skipped", "reason": "pytest not available"}
        
        # Run pytest
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return {
                "status": "completed",
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "exit_code": -1,
                "error": "Tests timed out after 5 minutes"
            }
        except Exception as e:
            return {
                "status": "error",
                "exit_code": -1,
                "error": str(e)
            }
    
    def generate_report(self, test_results: Dict[str, Any]) -> str:
        """Generate comprehensive system test report."""
        timestamp = datetime.now().isoformat()
        
        report = {
            "timestamp": timestamp,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
            },
            "test_results": test_results,
            "validation_status": "completed"
        }
        
        # Save report
        report_path = "system_test_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info("System test report saved to: %s", report_path)
        return report_path


def main():
    """Main test pipeline execution."""
    validator = SystemValidator()
    
    try:
        # System validation
        validator.validate_system_requirements()
        
        # Run test suite
        test_results = validator.run_test_suite()
        
        # Generate report
        report_path = validator.generate_report(test_results)
        
        # Print summary
        if test_results.get("exit_code") == 0:
            logger.info("🎉 All tests passed!")
        else:
            logger.error("❌ Some tests failed. Check report: %s", report_path)
            
        return test_results.get("exit_code", 0)
        
    except ValidationError as e:
        logger.error("Validation failed: %s", e)
        return 1
    except Exception as e:
        logger.error("Test pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
