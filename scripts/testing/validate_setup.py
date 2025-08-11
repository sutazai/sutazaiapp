#!/usr/bin/env python3
"""
Validate Integration Test Setup

Purpose: Verify all test components are properly configured
Usage: python validate_setup.py
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import os
import sys
import subprocess
import requests
from pathlib import Path
import importlib.util

def check_component(name, check_func):
    """Check a component and report status"""
    try:
        result = check_func()
        if result:
            print(f"✓ {name}: OK")
            return True
        else:
            print(f"✗ {name}: FAILED")
            return False
    except Exception as e:
        print(f"✗ {name}: ERROR - {str(e)}")
        return False

def check_python_version():
    """Check Python version is 3.6+"""
    return sys.version_info >= (3, 6)

def check_required_modules():
    """Check all required Python modules"""
    required = ['requests', 'psutil', 'docker', 'schedule']
    missing = []
    
    for module in required:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    
    if missing:
        print(f"  Missing modules: {', '.join(missing)}")
        print("  Install with: pip3 install " + ' '.join(missing))
        return False
    return True

def check_agent_running():
    """Check if hardware optimizer agent is running"""
    try:
        response = requests.get("http://localhost:8116/health", timeout=2)
        return response.status_code == 200
    except (AssertionError, Exception) as e:
        logger.warning(f"Exception caught, returning: {e}")
        return False

def check_test_files():
    """Check all test files exist"""
    test_dir = Path(__file__).parent
    required_files = [
        "integration_test_suite.py",
        "continuous_validator.py",
        "run_tests.sh",
        "TEST_GUIDE.md",
        "hardware-optimizer-tests.service"
    ]
    
    missing = []
    for file in required_files:
        if not (test_dir / file).exists():
            missing.append(file)
    
    if missing:
        print(f"  Missing files: {', '.join(missing)}")
        return False
    return True

def check_dashboard_file():
    """Check dashboard HTML exists"""
    dashboard = Path(__file__).parent.parent / "test_dashboard.html"
    return dashboard.exists()

def check_permissions():
    """Check file permissions"""
    script = Path(__file__).parent / "run_tests.sh"
    if script.exists():
        return os.access(str(script), os.X_OK)
    return False

def check_docker():
    """Check Docker availability"""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except (AssertionError, Exception) as e:
        logger.error(f"Unexpected exception: {e}", exc_info=True)
        print("  Docker not installed or not in PATH")
        return False

def check_disk_space():
    """Check available disk space"""
    import shutil
    stat = shutil.disk_usage("/tmp")
    free_gb = stat.free / (1024**3)
    if free_gb < 1:
        print(f"  Low disk space: {free_gb:.1f} GB free")
        return False
    return True

def check_port_availability():
    """Check if dashboard port is available"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('', 8117))
        sock.close()
        return True
    except (AssertionError, Exception) as e:
        logger.error(f"Unexpected exception: {e}", exc_info=True)
        print("  Port 8117 is in use")
        return False

def main():
    """Run all validation checks"""
    print("Hardware Optimizer Test Setup Validation")
    print("=" * 40)
    print()
    
    checks = [
        ("Python Version (3.6+)", check_python_version),
        ("Required Python Modules", check_required_modules),
        ("Test Files Present", check_test_files),
        ("Dashboard File", check_dashboard_file),
        ("Script Permissions", check_permissions),
        ("Docker Available", check_docker),
        ("Disk Space (>1GB)", check_disk_space),
        ("Dashboard Port 8117", check_port_availability),
        ("Hardware Optimizer Agent", check_agent_running),
    ]
    
    passed = 0
    failed = 0
    
    for name, check_func in checks:
        if check_component(name, check_func):
            passed += 1
        else:
            failed += 1
    
    print()
    print("=" * 40)
    print(f"Total: {passed + failed} checks")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()
    
    if failed == 0:
        print("✓ All checks passed! Ready to run tests.")
        print()
        print("Quick start:")
        print("  ./run_tests.sh")
        return 0
    else:
        print("✗ Some checks failed. Please fix issues before running tests.")
        
        if not check_agent_running():
            print()
            print("To start the agent:")
            print("  cd /opt/sutazaiapp/agents/hardware-resource-optimizer")
            print("  python3 app.py")
        
        return 1

if __name__ == "__main__":
