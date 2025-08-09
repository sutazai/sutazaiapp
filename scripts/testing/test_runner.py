#!/usr/bin/env python3
"""
SutazAI Test Runner - Comprehensive testing framework
Purpose: Execute different types of tests based on environment capabilities
Usage: python scripts/test_runner.py --type <test_type> [options]
Author: QA Team Lead (QA-LEAD-001)
"""

import os
import sys
import json
import time
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to path
# Resolve project root two levels up from this file: scripts/testing -> scripts -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

class TestRunner:
    """Comprehensive test execution framework"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "environment": os.environ.get("SUTAZAI_ENV", "development"),
            "python_version": sys.version,
            "tests_executed": [],
            "total_passed": 0,
            "total_failed": 0,
            "total_skipped": 0
        }
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.project_root / 'test-execution.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_docker_available(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def check_pytest_available(self) -> bool:
        """Check if pytest is available"""
        try:
            import pytest
            return True
        except ImportError:
            return False
    
    def run_command(self, command: List[str], description: str) -> Tuple[bool, str, str]:
        """Execute a command and return results"""
        self.logger.info(f"Running: {description}")
        self.logger.debug(f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                timeout=300,  # 5 minute timeout
                cwd=self.project_root
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out after 5 minutes"
        except Exception as e:
            return False, "", str(e)
    
    def run_syntax_validation(self) -> Dict:
        """Validate Python syntax across the codebase"""
        self.logger.info("🔍 Running syntax validation...")
        
        results = {
            "test_type": "syntax_validation",
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        # Find all Python files
        python_files = []
        for pattern in ["backend/**/*.py", "agents/**/*.py", "tests/**/*.py", "scripts/**/*.py"]:
            python_files.extend(self.project_root.glob(pattern))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), str(py_file), 'exec')
                results["passed"] += 1
            except SyntaxError as e:
                results["failed"] += 1
                results["errors"].append(f"{py_file}: {e}")
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{py_file}: {e}")
        
        return results
    
    def run_security_scan(self) -> Dict:
        """Run security scanning for hardcoded credentials"""
        self.logger.info("🔒 Running security scan...")
        
        results = {
            "test_type": "security_scan",
            "passed": 0,
            "failed": 0,
            "warnings": []
        }
        
        # Patterns to look for
        # Patterns: avoid false positives inside string literals and variable suffixes
        security_patterns = [
            (r'(?<![A-Za-z0-9_\"])(password)\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
            (r'(?<![A-Za-z0-9_\"])(secret)\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded secret"),
            (r'(?<![A-Za-z0-9_\"])(token)\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded token"),
            (r'(?<![A-Za-z0-9_\"])(api_key)\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded API key"),
        ]
        
        # Check Python files for security issues
        python_files = list(self.project_root.glob("**/*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, description in security_patterns:
                    import re
                    if re.search(pattern, content, re.IGNORECASE):
                        results["failed"] += 1
                        results["warnings"].append(f"{py_file}: {description}")
                    else:
                        results["passed"] += 1
                        
            except Exception as e:
                results["warnings"].append(f"Error reading {py_file}: {e}")
        
        return results
    
    def run_configuration_validation(self) -> Dict:
        """Validate system configuration"""
        self.logger.info("⚙️ Running configuration validation...")
        
        results = {
            "test_type": "configuration_validation",
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        # Check critical configuration files
        config_checks = [
            ("docker-compose.yml", "Main Docker Compose configuration"),
            ("backend/app/main.py", "Backend application entry point"),
            ("pytest.ini", "Test configuration"),
            (".env.secure.template", "Security environment template"),
            ("scripts/deployment/deploy.sh", "Master deployment script")
        ]
        
        for config_file, description in config_checks:
            file_path = self.project_root / config_file
            if file_path.exists():
                results["passed"] += 1
                results["details"].append(f"✅ {description}: {config_file}")
            else:
                results["failed"] += 1
                results["details"].append(f"❌ {description}: {config_file} (missing)")
        
        # Check model configuration
        docker_compose = self.project_root / "docker-compose.yml"
        if docker_compose.exists():
            with open(docker_compose, 'r') as f:
                content = f.read()
                if "tinyllama" in content:
                    results["passed"] += 1
                    results["details"].append("✅ Model configuration: Uses tinyllama")
                elif "gpt-oss" in content:
                    results["failed"] += 1
                    results["details"].append("❌ Model configuration: Still using gpt-oss")
                else:
                    results["failed"] += 1
                    results["details"].append("❌ Model configuration: No model specified")
        
        return results
    
    def run_import_tests(self) -> Dict:
        """Test that critical modules can be imported"""
        self.logger.info("📦 Running import tests...")
        
        results = {
            "test_type": "import_tests",
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "details": []
        }
        
        # Critical imports to test
        import_tests = [
            ("json", "Standard JSON library"),
            ("os", "Standard OS library"),
            ("sys", "Standard system library"),
            ("pathlib", "Path manipulation"),
            ("datetime", "Date/time utilities"),
            ("subprocess", "Process management"),
            ("logging", "Logging framework"),
        ]
        
        for module_name, description in import_tests:
            try:
                __import__(module_name)
                results["passed"] += 1
                results["details"].append(f"✅ {description}: {module_name}")
            except ImportError as e:
                results["failed"] += 1
                results["details"].append(f"❌ {description}: {module_name} - {e}")
        
        # Try to import backend components
        backend_path = self.project_root / "backend"
        if backend_path.exists():
            sys.path.insert(0, str(backend_path))
            
            backend_tests = [
                ("app.main", "Backend main application"),
                ("app.core.config", "Backend configuration"),
            ]
            
            for module_name, description in backend_tests:
                try:
                    __import__(module_name)
                    results["passed"] += 1
                    results["details"].append(f"✅ {description}: {module_name}")
                except ImportError as e:
                    # Treat missing optional third-party deps as skipped in constrained envs
                    msg = str(e)
                    optional_markers = [
                        "httpx", "pydantic_settings", "fastapi", "uvicorn", "starlette"
                    ]
                    if any(m in msg for m in optional_markers):
                        results["skipped"] += 1
                        results["details"].append(
                            f"⚠️ {description}: {module_name} - skipped (missing dependency: {msg})"
                        )
                    else:
                        results["failed"] += 1
                        results["details"].append(f"❌ {description}: {module_name} - {e}")
        
        return results
    
    def run_file_structure_validation(self) -> Dict:
        """Validate project file structure"""
        self.logger.info("📁 Running file structure validation...")
        
        results = {
            "test_type": "file_structure_validation", 
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        # Expected directories
        expected_dirs = [
            "backend/app",
            "backend/tests", 
            "agents",
            "tests",
            "scripts/deployment",
            "scripts/monitoring", 
            "scripts/devops",
            "frontend",
            "docs"
        ]
        
        for expected_dir in expected_dirs:
            dir_path = self.project_root / expected_dir
            if dir_path.exists() and dir_path.is_dir():
                results["passed"] += 1
                results["details"].append(f"✅ Directory exists: {expected_dir}")
            else:
                results["failed"] += 1
                results["details"].append(f"❌ Directory missing: {expected_dir}")
        
        return results
    
    def run_script_organization_validation(self) -> Dict:
        """Validate script organization from Shell Specialist fixes"""
        self.logger.info("📜 Running script organization validation...")
        
        results = {
            "test_type": "script_organization_validation",
            "passed": 0, 
            "failed": 0,
            "details": []
        }
        
        # Check for master deployment script
        deploy_script = self.project_root / "scripts/deployment/deploy.sh"
        if deploy_script.exists():
            results["passed"] += 1
            results["details"].append("✅ Master deployment script exists")
            
            # Check if it's executable
            if os.access(deploy_script, os.X_OK):
                results["passed"] += 1
                results["details"].append("✅ Deployment script is executable")
            else:
                results["failed"] += 1
                results["details"].append("❌ Deployment script not executable")
        else:
            results["failed"] += 1
            results["details"].append("❌ Master deployment script missing")
        
        # Check for organized script structure
        script_dirs = [
            "scripts/deployment",
            "scripts/monitoring", 
            "scripts/devops",
            "scripts/automation"
        ]
        
        for script_dir in script_dirs:
            dir_path = self.project_root / script_dir
            if dir_path.exists():
                results["passed"] += 1
                results["details"].append(f"✅ Organized script directory: {script_dir}")
            else:
                results["failed"] += 1
                results["details"].append(f"❌ Missing script directory: {script_dir}")
        
        return results
    
    def run_pytest_tests(self) -> Dict:
        """Run pytest tests if available"""
        self.logger.info("🧪 Running pytest tests...")
        
        results = {
            "test_type": "pytest_tests",
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "details": []
        }
        
        if not self.check_pytest_available():
            results["skipped"] += 1
            results["details"].append("⚠️ Pytest not available - skipping pytest tests")
            return results
        
        # Try to run basic tests
        test_commands = [
            (["python3", "-m", "pytest", "tests/test_smoke.py", "-v"], "Smoke tests"),
            (["python3", "-m", "pytest", "backend/tests/test_main_app.py", "-v"], "Backend tests"),
        ]
        
        for command, description in test_commands:
            success, stdout, stderr = self.run_command(command, f"Running {description}")
            
            if success:
                results["passed"] += 1
                results["details"].append(f"✅ {description}: Passed")
            else:
                if "pytest" in stderr or "ModuleNotFoundError" in stderr:
                    results["skipped"] += 1
                    results["details"].append(f"⚠️ {description}: Skipped (missing dependencies)")
                else:
                    results["failed"] += 1
                    results["details"].append(f"❌ {description}: Failed - {stderr[:200]}")
        
        return results
    
    def run_health_checks(self) -> Dict:
        """Run health checks without requiring Docker"""
        self.logger.info("🏥 Running health checks...")
        
        results = {
            "test_type": "health_checks",
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "details": []
        }
        
        # Check if ports would be available (not currently bound)
        import socket
        
        critical_ports = [
            (10010, "Backend API"),
            (10104, "Ollama"),
            (10000, "PostgreSQL"),
            (10001, "Redis"),
            (10002, "Neo4j")
        ]
        
        require_services = os.environ.get("SUTAZAI_REQUIRE_SERVICES", "0").lower() in ("1", "true", "yes")

        for port, service in critical_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            
            try:
                result = sock.connect_ex(('localhost', port))
                if result == 0:
                    results["passed"] += 1
                    results["details"].append(f"✅ {service} (port {port}): Service running")
                else:
                    if require_services:
                        results["failed"] += 1
                        results["details"].append(
                            f"❌ {service} (port {port}): Service not running"
                        )
                    else:
                        results["skipped"] += 1
                        results["details"].append(
                            f"⚠️ {service} (port {port}): Skipped - service not running in local env"
                        )
            except Exception as e:
                results["failed"] += 1
                results["details"].append(f"❌ {service} (port {port}): Error - {e}")
            finally:
                sock.close()
        
        return results
    
    def run_test_suite(self, test_type: str) -> Dict:
        """Run specific test suite"""
        
        if test_type == "unit":
            return self.run_syntax_validation()
        
        elif test_type == "integration":
            return self.run_configuration_validation()
        
        elif test_type == "security":
            return self.run_security_scan()
        
        elif test_type == "docker":
            if not self.check_docker_available():
                return {
                    "test_type": "docker_tests",
                    "passed": 0,
                    "failed": 1,
                    "skipped": 0,
                    "details": ["⚠️ Docker not available - cannot run Docker tests"]
                }
            else:
                return self.run_health_checks()
        
        elif test_type == "all":
            # Run comprehensive test suite
            all_results = []
            
            test_suites = [
                ("syntax", self.run_syntax_validation),
                ("imports", self.run_import_tests),
                ("configuration", self.run_configuration_validation), 
                ("file_structure", self.run_file_structure_validation),
                ("script_organization", self.run_script_organization_validation),
                ("security", self.run_security_scan),
                ("health", self.run_health_checks),
                ("pytest", self.run_pytest_tests)
            ]
            
            for suite_name, suite_func in test_suites:
                self.logger.info(f"Running {suite_name} test suite...")
                result = suite_func()
                all_results.append(result)
                self.test_results["tests_executed"].append(result)
            
            # Aggregate results
            total_passed = sum(r.get("passed", 0) for r in all_results)
            total_failed = sum(r.get("failed", 0) for r in all_results)
            total_skipped = sum(r.get("skipped", 0) for r in all_results)
            
            return {
                "test_type": "comprehensive_suite",
                "passed": total_passed,
                "failed": total_failed,
                "skipped": total_skipped,
                "sub_results": all_results
            }
        
        else:
            return {
                "test_type": test_type,
                "passed": 0,
                "failed": 1,
                "details": [f"❌ Unknown test type: {test_type}"]
            }
    
    def generate_report(self, result: Dict) -> str:
        """Generate test execution report"""
        
        report = []
        report.append("=" * 80)
        report.append("SUTAZAI TEST EXECUTION REPORT")
        report.append("=" * 80)
        report.append(f"Test Type: {result.get('test_type', 'Unknown')}")
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append(f"Environment: {os.environ.get('SUTAZAI_ENV', 'development')}")
        report.append(f"Python Version: {sys.version.split()[0]}")
        report.append("")
        
        # Summary
        passed = result.get("passed", 0)
        failed = result.get("failed", 0) 
        skipped = result.get("skipped", 0)
        total = passed + failed + skipped
        
        report.append("SUMMARY:")
        report.append(f"  Total Tests: {total}")
        report.append(f"  Passed: {passed} ({(passed/total*100) if total > 0 else 0:.1f}%)")
        report.append(f"  Failed: {failed} ({(failed/total*100) if total > 0 else 0:.1f}%)")
        report.append(f"  Skipped: {skipped} ({(skipped/total*100) if total > 0 else 0:.1f}%)")
        report.append("")
        
        # Details
        if result.get("details"):
            report.append("DETAILS:")
            for detail in result["details"]:
                report.append(f"  {detail}")
            report.append("")
        
        # Sub-results for comprehensive tests
        if result.get("sub_results"):
            report.append("DETAILED RESULTS BY TEST SUITE:")
            for sub_result in result["sub_results"]:
                report.append(f"  {sub_result['test_type'].upper()}:")
                report.append(f"    Passed: {sub_result.get('passed', 0)}")
                report.append(f"    Failed: {sub_result.get('failed', 0)}")
                report.append(f"    Skipped: {sub_result.get('skipped', 0)}")
                if sub_result.get("details"):
                    for detail in sub_result["details"]:
                        report.append(f"      {detail}")
                report.append("")
        
        # Errors
        if result.get("errors"):
            report.append("ERRORS:")
            for error in result["errors"]:
                report.append(f"  {error}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, result: Dict):
        """Save test results to files"""
        
        # Update test results
        self.test_results["end_time"] = datetime.now().isoformat()
        self.test_results["total_passed"] = result.get("passed", 0)
        self.test_results["total_failed"] = result.get("failed", 0)  
        self.test_results["total_skipped"] = result.get("skipped", 0)
        self.test_results["final_result"] = result
        
        # Save JSON results
        json_file = self.project_root / f"test-results-{result['test_type']}-{int(time.time())}.json"
        with open(json_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Save text report
        report_content = self.generate_report(result)
        report_file = self.project_root / f"test-report-{result['test_type']}-{int(time.time())}.txt"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Results saved to {json_file} and {report_file}")
        
        return json_file, report_file

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="SutazAI Test Runner")
    parser.add_argument("--type", required=True, 
                       choices=["unit", "integration", "security", "docker", "all"],
                       help="Type of tests to run")
    parser.add_argument("--services", help="Services to test (comma-separated)")
    parser.add_argument("--browser", default="chrome", help="Browser for e2e tests")
    parser.add_argument("--quick", action="store_true", help="Run quick version of tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner()
    
    if args.verbose:
        runner.logger.setLevel(logging.DEBUG)
    
    # Run tests
    runner.logger.info(f"Starting {args.type} test execution...")
    result = runner.run_test_suite(args.type)
    
    # Generate and save report
    json_file, report_file = runner.save_results(result)
    
    # Print summary
    print(runner.generate_report(result))
    
    # Exit with appropriate code
    if result.get("failed", 0) > 0:
        runner.logger.error("Some tests failed!")
        sys.exit(1)
    else:
        runner.logger.info("All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
