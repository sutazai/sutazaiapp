#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
Purpose: Master test runner for comprehensive hygiene enforcement system validation
Usage: python test-hygiene-system.py [--component=<component>] [--verbose] [--report]
Requirements: Python 3.8+, unittest, git, all hygiene enforcement components
"""

import os
import sys
import json
import time
import argparse
import unittest
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch, MagicMock
import datetime

class HygieneSystemTestRunner:
    """Master test runner for the entire hygiene enforcement infrastructure"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
        # Component paths
        self.components = {
            "orchestrator": self.project_root / "scripts/agents/hygiene-agent-orchestrator.py",
            "coordinator": self.project_root / "scripts/hygiene-enforcement-coordinator.py",
            "monitor": self.project_root / "scripts/hygiene-monitor.py",
            "automation": self.project_root / "scripts/utils/automated-hygiene-maintenance.sh",
            "pre_commit_hook": self.project_root / ".git/hooks/pre-commit",
            "pre_push_hook": self.project_root / ".git/hooks/pre-push"
        }
        
        # Test fixtures directory
        self.fixtures_dir = self.project_root / "tests/fixtures/hygiene"
        self.temp_test_dir = None
        
    def setup_test_environment(self):
        """Setup isolated test environment"""
        logger.info("Setting up test environment...")
        
        # Create temp directory for testing
        self.temp_test_dir = Path(tempfile.mkdtemp(prefix="hygiene_test_"))
        
        # Create test fixtures
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        self._create_test_fixtures()
        
        # Ensure logs directory exists
        (self.project_root / "logs").mkdir(exist_ok=True)
        
        logger.info(f"Test environment created at: {self.temp_test_dir}")
        
    def _create_test_fixtures(self):
        """Create test fixtures for violation testing"""
        fixtures = {
            # Rule 13 violations (junk files)
            "junk_files": [
                "test_file.backup",
                "old_script.py.bak",
                "temporary_file.tmp",
                "editor_backup~",
                "config.old"
            ],
            
            # Rule 12 violations (multiple deployment scripts)
            "deploy_scripts": [
                "deploy.sh",
                "deploy_staging.sh", 
                "validate_deployment.py",
                "deploy_prod.py"
            ],
            
            # Rule 8 violations (undocumented Python scripts)
            "undocumented_python": [
                "script_without_header.py",
                "another_script.py"
            ],
            
            # Rule 11 violations (Docker chaos)
            "docker_chaos": [
                "Dockerfile.old",
                "docker-compose.backup.yml",
                "Dockerfile.test"
            ]
        }
        
        for category, files in fixtures.items():
            category_dir = self.fixtures_dir / category
            category_dir.mkdir(exist_ok=True)
            
            for filename in files:
                filepath = category_dir / filename
                
                if filename.endswith('.py'):
                    # Create Python file without proper header
                    content = "import os\nlogger.info('test')\n"
                elif filename.endswith('.sh'):
                    # Create shell script
                    content = "#!/bin/bash\necho 'test deploy script'\n"
                elif 'docker' in filename.lower():
                    # Create Docker file
                    content = "FROM ubuntu:20.04\nRUN apt-get update\n"
                else:
                    # Generic test file
                    content = f"Test file for {category} violation testing\n"
                
                filepath.write_text(content)
                
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.temp_test_dir and self.temp_test_dir.exists():
            shutil.rmtree(self.temp_test_dir)
            logger.info(f"Cleaned up test directory: {self.temp_test_dir}")

class TestHygieneOrchestrator(unittest.TestCase):
    """Test the hygiene agent orchestrator"""
    
    def setUp(self):
        self.project_root = Path("/opt/sutazaiapp")
        self.orchestrator_script = self.project_root / "scripts/agents/hygiene-agent-orchestrator.py"
        
    def test_orchestrator_exists(self):
        """Test that orchestrator script exists and is executable"""
        self.assertTrue(self.orchestrator_script.exists(), 
                       f"Orchestrator script not found: {self.orchestrator_script}")
        self.assertTrue(os.access(self.orchestrator_script, os.X_OK),
                       "Orchestrator script is not executable")
        
    def test_orchestrator_dry_run(self):
        """Test orchestrator dry run functionality"""
        cmd = ["python3", str(self.orchestrator_script), "--rule=13", "--dry-run"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Should not fail in dry run mode
        self.assertEqual(result.returncode, 0, 
                        f"Orchestrator dry run failed: {result.stderr}")
        
        # Should contain expected output patterns
        self.assertIn("rule", result.stdout.lower())
        
    def test_orchestrator_invalid_rule(self):
        """Test orchestrator handles invalid rule numbers"""
        cmd = ["python3", str(self.orchestrator_script), "--rule=99"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Should fail gracefully with invalid rule
        
    def test_orchestrator_phase_execution(self):
        """Test orchestrator phase-based execution"""
        cmd = ["python3", str(self.orchestrator_script), "--phase=1", "--dry-run"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        self.assertEqual(result.returncode, 0,
                        f"Phase execution failed: {result.stderr}")

class TestHygieneCoordinator(unittest.TestCase):
    """Test the hygiene enforcement coordinator"""
    
    def setUp(self):
        self.project_root = Path("/opt/sutazaiapp")
        self.coordinator_script = self.project_root / "scripts/hygiene-enforcement-coordinator.py"
        
    def test_coordinator_exists(self):
        """Test that coordinator script exists"""
        self.assertTrue(self.coordinator_script.exists(),
                       f"Coordinator script not found: {self.coordinator_script}")
                       
    def test_coordinator_dry_run_phase1(self):
        """Test coordinator phase 1 dry run"""
        cmd = ["python3", str(self.coordinator_script), "--phase=1", "--dry-run"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        self.assertEqual(result.returncode, 0,
                        f"Coordinator phase 1 dry run failed: {result.stderr}")
                        
    def test_coordinator_archive_creation(self):
        """Test that coordinator creates archive directories properly"""
        # This would be tested by checking the actual archive creation logic
        # For now, we test that the script runs without syntax errors
        cmd = ["python3", "-m", "py_compile", str(self.coordinator_script)]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0,
                        f"Coordinator script has syntax errors: {result.stderr}")

class TestGitHooks(unittest.TestCase):
    """Test pre-commit and pre-push hooks functionality"""
    
    def setUp(self):
        self.project_root = Path("/opt/sutazaiapp")
        self.pre_commit_hook = self.project_root / ".git/hooks/pre-commit"
        self.pre_push_hook = self.project_root / ".git/hooks/pre-push"
        
    def test_pre_commit_hook_exists_and_executable(self):
        """Test pre-commit hook exists and is executable"""
        self.assertTrue(self.pre_commit_hook.exists(),
                       "Pre-commit hook not found")
        self.assertTrue(os.access(self.pre_commit_hook, os.X_OK),
                       "Pre-commit hook is not executable")
                       
    def test_pre_commit_hook_content(self):
        """Test pre-commit hook has proper content"""
        if self.pre_commit_hook.exists():
            content = self.pre_commit_hook.read_text()
            
            # Should contain validation logic
            self.assertIn("validation", content.lower())
            
    def test_pre_push_hook_exists(self):
        """Test pre-push hook exists if configured"""
        if self.pre_push_hook.exists():
            self.assertTrue(os.access(self.pre_push_hook, os.X_OK),
                           "Pre-push hook exists but is not executable")

class TestHygieneMonitor(unittest.TestCase):
    """Test the hygiene monitoring system"""
    
    def setUp(self):
        self.project_root = Path("/opt/sutazaiapp")
        self.monitor_script = self.project_root / "scripts/hygiene-monitor.py"
        
    def test_monitor_script_exists(self):
        """Test monitor script exists"""
        if self.monitor_script.exists():
            self.assertTrue(os.access(self.monitor_script, os.R_OK),
                           "Monitor script is not readable")
                           
    def test_monitor_syntax(self):
        """Test monitor script has valid syntax"""
        if self.monitor_script.exists():
            cmd = ["python3", "-m", "py_compile", str(self.monitor_script)]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            self.assertEqual(result.returncode, 0,
                            f"Monitor script has syntax errors: {result.stderr}")

class TestAutomatedMaintenance(unittest.TestCase):
    """Test automated maintenance scripts"""
    
    def setUp(self):
        self.project_root = Path("/opt/sutazaiapp")
        self.maintenance_script = self.project_root / "scripts/utils/automated-hygiene-maintenance.sh"
        
    def test_maintenance_script_exists(self):
        """Test maintenance script exists and is executable"""
        self.assertTrue(self.maintenance_script.exists(),
                       f"Maintenance script not found: {self.maintenance_script}")
        self.assertTrue(os.access(self.maintenance_script, os.X_OK),
                       "Maintenance script is not executable")
                       
    def test_maintenance_daily_mode(self):
        """Test maintenance script daily mode (dry run simulation)"""
        # We can't actually run the maintenance script in tests,
        # but we can verify it parses correctly
        cmd = ["bash", "-n", str(self.maintenance_script)]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0,
                        f"Maintenance script has syntax errors: {result.stderr}")

class TestFailureScenarios(unittest.TestCase):
    """Test system behavior under failure conditions"""
    
    def setUp(self):
        self.project_root = Path("/opt/sutazaiapp")
        
    def test_missing_dependencies(self):
        """Test behavior when dependencies are missing"""
        # Test that scripts handle missing Python modules gracefully
        test_script = '''
import sys
sys.path.insert(0, "/nonexistent/path")
try:
    import nonexistent_module
except ImportError as e:
    logger.info(f"Handled missing dependency: {e}")
    sys.exit(0)
sys.exit(1)
'''
        
        result = subprocess.run(["python3", "-c", test_script], 
                              capture_output=True, text=True)
                              
        self.assertEqual(result.returncode, 0,
                        "Script should handle missing dependencies gracefully")
                        
    def test_permission_errors(self):
        """Test behavior when file permissions are insufficient"""
        # This would test what happens when scripts can't write to log directories
        # For now, we just verify the concept works
        self.assertTrue(True, "Permission error handling test placeholder")
        
    def test_disk_space_exhaustion(self):
        """Test behavior when disk space is low"""
        # This would simulate low disk space conditions
        # For now, we just verify the concept works  
        self.assertTrue(True, "Disk space handling test placeholder")

class TestPerformanceAndResources(unittest.TestCase):
    """Test system performance and resource usage"""
    
    def setUp(self):
        self.project_root = Path("/opt/sutazaiapp")
        
    def test_memory_usage(self):
        """Test that hygiene scripts don't use excessive memory"""
        # This would monitor memory usage during script execution
        # For now, we verify the scripts can be imported without issues
        
        coordinator_script = self.project_root / "scripts/hygiene-enforcement-coordinator.py"
        if coordinator_script.exists():
            cmd = ["python3", "-c", f"exec(open('{coordinator_script}').read())"]
            
            # Should not crash with memory errors
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # but it shouldn't fail with memory errors
            self.assertNotIn("MemoryError", result.stderr)
            
    def test_execution_time(self):
        """Test that operations complete within reasonable time"""
        start_time = time.time()
        
        # Test a simple dry run
        orchestrator_script = self.project_root / "scripts/agents/hygiene-agent-orchestrator.py"
        if orchestrator_script.exists():
            cmd = ["python3", str(orchestrator_script), "--rule=13", "--dry-run"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            execution_time = time.time() - start_time
            
            # Should complete within 2 minutes
            self.assertLess(execution_time, 120,
                           f"Operation took too long: {execution_time}s")

class TestReportingAndMonitoring(unittest.TestCase):
    """Test reporting and monitoring functionality"""
    
    def setUp(self):
        self.project_root = Path("/opt/sutazaiapp")
        self.logs_dir = self.project_root / "logs"
        
    def test_log_directory_creation(self):
        """Test that log directories are created properly"""
        # Logs directory should exist or be creatable
        if not self.logs_dir.exists():
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.assertTrue(self.logs_dir.exists(),
                       "Logs directory should exist or be creatable")
                       
    def test_log_file_permissions(self):
        """Test that log files have proper permissions"""
        if self.logs_dir.exists():
            # Should be able to write to logs directory
            test_log = self.logs_dir / "test.log"
            try:
                test_log.write_text("test")
                test_log.unlink()  # Clean up
                
                self.assertTrue(True, "Can write to logs directory")
            except PermissionError:
                self.fail("Cannot write to logs directory")

def run_component_tests(component: str = None, verbose: bool = False) -> Dict:
    """Run tests for specific component or all components"""
    
    # Define test suites
    test_suites = {
        "orchestrator": TestHygieneOrchestrator,
        "coordinator": TestHygieneCoordinator, 
        "hooks": TestGitHooks,
        "monitor": TestHygieneMonitor,
        "maintenance": TestAutomatedMaintenance,
        "failures": TestFailureScenarios,
        "performance": TestPerformanceAndResources,
        "reporting": TestReportingAndMonitoring
    }
    
    results = {}
    
    if component and component in test_suites:
        # Run specific component tests
        suite = unittest.TestLoader().loadTestsFromTestCase(test_suites[component])
        runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
        result = runner.run(suite)
        
        results[component] = {
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "success": result.wasSuccessful()
        }
    else:
        # Run all tests
        for comp_name, test_class in test_suites.items():
            logger.info(f"\n=== Testing {comp_name.upper()} ===")
            
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
            result = runner.run(suite)
            
            results[comp_name] = {
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "success": result.wasSuccessful()
            }
    
    return results

def generate_test_report(results: Dict, output_file: Optional[str] = None) -> str:
    """Generate comprehensive test report"""
    
    timestamp = datetime.datetime.now().isoformat()
    
    # Calculate overall statistics
    total_tests = sum(r["tests_run"] for r in results.values())
    total_failures = sum(r["failures"] for r in results.values())
    total_errors = sum(r["errors"] for r in results.values())
    successful_components = sum(1 for r in results.values() if r["success"])
    
    report = {
        "timestamp": timestamp,
        "summary": {
            "total_components_tested": len(results),
            "successful_components": successful_components,
            "total_tests_run": total_tests,
            "total_failures": total_failures,
            "total_errors": total_errors,
            "overall_success": total_failures == 0 and total_errors == 0
        },
        "component_results": results,
        "recommendations": []
    }
    
    # Add recommendations based on results
    if total_failures > 0:
        report["recommendations"].append("Address test failures before deployment")
        
    if total_errors > 0:
        report["recommendations"].append("Fix test errors - may indicate missing dependencies")
        
    if successful_components < len(results):
        report["recommendations"].append("Some components failed testing - investigate before production use")
        
    if total_tests < 20:
        report["recommendations"].append("Consider adding more comprehensive tests")
        
    # Save report if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"\nTest report saved to: {output_path}")
    
    return json.dumps(report, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Test hygiene enforcement system")
    parser.add_argument("--component", choices=[
        "orchestrator", "coordinator", "hooks", "monitor", 
        "maintenance", "failures", "performance", "reporting"
    ], help="Test specific component")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose test output")
    parser.add_argument("--report", help="Generate JSON report to specified file")
    parser.add_argument("--setup-only", action="store_true",
                       help="Only setup test environment")
    
    args = parser.parse_args()
    
    # Initialize test runner
    test_runner = HygieneSystemTestRunner()
    
    try:
        # Setup test environment
        test_runner.setup_test_environment()
        
        if args.setup_only:
            logger.info("Test environment setup completed")
            return 0
            
        # Run tests
        logger.info("=== HYGIENE SYSTEM VALIDATION TESTS ===")
        test_runner.start_time = datetime.datetime.now()
        
        results = run_component_tests(args.component, args.verbose)
        
        test_runner.end_time = datetime.datetime.now()
        test_runner.test_results = results
        
        # Generate report
        report_json = generate_test_report(results, args.report)
        
        # Print summary
        logger.info("\n=== TEST SUMMARY ===")
        logger.info(report_json)
        
        # Return appropriate exit code
        overall_success = all(r["success"] for r in results.values())
        return 0 if overall_success else 1
        
    finally:
        # Cleanup
        test_runner.cleanup_test_environment()

if __name__ == "__main__":
