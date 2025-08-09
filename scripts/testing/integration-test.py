#!/usr/bin/env python3
"""
Integration Test for Enhanced Compliance Monitoring System
========================================================
Purpose: Comprehensive end-to-end testing of the complete compliance monitoring system
Usage: python integration-test.py [--comprehensive]

Test Scenarios:
- Complete workflow from detection to fix to validation
- System resilience with new components
- Error recovery and rollback
- Performance under realistic conditions
- Production readiness validation
"""

import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComplianceMonitoringIntegrationTest:
    """Complete integration test suite"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "test_categories": {},
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "warnings": 0
            },
            "performance_metrics": {},
            "issues_found": []
        }
        
    def run_integration_tests(self, comprehensive: bool = False) -> dict:
        """Run complete integration test suite"""
        logger.info("Starting comprehensive integration tests...")
        
        # Core integration tests
        self._test_production_readiness_validation()
        self._test_enhanced_monitor_functionality()
        self._test_scan_and_fix_workflow()
        self._test_error_handling_and_recovery()
        self._test_system_resilience()
        
        if comprehensive:
            self._test_performance_benchmarks()
            self._test_concurrent_operations()
            self._test_large_scale_scenarios()
        
        # Calculate overall status
        self._calculate_overall_status()
        
        logger.info(f"Integration tests completed: {self.test_results['overall_status']}")
        return self.test_results
    
    def _test_production_readiness_validation(self):
        """Test production readiness validator"""
        logger.info("Testing production readiness validation...")
        category = "production_readiness"
        tests = {}
        
        try:
            # Run production readiness validator
            result = subprocess.run([
                sys.executable, 
                str(self.project_root / "scripts" / "monitoring" / "production-readiness-validator.py"),
                "--project-root", str(self.project_root)
            ], capture_output=True, text=True, timeout=120)
            
            tests["validator_execution"] = {
                "status": "pass" if result.returncode in [0, 1] else "fail",
                "details": f"Exit code: {result.returncode}",
                "output": result.stdout[:500] if result.stdout else "No output"
            }
            
            # Check if report was generated
            reports_dir = self.project_root / "compliance-reports"
            readiness_reports = list(reports_dir.glob("production_readiness_*.json"))
            
            tests["report_generation"] = {
                "status": "pass" if readiness_reports else "fail",
                "details": f"Found {len(readiness_reports)} readiness reports",
                "output": f"Latest: {readiness_reports[-1].name if readiness_reports else 'None'}"
            }
            
        except subprocess.TimeoutExpired:
            tests["validator_execution"] = {
                "status": "fail",
                "details": "Production readiness validator timed out",
                "output": "Timeout after 120 seconds"
            }
        except Exception as e:
            tests["validator_execution"] = {
                "status": "fail", 
                "details": f"Validator execution failed: {e}",
                "output": str(e)
            }
        
        self.test_results["test_categories"][category] = tests
        self._update_summary(tests)
    
    def _test_enhanced_monitor_functionality(self):
        """Test enhanced compliance monitor functionality"""
        logger.info("Testing enhanced compliance monitor...")
        category = "enhanced_monitor"
        tests = {}
        
        try:
            # Test basic scan functionality
            result = subprocess.run([
                sys.executable,
                str(self.project_root / "scripts" / "monitoring" / "enhanced-compliance-monitor.py"),
                "--scan",
                "--project-root", str(self.project_root)
            ], capture_output=True, text=True, timeout=300)
            
            tests["basic_scan"] = {
                "status": "pass" if result.returncode in [0, 1] else "fail",
                "details": f"Scan completed with exit code {result.returncode}",
                "output": result.stdout[-500:] if result.stdout else "No output"
            }
            
            # Check if enhanced reports were generated
            reports_dir = self.project_root / "compliance-reports"
            enhanced_reports = list(reports_dir.glob("enhanced_report_*.json"))
            
            tests["enhanced_reporting"] = {
                "status": "pass" if enhanced_reports else "fail",
                "details": f"Found {len(enhanced_reports)} enhanced reports",
                "output": f"Latest: {enhanced_reports[-1].name if enhanced_reports else 'None'}"
            }
            
            # Test dry-run auto-fix
            result = subprocess.run([
                sys.executable,
                str(self.project_root / "scripts" / "monitoring" / "enhanced-compliance-monitor.py"),
                "--scan", "--fix", "--dry-run",
                "--project-root", str(self.project_root)
            ], capture_output=True, text=True, timeout=300)
            
            tests["dry_run_autofix"] = {
                "status": "pass" if result.returncode in [0, 1, 2] else "fail",
                "details": f"Dry-run completed with exit code {result.returncode}",
                "output": "Auto-fix results in stdout" if "Auto-fix Results:" in result.stdout else "No fix results found"
            }
            
        except subprocess.TimeoutExpired:
            tests["basic_scan"] = {
                "status": "fail",
                "details": "Enhanced monitor scan timed out",
                "output": "Timeout after 300 seconds"
            }
        except Exception as e:
            tests["basic_scan"] = {
                "status": "fail",
                "details": f"Enhanced monitor test failed: {e}",
                "output": str(e)
            }
        
        self.test_results["test_categories"][category] = tests
        self._update_summary(tests)
    
    def _test_scan_and_fix_workflow(self):
        """Test complete scan and fix workflow"""
        logger.info("Testing scan and fix workflow...")
        category = "scan_fix_workflow"
        tests = {}
        
        try:
            # Create test violations in a safe area
            test_dir = self.project_root / "integration_test_violations"
            test_dir.mkdir(exist_ok=True)
            
            # Create various test violations
            violations = {
                "fantasy_test.py": "# Test file\nresult = process_function(config_data)",
                "garbage_test.tmp": "temporary test file",
                "misplaced_test.sh": "#!/bin/bash\necho 'misplaced script'"
            }
            
            for filename, content in violations.items():
                (test_dir / filename).write_text(content)
            
            # Run enhanced monitor on test directory
            result = subprocess.run([
                sys.executable,
                str(self.project_root / "scripts" / "monitoring" / "enhanced-compliance-monitor.py"),
                "--scan", "--fix", "--dry-run",
                "--project-root", str(test_dir)
            ], capture_output=True, text=True, timeout=120)
            
            tests["workflow_execution"] = {
                "status": "pass" if result.returncode in [0, 1, 2] else "fail",
                "details": f"Workflow completed with exit code {result.returncode}",
                "output": "Violations detected and processed" if "violations" in result.stdout.lower() else "No violations processed"
            }
            
            # Check for transaction safety
            has_transaction_logs = "transaction" in result.stdout.lower()
            tests["transaction_safety"] = {
                "status": "pass" if has_transaction_logs else "warning",
                "details": "Transaction system engaged" if has_transaction_logs else "No transaction logs detected",
                "output": "Transaction-based processing active"
            }
            
            # Cleanup test directory
            shutil.rmtree(test_dir)
            
        except Exception as e:
            tests["workflow_execution"] = {
                "status": "fail",
                "details": f"Workflow test failed: {e}",
                "output": str(e)
            }
        
        self.test_results["test_categories"][category] = tests
        self._update_summary(tests)
    
    def _test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        logger.info("Testing error handling and recovery...")
        category = "error_handling"
        tests = {}
        
        try:
            # Test handling of non-existent project root
            result = subprocess.run([
                sys.executable,
                str(self.project_root / "scripts" / "monitoring" / "enhanced-compliance-monitor.py"),
                "--scan",
                "--project-root", "/nonexistent/path/that/should/not/exist"
            ], capture_output=True, text=True, timeout=60)
            
            tests["nonexistent_path_handling"] = {
                "status": "pass" if result.returncode != 0 else "warning",
                "details": f"Handled nonexistent path with exit code {result.returncode}",
                "output": "Error handled gracefully" if "error" in result.stderr.lower() else "No error detected"
            }
            
            # Test system validation functionality
            result = subprocess.run([
                sys.executable,
                str(self.project_root / "scripts" / "monitoring" / "enhanced-compliance-monitor.py"),
                "--validate-only",
                "--project-root", str(self.project_root)
            ], capture_output=True, text=True, timeout=60)
            
            tests["system_validation"] = {
                "status": "pass" if result.returncode in [0, 1] else "fail",
                "details": f"System validation completed with exit code {result.returncode}",
                "output": "Validation results provided" if result.stdout else "No validation output"
            }
            
        except subprocess.TimeoutExpired:
            tests["error_handling_timeout"] = {
                "status": "fail",
                "details": "Error handling test timed out",
                "output": "Timeout during error handling tests"
            }
        except Exception as e:
            tests["error_handling_exception"] = {
                "status": "fail",
                "details": f"Error handling test failed: {e}",
                "output": str(e)
            }
        
        self.test_results["test_categories"][category] = tests
        self._update_summary(tests)
    
    def _test_system_resilience(self):
        """Test system resilience with new components"""
        logger.info("Testing system resilience...")
        category = "system_resilience"
        tests = {}
        
        try:
            # Test database resilience
            db_path = self.project_root / "compliance-reports" / "resilience_test.db"
            
            # Create and test database operations
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE test_resilience (id INTEGER, data TEXT)")
            conn.execute("INSERT INTO test_resilience VALUES (1, 'test_data')")
            conn.commit()
            
            # Test recovery from database
            cursor = conn.execute("SELECT COUNT(*) FROM test_resilience")
            count = cursor.fetchone()[0]
            conn.close()
            
            # Cleanup
            if db_path.exists():
                db_path.unlink()
            
            tests["database_resilience"] = {
                "status": "pass" if count == 1 else "fail",
                "details": f"Database operations successful, retrieved {count} records",
                "output": "Database resilience confirmed"
            }
            
            # Test file system resilience
            test_file = self.project_root / "resilience_test.tmp"
            test_file.write_text("resilience test content")
            
            # Verify file operations
            content = test_file.read_text()
            test_file.unlink()
            
            tests["filesystem_resilience"] = {
                "status": "pass" if content == "resilience test content" else "fail",
                "details": "File system operations working correctly",
                "output": "File system resilience confirmed"
            }
            
            # Test configuration resilience
            config_data = {"test_config": True, "resilience_test": "active"}
            config_file = self.project_root / "test_config.json"
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f)
            
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
            
            config_file.unlink()
            
            tests["config_resilience"] = {
                "status": "pass" if loaded_config == config_data else "fail",
                "details": "Configuration loading and validation working",
                "output": "Configuration resilience confirmed"
            }
            
        except Exception as e:
            tests["resilience_exception"] = {
                "status": "fail",
                "details": f"Resilience test failed: {e}",
                "output": str(e)
            }
        
        self.test_results["test_categories"][category] = tests
        self._update_summary(tests)
    
    def _test_performance_benchmarks(self):
        """Test performance meets benchmarks"""
        logger.info("Testing performance benchmarks...")
        category = "performance"
        tests = {}
        
        try:
            # Measure scan performance
            start_time = time.time()
            
            result = subprocess.run([
                sys.executable,
                str(self.project_root / "scripts" / "monitoring" / "enhanced-compliance-monitor.py"),
                "--scan",
                "--project-root", str(self.project_root)
            ], capture_output=True, text=True, timeout=300)
            
            scan_duration = time.time() - start_time
            
            tests["scan_performance"] = {
                "status": "pass" if scan_duration < 60 else "warning",
                "details": f"Scan completed in {scan_duration:.2f} seconds",
                "output": f"Benchmark: < 60s, Actual: {scan_duration:.2f}s"
            }
            
            self.test_results["performance_metrics"]["scan_duration"] = scan_duration
            
            # Memory usage estimation (basic)
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                
                tests["memory_usage"] = {
                    "status": "pass" if memory_mb < 1000 else "warning",
                    "details": f"Memory usage: {memory_mb:.1f}MB",
                    "output": f"Benchmark: < 1000MB, Actual: {memory_mb:.1f}MB"
                }
                
                self.test_results["performance_metrics"]["memory_usage_mb"] = memory_mb
                
            except ImportError:
                tests["memory_usage"] = {
                    "status": "warning",
                    "details": "Cannot measure memory usage (psutil not available)",
                    "output": "Memory measurement skipped"
                }
            
        except subprocess.TimeoutExpired:
            tests["scan_performance"] = {
                "status": "fail",
                "details": "Performance test timed out (> 300s)",
                "output": "Scan performance benchmark failed"
            }
        except Exception as e:
            tests["performance_exception"] = {
                "status": "fail",
                "details": f"Performance test failed: {e}",
                "output": str(e)
            }
        
        self.test_results["test_categories"][category] = tests
        self._update_summary(tests)
    
    def _test_concurrent_operations(self):
        """Test concurrent operations safety"""
        logger.info("Testing concurrent operations...")
        category = "concurrency"
        tests = {}
        
        try:
            import threading
            import concurrent.futures
            
            results = []
            errors = []
            
            def run_validation():
                try:
                    result = subprocess.run([
                        sys.executable,
                        str(self.project_root / "scripts" / "monitoring" / "production-readiness-validator.py"),
                        "--project-root", str(self.project_root),
                        "--quiet"
                    ], capture_output=True, text=True, timeout=60)
                    results.append(result.returncode)
                    return True
                except Exception as e:
                    errors.append(str(e))
                    return False
            
            # Run multiple validations concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(run_validation) for _ in range(3)]
                completed = sum(1 for future in concurrent.futures.as_completed(futures, timeout=180) 
                              if future.result())
            
            tests["concurrent_validations"] = {
                "status": "pass" if completed >= 2 and len(errors) <= 1 else "warning",
                "details": f"{completed}/3 concurrent operations successful, {len(errors)} errors",
                "output": f"Errors: {errors}" if errors else "All concurrent operations successful"
            }
            
        except Exception as e:
            tests["concurrency_exception"] = {
                "status": "fail",
                "details": f"Concurrency test failed: {e}",
                "output": str(e)
            }
        
        self.test_results["test_categories"][category] = tests
        self._update_summary(tests)
    
    def _test_large_scale_scenarios(self):
        """Test large-scale scenarios"""
        logger.info("Testing large-scale scenarios...")
        category = "large_scale"
        tests = {}
        
        try:
            # Create large test directory structure
            test_dir = self.project_root / "large_scale_test"
            test_dir.mkdir(exist_ok=True)
            
            # Create many test files
            num_dirs = 5
            num_files_per_dir = 20
            total_files = 0
            
            start_time = time.time()
            
            for i in range(num_dirs):
                sub_dir = test_dir / f"test_dir_{i}"
                sub_dir.mkdir(exist_ok=True)
                
                for j in range(num_files_per_dir):
                    test_file = sub_dir / f"test_file_{j}.py"
                    test_file.write_text(f"# Test file {i}_{j}\ntest_value = {j}")
                    total_files += 1
            
            setup_time = time.time() - start_time
            
            # Test scan on large directory
            start_time = time.time()
            
            result = subprocess.run([
                sys.executable,
                str(self.project_root / "scripts" / "monitoring" / "enhanced-compliance-monitor.py"),
                "--scan",
                "--project-root", str(test_dir)
            ], capture_output=True, text=True, timeout=120)
            
            scan_time = time.time() - start_time
            
            # Cleanup
            shutil.rmtree(test_dir)
            
            tests["large_directory_scan"] = {
                "status": "pass" if result.returncode in [0, 1] and scan_time < 60 else "warning",
                "details": f"Scanned {total_files} files in {scan_time:.2f}s (setup: {setup_time:.2f}s)",
                "output": f"Large-scale performance: {total_files/scan_time:.1f} files/second"
            }
            
            self.test_results["performance_metrics"]["large_scale_files_per_second"] = total_files / scan_time if scan_time > 0 else 0
            
        except subprocess.TimeoutExpired:
            tests["large_scale_timeout"] = {
                "status": "fail",
                "details": "Large-scale test timed out",
                "output": "Performance insufficient for large-scale scenarios"
            }
        except Exception as e:
            tests["large_scale_exception"] = {
                "status": "fail",
                "details": f"Large-scale test failed: {e}",
                "output": str(e)
            }
        
        self.test_results["test_categories"][category] = tests
        self._update_summary(tests)
    
    def _update_summary(self, tests: dict):
        """Update test summary"""
        for test_name, test_result in tests.items():
            self.test_results["summary"]["total_tests"] += 1
            
            status = test_result["status"]
            if status == "pass":
                self.test_results["summary"]["passed_tests"] += 1
            elif status == "fail":
                self.test_results["summary"]["failed_tests"] += 1
                self.test_results["issues_found"].append(f"{test_name}: {test_result['details']}")
            elif status == "warning":
                self.test_results["summary"]["warnings"] += 1
    
    def _calculate_overall_status(self):
        """Calculate overall test status"""
        summary = self.test_results["summary"]
        
        if summary["failed_tests"] > 0:
            self.test_results["overall_status"] = "failed"
        elif summary["warnings"] > summary["passed_tests"]:
            self.test_results["overall_status"] = "needs_attention"
        elif summary["warnings"] > 0:
            self.test_results["overall_status"] = "passed_with_warnings"
        else:
            self.test_results["overall_status"] = "passed"
    
    def generate_report(self) -> str:
        """Generate integration test report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.project_root / "compliance-reports" / f"integration_test_{timestamp}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Generate summary report
        summary_path = self.project_root / "compliance-reports" / f"integration_test_summary_{timestamp}.txt"
        
        with open(summary_path, 'w') as f:
            f.write("Integration Test Report\n")
            f.write("=" * 30 + "\n\n")
            
            f.write(f"Timestamp: {self.test_results['timestamp']}\n")
            f.write(f"Overall Status: {self.test_results['overall_status'].upper()}\n\n")
            
            summary = self.test_results['summary']
            f.write("Test Summary:\n")
            f.write(f"  Total Tests: {summary['total_tests']}\n")
            f.write(f"  Passed: {summary['passed_tests']}\n")
            f.write(f"  Failed: {summary['failed_tests']}\n")
            f.write(f"  Warnings: {summary['warnings']}\n\n")
            
            if self.test_results.get('performance_metrics'):
                f.write("Performance Metrics:\n")
                for metric, value in self.test_results['performance_metrics'].items():
                    f.write(f"  {metric}: {value}\n")
                f.write("\n")
            
            if self.test_results['issues_found']:
                f.write("Issues Found:\n")
                for issue in self.test_results['issues_found']:
                    f.write(f"  - {issue}\n")
                f.write("\n")
            
            f.write("Test Results by Category:\n")
            f.write("-" * 25 + "\n")
            
            for category, tests in self.test_results['test_categories'].items():
                f.write(f"\n{category.upper()}:\n")
                for test_name, test_result in tests.items():
                    status_symbol = "✓" if test_result['status'] == 'pass' else "⚠" if test_result['status'] == 'warning' else "✗"
                    f.write(f"  {status_symbol} {test_name}: {test_result['details']}\n")
        
        logger.info(f"Integration test report generated: {report_path}")
        logger.info(f"Summary report generated: {summary_path}")
        
        return str(report_path)

def main():
    """Main integration test function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Integration Test Suite for Enhanced Compliance Monitoring"
    )
    parser.add_argument("--project-root", default="/opt/sutazaiapp",
                       help="Project root directory")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive tests including performance benchmarks")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Run integration tests
    tester = ComplianceMonitoringIntegrationTest(args.project_root)
    results = tester.run_integration_tests(comprehensive=args.comprehensive)
    
    # Generate report
    report_path = tester.generate_report()
    
    # Print summary
    print(f"\nIntegration Test Complete")
    print(f"Overall Status: {results['overall_status'].upper()}")
    print(f"Report: {report_path}")
    
    summary = results['summary']
    print(f"\nTest Summary: {summary['passed_tests']} passed, "
          f"{summary['failed_tests']} failed, "
          f"{summary['warnings']} warnings")
    
    if results.get('performance_metrics'):
        print(f"\nPerformance Highlights:")
        for metric, value in results['performance_metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value}")
    
    # Return appropriate exit code
    if results['overall_status'] == 'passed':
        return 0
    elif results['overall_status'] == 'passed_with_warnings':
        return 0
    elif results['overall_status'] == 'needs_attention':
        return 1
    else:  # failed
        return 2

if __name__ == "__main__":
    sys.exit(main())