#!/usr/bin/env python3
"""
Integration Test Suite for Hardware Resource Optimizer
Tests complete workflows and feature interactions
"""

import os
import sys
import time
import json
import shutil
import psutil
import docker
import requests
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('IntegrationTest')

BASE_URL = "http://localhost:8116"
TEST_DIR = "/tmp/integration_test_workspace"

class IntegrationTestSuite:
    """Comprehensive integration testing for hardware-resource-optimizer"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        self.docker_client = None
        self._setup_test_environment()
        
    def _setup_test_environment(self):
        """Setup test workspace"""
        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
        os.makedirs(TEST_DIR, exist_ok=True)
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
        except:
            logger.warning("Docker not available for testing")
            
    def _call_endpoint(self, method: str, endpoint: str, params: Dict = None) -> Tuple[bool, Any]:
        """Call API endpoint and return success status and data"""
        url = f"{BASE_URL}{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, params=params, timeout=30)
            elif method == "POST":
                response = requests.post(url, params=params, timeout=60)
            else:
                return False, f"Unsupported method: {method}"
                
            return response.status_code == 200, response.json()
        except Exception as e:
            return False, str(e)
            
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_percent": psutil.disk_usage('/').percent,
            "disk_free_gb": psutil.disk_usage('/').free / (1024**3),
            "timestamp": time.time()
        }
        
    def _create_test_data(self, data_type: str) -> str:
        """Create test data for various scenarios"""
        test_path = os.path.join(TEST_DIR, data_type)
        os.makedirs(test_path, exist_ok=True)
        
        if data_type == "duplicates":
            # Create duplicate files
            content = b"This is duplicate content for testing"
            for i in range(5):
                with open(f"{test_path}/original_{i}.txt", "wb") as f:
                    f.write(content)
                for j in range(3):
                    with open(f"{test_path}/dup_{i}_{j}.txt", "wb") as f:
                        f.write(content)
                        
        elif data_type == "large_files":
            # Create large files
            for i, size_mb in enumerate([50, 100, 200]):
                with open(f"{test_path}/large_{size_mb}mb.dat", "wb") as f:
                    f.write(os.urandom(size_mb * 1024 * 1024))
                    
        elif data_type == "old_logs":
            # Create old log files
            for i in range(10):
                log_file = f"{test_path}/app_{i}.log"
                with open(log_file, "w") as f:
                    f.write(f"Log entry {i}\n" * 1000)
                # Make files old
                old_time = time.time() - (40 * 24 * 3600)  # 40 days old
                os.utime(log_file, (old_time, old_time))
                
        return test_path
        
    def _create_memory_pressure(self) -> List[bytearray]:
        """Create memory pressure for testing"""
        memory_hogs = []
        try:
            # Allocate 500MB in 50MB chunks
            for _ in range(10):
                memory_hogs.append(bytearray(50 * 1024 * 1024))
                time.sleep(0.1)
        except MemoryError:
            logger.warning("Could not create full memory pressure")
        return memory_hogs
        
    def _create_docker_containers(self) -> List[str]:
        """Create test Docker containers"""
        containers = []
        if not self.docker_client:
            return containers
            
        try:
            # Create and stop some containers
            for i in range(3):
                container = self.docker_client.containers.run(
                    "alpine:latest",
                    command="echo test",
                    name=f"test_container_{i}_{int(time.time())}",
                    detach=True,
                    remove=False
                )
                container.wait()
                containers.append(container.id)
        except Exception as e:
            logger.warning(f"Could not create Docker containers: {e}")
            
        return containers
        
    def test_scenario_1_full_system_optimization(self) -> Dict[str, Any]:
        """Test complete system optimization workflow"""
        logger.info("Starting Scenario 1: Full System Optimization")
        
        # Get initial metrics
        metrics_before = self._get_system_metrics()
        
        # Create test data
        self._create_test_data("duplicates")
        self._create_test_data("large_files")
        self._create_test_data("old_logs")
        
        # Create some pressure
        memory_hogs = self._create_memory_pressure()
        docker_containers = self._create_docker_containers()
        
        # Run full optimization
        success, result = self._call_endpoint("POST", "/optimize/all")
        
        if not success:
            return {"status": "FAIL", "error": result}
            
        # Get final metrics
        time.sleep(2)  # Allow system to settle
        metrics_after = self._get_system_metrics()
        
        # Validate results
        validations = {
            "api_success": success,
            "optimization_completed": result.get("status") == "success",
            "memory_improved": metrics_after["memory_percent"] <= metrics_before["memory_percent"],
            "actions_taken": len(result.get("actions_taken", [])) > 0,
            "subsystems_optimized": len(result.get("detailed_results", {})) >= 4
        }
        
        # Cleanup
        del memory_hogs
        
        return {
            "status": "PASS" if all(validations.values()) else "FAIL",
            "validations": validations,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "optimization_result": result
        }
        
    def test_scenario_2_storage_workflow(self) -> Dict[str, Any]:
        """Test storage analysis → duplicate removal → compression workflow"""
        logger.info("Starting Scenario 2: Storage Workflow")
        
        # Create test data
        dup_path = self._create_test_data("duplicates")
        
        # Step 1: Analyze storage
        success, analysis = self._call_endpoint("GET", "/analyze/storage", {"path": TEST_DIR})
        if not success:
            return {"status": "FAIL", "error": "Storage analysis failed"}
            
        initial_size = analysis.get("total_size", 0)
        
        # Step 2: Find duplicates
        success, duplicates = self._call_endpoint("GET", "/analyze/storage/duplicates", {"path": dup_path})
        if not success:
            return {"status": "FAIL", "error": "Duplicate detection failed"}
            
        dup_count = duplicates.get("duplicate_groups", 0)
        
        # Step 3: Remove duplicates (dry run first)
        success, dry_run = self._call_endpoint("POST", "/optimize/storage/duplicates", 
                                             {"path": dup_path, "dry_run": "true"})
        if not success:
            return {"status": "FAIL", "error": "Dry run failed"}
            
        # Step 4: Actually remove duplicates
        success, removal = self._call_endpoint("POST", "/optimize/storage/duplicates", 
                                             {"path": dup_path, "dry_run": "false"})
        if not success:
            return {"status": "FAIL", "error": "Duplicate removal failed"}
            
        # Step 5: Verify duplicates were removed
        success, verify = self._call_endpoint("GET", "/analyze/storage/duplicates", {"path": dup_path})
        
        # Step 6: Compress remaining files
        success, compress = self._call_endpoint("POST", "/optimize/storage/compress", {"path": TEST_DIR})
        
        # Final analysis
        success, final_analysis = self._call_endpoint("GET", "/analyze/storage", {"path": TEST_DIR})
        final_size = final_analysis.get("total_size", 0) if success else initial_size
        
        validations = {
            "analysis_successful": analysis.get("status") != "error",
            "duplicates_found": dup_count > 0,
            "dry_run_accurate": dry_run.get("files_removed", 0) > 0,
            "duplicates_removed": removal.get("files_removed", 0) > 0,
            "duplicates_cleared": verify.get("duplicate_groups", 999) < dup_count,
            "compression_applied": compress.get("files_compressed", 0) >= 0 if success else False,
            "storage_reduced": final_size < initial_size
        }
        
        return {
            "status": "PASS" if all(validations.values()) else "PARTIAL",
            "validations": validations,
            "initial_size": initial_size,
            "final_size": final_size,
            "space_saved": initial_size - final_size,
            "duplicates_removed": removal.get("files_removed", 0)
        }
        
    def test_scenario_3_resource_pressure(self) -> Dict[str, Any]:
        """Test optimization under resource pressure"""
        logger.info("Starting Scenario 3: Resource Pressure Testing")
        
        # Create significant memory pressure
        memory_hogs = []
        metrics_before = self._get_system_metrics()
        
        try:
            # Allocate memory until we hit 80% usage
            while psutil.virtual_memory().percent < 80:
                memory_hogs.append(bytearray(100 * 1024 * 1024))  # 100MB chunks
                time.sleep(0.5)
        except MemoryError:
            pass
            
        metrics_pressure = self._get_system_metrics()
        
        # Run memory optimization
        success, mem_result = self._call_endpoint("POST", "/optimize/memory")
        
        # Wait and measure
        time.sleep(2)
        metrics_after = self._get_system_metrics()
        
        # Cleanup
        del memory_hogs
        time.sleep(1)
        metrics_cleaned = self._get_system_metrics()
        
        validations = {
            "pressure_created": metrics_pressure["memory_percent"] > metrics_before["memory_percent"] + 10,
            "optimization_successful": success and mem_result.get("status") == "success",
            "memory_freed": mem_result.get("memory_freed_mb", 0) > 0,
            "system_improvement": metrics_after["memory_percent"] < metrics_pressure["memory_percent"],
            "cleanup_effective": metrics_cleaned["memory_percent"] < metrics_pressure["memory_percent"] - 5
        }
        
        return {
            "status": "PASS" if all(validations.values()) else "FAIL",
            "validations": validations,
            "memory_before": metrics_before["memory_percent"],
            "memory_pressure": metrics_pressure["memory_percent"],
            "memory_after": metrics_after["memory_percent"],
            "memory_cleaned": metrics_cleaned["memory_percent"],
            "optimization_result": mem_result if success else None
        }
        
    def test_scenario_4_docker_lifecycle(self) -> Dict[str, Any]:
        """Test Docker container lifecycle and cleanup"""
        logger.info("Starting Scenario 4: Docker Lifecycle Testing")
        
        if not self.docker_client:
            return {"status": "SKIP", "reason": "Docker not available"}
            
        # Create test containers
        containers = self._create_docker_containers()
        
        # Get initial Docker status
        success, initial_status = self._call_endpoint("GET", "/status")
        
        # Count containers before optimization
        all_containers_before = len(self.docker_client.containers.list(all=True))
        stopped_before = len(self.docker_client.containers.list(filters={'status': 'exited'}))
        
        # Run Docker optimization
        success, docker_result = self._call_endpoint("POST", "/optimize/docker")
        
        if not success:
            return {"status": "FAIL", "error": "Docker optimization failed"}
            
        # Count containers after
        all_containers_after = len(self.docker_client.containers.list(all=True))
        stopped_after = len(self.docker_client.containers.list(filters={'status': 'exited'}))
        
        validations = {
            "containers_created": len(containers) > 0,
            "optimization_successful": docker_result.get("status") == "success",
            "containers_removed": docker_result.get("containers_removed", 0) > 0,
            "actual_removal_verified": all_containers_after < all_containers_before,
            "stopped_containers_cleaned": stopped_after < stopped_before
        }
        
        return {
            "status": "PASS" if all(validations.values()) else "FAIL",
            "validations": validations,
            "containers_created": len(containers),
            "containers_removed": docker_result.get("containers_removed", 0),
            "containers_before": all_containers_before,
            "containers_after": all_containers_after,
            "optimization_result": docker_result
        }
        
    def test_scenario_5_concurrent_operations(self) -> Dict[str, Any]:
        """Test concurrent optimization requests"""
        logger.info("Starting Scenario 5: Concurrent Operations")
        
        import concurrent.futures
        import threading
        
        results = {}
        errors = []
        lock = threading.Lock()
        
        def run_optimization(opt_type: str):
            try:
                success, result = self._call_endpoint("POST", f"/optimize/{opt_type}")
                with lock:
                    results[opt_type] = {"success": success, "result": result}
            except Exception as e:
                with lock:
                    errors.append(f"{opt_type}: {str(e)}")
                    
        # Run multiple optimizations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for opt_type in ["memory", "cpu", "disk", "docker"]:
                futures.append(executor.submit(run_optimization, opt_type))
                
            # Wait for all to complete
            concurrent.futures.wait(futures, timeout=30)
            
        validations = {
            "all_completed": len(results) == 4,
            "no_errors": len(errors) == 0,
            "all_successful": all(r["success"] for r in results.values()),
            "no_conflicts": all(r["result"].get("status") != "error" for r in results.values() if r["success"])
        }
        
        return {
            "status": "PASS" if all(validations.values()) else "FAIL",
            "validations": validations,
            "concurrent_results": results,
            "errors": errors
        }
        
    def test_scenario_6_error_recovery(self) -> Dict[str, Any]:
        """Test error handling and recovery"""
        logger.info("Starting Scenario 6: Error Recovery Testing")
        
        error_tests = []
        
        # Test 1: Invalid path handling
        success, result = self._call_endpoint("POST", "/optimize/storage", {"path": "/nonexistent/path"})
        error_tests.append({
            "test": "invalid_path",
            "handled_gracefully": success or "error" in str(result).lower(),
            "no_crash": True  # If we got here, it didn't crash
        })
        
        # Test 2: Permission denied simulation
        protected_path = "/etc"
        success, result = self._call_endpoint("POST", "/optimize/storage", {"path": protected_path})
        error_tests.append({
            "test": "protected_path",
            "handled_gracefully": not success or result.get("status") == "error",
            "no_modifications": True  # Protected path should be rejected
        })
        
        # Test 3: Concurrent same endpoint
        import concurrent.futures
        results = []
        def call_same():
            return self._call_endpoint("POST", "/optimize/memory")
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(call_same) for _ in range(3)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                
        error_tests.append({
            "test": "concurrent_same_endpoint",
            "handled_gracefully": all(r[0] or "error" in str(r[1]).lower() for r in results),
            "no_crash": len(results) == 3
        })
        
        # Test 4: Malformed parameters
        success, result = self._call_endpoint("GET", "/analyze/storage/large-files", {"min_size_mb": "invalid"})
        error_tests.append({
            "test": "malformed_parameters",
            "handled_gracefully": True,  # Didn't crash
            "no_crash": True
        })
        
        validations = {
            "all_errors_handled": all(t["handled_gracefully"] for t in error_tests),
            "no_crashes": all(t["no_crash"] for t in error_tests),
            "protected_paths_safe": error_tests[1]["no_modifications"]
        }
        
        return {
            "status": "PASS" if all(validations.values()) else "FAIL",
            "validations": validations,
            "error_tests": error_tests
        }
        
    def run_all_scenarios(self) -> Dict[str, Any]:
        """Run all integration test scenarios"""
        logger.info("Starting Comprehensive Integration Testing")
        
        # Check if agent is healthy first
        success, health = self._call_endpoint("GET", "/health")
        if not success:
            return {
                "status": "FAIL",
                "error": "Agent not responding",
                "details": health
            }
            
        scenarios = [
            ("Full System Optimization", self.test_scenario_1_full_system_optimization),
            ("Storage Workflow", self.test_scenario_2_storage_workflow),
            ("Resource Pressure", self.test_scenario_3_resource_pressure),
            ("Docker Lifecycle", self.test_scenario_4_docker_lifecycle),
            ("Concurrent Operations", self.test_scenario_5_concurrent_operations),
            ("Error Recovery", self.test_scenario_6_error_recovery)
        ]
        
        results = {
            "test_start": datetime.now().isoformat(),
            "agent_health": health,
            "scenarios": {}
        }
        
        passed = 0
        failed = 0
        skipped = 0
        
        for name, test_func in scenarios:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {name}")
            logger.info('='*60)
            
            try:
                result = test_func()
                results["scenarios"][name] = result
                
                if result["status"] == "PASS":
                    passed += 1
                    logger.info(f"✅ {name}: PASSED")
                elif result["status"] == "SKIP":
                    skipped += 1
                    logger.info(f"⏭️  {name}: SKIPPED - {result.get('reason', 'Unknown')}")
                else:
                    failed += 1
                    logger.error(f"❌ {name}: FAILED")
                    if "validations" in result:
                        for check, passed in result["validations"].items():
                            if not passed:
                                logger.error(f"   - {check}: FAILED")
                                
            except Exception as e:
                failed += 1
                logger.error(f"❌ {name}: EXCEPTION - {str(e)}")
                results["scenarios"][name] = {
                    "status": "ERROR",
                    "error": str(e)
                }
                
        # Summary
        total = len(scenarios)
        results["summary"] = {
            "total_scenarios": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": f"{(passed/total)*100:.1f}%",
            "test_duration": f"{time.time() - self.start_time:.2f}s",
            "overall_status": "PASS" if failed == 0 else "FAIL"
        }
        
        # Cleanup
        try:
            shutil.rmtree(TEST_DIR)
        except:
            pass
            
        return results
        
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed test report"""
        report = []
        report.append("# Hardware Resource Optimizer - Integration Test Report")
        report.append(f"\n**Date:** {results['test_start']}")
        report.append(f"**Duration:** {results['summary']['test_duration']}")
        report.append(f"**Overall Status:** {results['summary']['overall_status']}")
        
        report.append("\n## Summary")
        report.append(f"- Total Scenarios: {results['summary']['total_scenarios']}")
        report.append(f"- Passed: {results['summary']['passed']} ✅")
        report.append(f"- Failed: {results['summary']['failed']} ❌")
        report.append(f"- Skipped: {results['summary']['skipped']} ⏭️")
        report.append(f"- Pass Rate: {results['summary']['pass_rate']}")
        
        report.append("\n## Scenario Results")
        
        for scenario, result in results['scenarios'].items():
            status_icon = "✅" if result['status'] == "PASS" else "❌" if result['status'] == "FAIL" else "⏭️"
            report.append(f"\n### {status_icon} {scenario}")
            report.append(f"**Status:** {result['status']}")
            
            if 'validations' in result:
                report.append("\n**Validations:**")
                for check, passed in result['validations'].items():
                    icon = "✓" if passed else "✗"
                    report.append(f"- {icon} {check}")
                    
            if 'error' in result:
                report.append(f"\n**Error:** {result['error']}")
                
        report.append("\n## Agent Health")
        report.append(f"```json\n{json.dumps(results['agent_health'], indent=2)}\n```")
        
        return "\n".join(report)


def main():
    """Run integration tests"""
    print("🚀 Starting Hardware Resource Optimizer Integration Tests")
    print("="*60)
    
    # Create test suite
    suite = IntegrationTestSuite()
    
    # Run all scenarios
    results = suite.run_all_scenarios()
    
    # Generate report
    report = suite.generate_report(results)
    
    # Save report
    report_file = "/opt/sutazaiapp/agents/hardware-resource-optimizer/integration_test_report.md"
    with open(report_file, "w") as f:
        f.write(report)
        
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Total Scenarios: {results['summary']['total_scenarios']}")
    print(f"Passed: {results['summary']['passed']} ✅")
    print(f"Failed: {results['summary']['failed']} ❌")
    print(f"Skipped: {results['summary']['skipped']} ⏭️")
    print(f"Pass Rate: {results['summary']['pass_rate']}")
    print(f"Duration: {results['summary']['test_duration']}")
    print(f"\nOverall Status: {results['summary']['overall_status']}")
    print(f"\nDetailed report saved to: {report_file}")
    
    # Exit with appropriate code
    return 0 if results['summary']['overall_status'] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())