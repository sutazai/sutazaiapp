#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for Hardware Resource Optimizer

Purpose: End-to-end testing of all optimization workflows
Usage: python integration_test_suite.py [--continuous] [--scenario SCENARIO_NAME]
Requirements: Agent running on port 8116, system access for validation
"""

import os
import sys
import json
import time
import psutil
import requests
import tempfile
import shutil
import subprocess
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integration_tests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntegrationTestSuite:
    """Comprehensive integration testing for hardware optimizer"""
    
    def __init__(self, base_url: str = "http://localhost:8116"):
        self.base_url = base_url
        self.test_results = defaultdict(list)
        self.system_baseline = None
        self.test_dir = tempfile.mkdtemp(prefix="hw_opt_test_")
        self.scenarios_passed = 0
        self.scenarios_failed = 0
        
        logger.info(f"Initialized test suite with base URL: {base_url}")
        logger.info(f"Test directory: {self.test_dir}")
    
    def setup(self):
        """Setup test environment and capture baseline"""
        logger.info("Setting up test environment...")
        
        # Capture system baseline
        self.system_baseline = self._capture_system_state()
        
        # Create test data
        self._create_test_data()
        
        # Verify agent is running
        if not self._verify_agent_health():
            raise RuntimeError("Hardware optimizer agent not responding")
        
        logger.info("Test environment setup complete")
    
    def teardown(self):
        """Cleanup test environment"""
        logger.info("Cleaning up test environment...")
        
        # Remove test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        # Generate final report
        self._generate_test_report()
        
        logger.info(f"Test suite complete. Passed: {self.scenarios_passed}, Failed: {self.scenarios_failed}")
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for comparison"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Count processes
        process_count = len(psutil.pids())
        
        # Docker containers if available
        docker_count = 0
        try:
            result = subprocess.run(['docker', 'ps', '-q'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                docker_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        except (AssertionError, Exception) as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / (1024**2),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
            "process_count": process_count,
            "docker_containers": docker_count,
            "open_files": len(psutil.Process().open_files())
        }
    
    def _create_test_data(self):
        """Create test data for optimization scenarios"""
        logger.info("Creating test data...")
        
        # Create duplicate files
        duplicate_dir = Path(self.test_dir) / "duplicates"
        duplicate_dir.mkdir(exist_ok=True)
        
        content = b"This is duplicate content for testing" * 1000
        for i in range(5):
            (duplicate_dir / f"duplicate_{i}.txt").write_bytes(content)
        
        # Create large files
        large_dir = Path(self.test_dir) / "large_files"
        large_dir.mkdir(exist_ok=True)
        
        # 10MB test file
        large_content = b"X" * (10 * 1024 * 1024)
        (large_dir / "large_test.bin").write_bytes(large_content)
        
        # Create cache-like structure
        cache_dir = Path(self.test_dir) / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        for i in range(20):
            (cache_dir / f"cache_file_{i}.tmp").write_text(f"Cache content {i}")
        
        # Create log files
        log_dir = Path(self.test_dir) / "logs"
        log_dir.mkdir(exist_ok=True)
        
        for i in range(10):
            log_file = log_dir / f"app_{i}.log"
            log_file.write_text(f"Log entry\n" * 100)
        
        logger.info(f"Created test data in {self.test_dir}")
    
    def _verify_agent_health(self) -> bool:
        """Verify agent is running and healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except (AssertionError, Exception) as e:
            logger.warning(f"Exception caught, returning: {e}")
            return False
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Tuple[bool, Any]:
        """Make API request and return success status and response"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, timeout=30, **kwargs)
            
            if response.status_code >= 200 and response.status_code < 300:
                return True, response.json() if response.text else None
            else:
                logger.error(f"Request failed: {method} {endpoint} - Status: {response.status_code}")
                return False, {"error": response.text}
        except Exception as e:
            logger.error(f"Request exception: {method} {endpoint} - {str(e)}")
            return False, {"error": str(e)}
    
    def run_scenario_1_full_system_optimization(self) -> bool:
        """Scenario 1: Full system optimization with all features"""
        logger.info("\n=== Running Scenario 1: Full System Optimization ===")
        
        start_state = self._capture_system_state()
        
        # Step 1: Analyze current state
        success, status = self._make_request("GET", "/status")
        if not success:
            logger.error("Failed to get system status")
            return False
        
        logger.info(f"Initial system state: CPU: {status['system_status']['cpu_percent']}%, "
                   f"Memory: {status['system_status']['memory_percent']}%, "
                   f"Disk: {status['system_status']['disk_percent']}%")
        
        # Step 2: Run full optimization
        logger.info("Running full system optimization...")
        success, result = self._make_request("POST", "/optimize/all", 
                                           params={"aggressive": "false"})
        
        if not success:
            logger.error("Full optimization failed")
            return False
        
        # Step 3: Verify improvements
        time.sleep(2)  # Allow optimizations to take effect
        end_state = self._capture_system_state()
        
        # Check for improvements
        memory_improved = end_state["memory_available_mb"] >= start_state["memory_available_mb"]
        processes_reduced = end_state["process_count"] <= start_state["process_count"]
        
        logger.info(f"Memory improvement: {memory_improved} "
                   f"({start_state['memory_available_mb']:.1f} -> {end_state['memory_available_mb']:.1f} MB)")
        logger.info(f"Process count: {start_state['process_count']} -> {end_state['process_count']}")
        
        # Record results
        self.test_results["scenario_1"].append({
            "timestamp": datetime.now().isoformat(),
            "success": memory_improved,
            "memory_freed_mb": end_state["memory_available_mb"] - start_state["memory_available_mb"],
            "processes_cleaned": start_state["process_count"] - end_state["process_count"],
            "optimization_results": result
        })
        
        scenario_passed = memory_improved and result.get("memory", {}).get("success", False)
        if scenario_passed:
            self.scenarios_passed += 1
            logger.info("✓ Scenario 1 PASSED")
        else:
            self.scenarios_failed += 1
            logger.error("✗ Scenario 1 FAILED")
        
        return scenario_passed
    
    def run_scenario_2_storage_workflow(self) -> bool:
        """Scenario 2: Complete storage optimization workflow"""
        logger.info("\n=== Running Scenario 2: Storage Workflow ===")
        
        # Step 1: Analyze storage
        logger.info("Analyzing storage...")
        success, analysis = self._make_request("GET", "/analyze/storage", 
                                             params={"path": self.test_dir})
        
        if not success:
            logger.error("Storage analysis failed")
            return False
        
        initial_usage = analysis["usage"]["used_gb"]
        logger.info(f"Initial storage usage: {initial_usage:.3f} GB")
        
        # Step 2: Find duplicates
        logger.info("Finding duplicate files...")
        success, duplicates = self._make_request("GET", "/analyze/storage/duplicates",
                                               params={"path": self.test_dir})
        
        if not success or not duplicates.get("duplicate_groups"):
            logger.error("Duplicate analysis failed or no duplicates found")
            return False
        
        duplicate_count = sum(len(group["files"]) - 1 for group in duplicates["duplicate_groups"])
        logger.info(f"Found {duplicate_count} duplicate files")
        
        # Step 3: Remove duplicates
        logger.info("Removing duplicates...")
        success, dup_result = self._make_request("POST", "/optimize/storage/duplicates",
                                               params={"path": self.test_dir, "dry_run": "false"})
        
        if not success:
            logger.error("Duplicate removal failed")
            return False
        
        # Step 4: Compress large files
        logger.info("Compressing large files...")
        success, compress_result = self._make_request("POST", "/optimize/storage/compress",
                                                    params={"path": self.test_dir, 
                                                           "min_size_mb": "5"})
        
        # Step 5: Verify improvements
        success, final_analysis = self._make_request("GET", "/analyze/storage",
                                                   params={"path": self.test_dir})
        
        if not success:
            logger.error("Final storage analysis failed")
            return False
        
        final_usage = final_analysis["usage"]["used_gb"]
        space_saved = initial_usage - final_usage
        
        logger.info(f"Storage optimization complete. Space saved: {space_saved:.3f} GB")
        
        # Record results
        self.test_results["scenario_2"].append({
            "timestamp": datetime.now().isoformat(),
            "success": space_saved > 0,
            "initial_usage_gb": initial_usage,
            "final_usage_gb": final_usage,
            "space_saved_gb": space_saved,
            "duplicates_removed": dup_result.get("files_removed", 0)
        })
        
        scenario_passed = space_saved > 0 and dup_result.get("success", False)
        if scenario_passed:
            self.scenarios_passed += 1
            logger.info("✓ Scenario 2 PASSED")
        else:
            self.scenarios_failed += 1
            logger.error("✗ Scenario 2 FAILED")
        
        return scenario_passed
    
    def run_scenario_3_resource_pressure(self) -> bool:
        """Scenario 3: Optimize under resource pressure"""
        logger.info("\n=== Running Scenario 3: Resource Pressure Optimization ===")
        
        # Create memory pressure
        logger.info("Creating resource pressure...")
        memory_hogs = []
        try:
            # Allocate memory blocks
            for i in range(5):
                memory_hogs.append(bytearray(50 * 1024 * 1024))  # 50MB each
            
            start_state = self._capture_system_state()
            logger.info(f"Memory usage under pressure: {start_state['memory_percent']:.1f}%")
            
            # Run memory optimization
            logger.info("Running memory optimization...")
            success, result = self._make_request("POST", "/optimize/memory",
                                               params={"threshold": "60"})
            
            if not success:
                logger.error("Memory optimization failed")
                return False
            
            # Clear our memory hogs
            memory_hogs.clear()
            import gc
            gc.collect()
            
            time.sleep(2)
            end_state = self._capture_system_state()
            
            memory_freed = start_state["memory_percent"] - end_state["memory_percent"]
            logger.info(f"Memory freed: {memory_freed:.1f}% "
                       f"({start_state['memory_percent']:.1f}% -> {end_state['memory_percent']:.1f}%)")
            
            # Record results
            self.test_results["scenario_3"].append({
                "timestamp": datetime.now().isoformat(),
                "success": memory_freed > 0,
                "initial_memory_percent": start_state["memory_percent"],
                "final_memory_percent": end_state["memory_percent"],
                "memory_freed_percent": memory_freed,
                "optimization_results": result
            })
            
            scenario_passed = memory_freed > 0 and result.get("success", False)
            if scenario_passed:
                self.scenarios_passed += 1
                logger.info("✓ Scenario 3 PASSED")
            else:
                self.scenarios_failed += 1
                logger.error("✗ Scenario 3 FAILED")
            
            return scenario_passed
            
        finally:
            # Ensure cleanup
            memory_hogs.clear()
    
    def run_scenario_4_docker_lifecycle(self) -> bool:
        """Scenario 4: Docker container lifecycle optimization"""
        logger.info("\n=== Running Scenario 4: Docker Lifecycle ===")
        
        # Check if Docker is available
        try:
            subprocess.run(['docker', '--version'], check=True, capture_output=True)
        except (AssertionError, Exception) as e:
            # TODO: Review this exception handling
            logger.error(f"Unexpected exception: {e}", exc_info=True)
            logger.warning("Docker not available, skipping scenario 4")
            return True  # Don't fail if Docker isn't available
        
        # Create test containers
        logger.info("Creating test containers...")
        test_containers = []
        
        try:
            # Create a few test containers
            for i in range(3):
                cmd = ['docker', 'run', '-d', '--name', f'hw_opt_test_{i}', 
                       'alpine', 'sleep', '300']
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    test_containers.append(f'hw_opt_test_{i}')
            
            if not test_containers:
                logger.warning("Could not create test containers")
                return True
            
            logger.info(f"Created {len(test_containers)} test containers")
            
            # Get initial Docker status
            success, initial_status = self._make_request("GET", "/status")
            initial_containers = initial_status.get("docker_status", {}).get("containers", 0)
            
            # Run Docker optimization
            logger.info("Running Docker optimization...")
            success, result = self._make_request("POST", "/optimize/docker",
                                               params={"remove_stopped": "true"})
            
            if not success:
                logger.error("Docker optimization failed")
                return False
            
            # Stop test containers
            for container in test_containers:
                subprocess.run(['docker', 'stop', container], capture_output=True)
            
            # Run optimization again to clean stopped containers
            time.sleep(2)
            success, cleanup_result = self._make_request("POST", "/optimize/docker",
                                                       params={"remove_stopped": "true"})
            
            # Verify cleanup
            success, final_status = self._make_request("GET", "/status")
            final_containers = final_status.get("docker_status", {}).get("containers", 0)
            
            containers_cleaned = initial_containers - final_containers
            logger.info(f"Containers cleaned: {containers_cleaned}")
            
            # Record results
            self.test_results["scenario_4"].append({
                "timestamp": datetime.now().isoformat(),
                "success": cleanup_result.get("success", False),
                "containers_removed": cleanup_result.get("containers_removed", 0),
                "images_removed": cleanup_result.get("images_removed", 0),
                "space_reclaimed_mb": cleanup_result.get("space_reclaimed_mb", 0)
            })
            
            scenario_passed = cleanup_result.get("success", False)
            if scenario_passed:
                self.scenarios_passed += 1
                logger.info("✓ Scenario 4 PASSED")
            else:
                self.scenarios_failed += 1
                logger.error("✗ Scenario 4 FAILED")
            
            return scenario_passed
            
        finally:
            # Cleanup any remaining test containers
            for container in test_containers:
                subprocess.run(['docker', 'rm', '-f', container], 
                             capture_output=True, stderr=subprocess.DEVNULL)
    
    def run_scenario_5_concurrent_operations(self) -> bool:
        """Scenario 5: Test concurrent optimization operations"""
        logger.info("\n=== Running Scenario 5: Concurrent Operations ===")
        
        results = []
        threads = []
        
        def run_optimization(opt_type: str, endpoint: str, params: dict = None):
            """Run optimization in thread"""
            logger.info(f"Starting concurrent {opt_type} optimization")
            success, result = self._make_request("POST", endpoint, params=params)
            results.append({
                "type": opt_type,
                "success": success,
                "result": result
            })
        
        # Launch concurrent optimizations
        operations = [
            ("memory", "/optimize/memory", {"threshold": "70"}),
            ("cpu", "/optimize/cpu", None),
            ("cache", "/optimize/storage/cache", {"path": self.test_dir}),
            ("logs", "/optimize/storage/logs", {"path": self.test_dir, "days": "1"})
        ]
        
        start_time = time.time()
        
        for opt_type, endpoint, params in operations:
            thread = threading.Thread(target=run_optimization, 
                                    args=(opt_type, endpoint, params))
            thread.start()
            threads.append(thread)
        
        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=60)
        
        duration = time.time() - start_time
        
        # Check results
        successful_ops = sum(1 for r in results if r["success"])
        logger.info(f"Concurrent operations completed in {duration:.1f}s")
        logger.info(f"Successful operations: {successful_ops}/{len(operations)}")
        
        # Record results
        self.test_results["scenario_5"].append({
            "timestamp": datetime.now().isoformat(),
            "success": successful_ops == len(operations),
            "duration_seconds": duration,
            "successful_operations": successful_ops,
            "total_operations": len(operations),
            "operation_results": results
        })
        
        scenario_passed = successful_ops == len(operations)
        if scenario_passed:
            self.scenarios_passed += 1
            logger.info("✓ Scenario 5 PASSED")
        else:
            self.scenarios_failed += 1
            logger.error("✗ Scenario 5 FAILED")
        
        return scenario_passed
    
    def run_scenario_6_error_recovery(self) -> bool:
        """Scenario 6: Test error handling and recovery"""
        logger.info("\n=== Running Scenario 6: Error Recovery ===")
        
        error_cases_passed = 0
        error_cases_total = 0
        
        # Test 1: Invalid path handling
        logger.info("Testing invalid path handling...")
        error_cases_total += 1
        success, result = self._make_request("GET", "/analyze/storage",
                                           params={"path": "/invalid/path/that/does/not/exist"})
        if not success or "error" in result:
            error_cases_passed += 1
            logger.info("✓ Invalid path handled correctly")
        else:
            logger.error("✗ Invalid path not handled properly")
        
        # Test 2: Permission denied simulation
        logger.info("Testing permission denied handling...")
        error_cases_total += 1
        protected_path = "/etc"
        success, result = self._make_request("POST", "/optimize/storage/duplicates",
                                           params={"path": protected_path, "dry_run": "false"})
        if result.get("files_removed", 0) == 0:  # Should not remove system files
            error_cases_passed += 1
            logger.info("✓ Protected path handled correctly")
        else:
            logger.error("✗ Protected path not handled properly")
        
        # Test 3: Resource limits
        logger.info("Testing resource limit handling...")
        error_cases_total += 1
        success, result = self._make_request("POST", "/optimize/storage/compress",
                                           params={"path": self.test_dir, 
                                                  "min_size_mb": "-1"})  # Invalid parameter
        if success:  # Should handle gracefully
            error_cases_passed += 1
            logger.info("✓ Invalid parameter handled correctly")
        else:
            logger.error("✗ Invalid parameter caused failure")
        
        # Test 4: Interruption recovery
        logger.info("Testing interruption recovery...")
        error_cases_total += 1
        
        # Start a large operation
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self._make_request, "POST", "/optimize/all",
                                   {"aggressive": "false"})
            
            # Give it a moment to start
            time.sleep(0.5)
            
            # Try another operation (should handle concurrent access)
            success, status = self._make_request("GET", "/status")
            
            if success:
                error_cases_passed += 1
                logger.info("✓ Concurrent access handled correctly")
            else:
                logger.error("✗ Concurrent access failed")
            
            # Wait for original operation
            future.result(timeout=30)
        
        # Record results
        self.test_results["scenario_6"].append({
            "timestamp": datetime.now().isoformat(),
            "success": error_cases_passed == error_cases_total,
            "error_cases_passed": error_cases_passed,
            "error_cases_total": error_cases_total,
            "pass_rate": error_cases_passed / error_cases_total * 100
        })
        
        scenario_passed = error_cases_passed == error_cases_total
        if scenario_passed:
            self.scenarios_passed += 1
            logger.info("✓ Scenario 6 PASSED")
        else:
            self.scenarios_failed += 1
            logger.error(f"✗ Scenario 6 FAILED ({error_cases_passed}/{error_cases_total} passed)")
        
        return scenario_passed
    
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        report = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "duration": time.time() - self.start_time if hasattr(self, 'start_time') else 0,
                "scenarios_passed": self.scenarios_passed,
                "scenarios_failed": self.scenarios_failed,
                "success_rate": (self.scenarios_passed / (self.scenarios_passed + self.scenarios_failed) * 100) 
                               if (self.scenarios_passed + self.scenarios_failed) > 0 else 0
            },
            "system_impact": {
                "baseline": self.system_baseline,
                "final": self._capture_system_state()
            },
            "scenario_results": dict(self.test_results)
        }
        
        # Save report
        report_file = f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        print(f"Total Scenarios: {self.scenarios_passed + self.scenarios_failed}")
        print(f"Passed: {self.scenarios_passed}")
        print(f"Failed: {self.scenarios_failed}")
        print(f"Success Rate: {report['test_run']['success_rate']:.1f}%")
        print("="*60)
    
    def run_all_scenarios(self):
        """Run all test scenarios"""
        self.start_time = time.time()
        
        scenarios = [
            self.run_scenario_1_full_system_optimization,
            self.run_scenario_2_storage_workflow,
            self.run_scenario_3_resource_pressure,
            self.run_scenario_4_docker_lifecycle,
            self.run_scenario_5_concurrent_operations,
            self.run_scenario_6_error_recovery
        ]
        
        for scenario in scenarios:
            try:
                scenario()
                time.sleep(2)  # Brief pause between scenarios
            except Exception as e:
                logger.error(f"Scenario failed with exception: {str(e)}")
                self.scenarios_failed += 1
    
    def run_single_scenario(self, scenario_name: str):
        """Run a single scenario by name"""
        self.start_time = time.time()
        
        scenario_map = {
            "full_system": self.run_scenario_1_full_system_optimization,
            "storage": self.run_scenario_2_storage_workflow,
            "pressure": self.run_scenario_3_resource_pressure,
            "docker": self.run_scenario_4_docker_lifecycle,
            "concurrent": self.run_scenario_5_concurrent_operations,
            "errors": self.run_scenario_6_error_recovery
        }
        
        if scenario_name in scenario_map:
            try:
                scenario_map[scenario_name]()
            except Exception as e:
                logger.error(f"Scenario failed with exception: {str(e)}")
                self.scenarios_failed += 1
        else:
            logger.error(f"Unknown scenario: {scenario_name}")
            logger.info(f"Available scenarios: {', '.join(scenario_map.keys())}")


def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hardware Optimizer Integration Tests")
    parser.add_argument("--continuous", action="store_true", 
                       help="Run tests continuously")
    parser.add_argument("--scenario", type=str, 
                       help="Run specific scenario (full_system, storage, pressure, docker, concurrent, errors)")
    parser.add_argument("--url", type=str, default="http://localhost:8116",
                       help="Agent base URL")
    
    args = parser.parse_args()
    
    # Create and run test suite
    suite = IntegrationTestSuite(base_url=args.url)
    
    try:
        suite.setup()
        
        if args.scenario:
            suite.run_single_scenario(args.scenario)
        else:
            suite.run_all_scenarios()
        
    except KeyboardInterrupt:
        logger.info("Test suite interrupted by user")
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
    finally:
        suite.teardown()


if __name__ == "__main__":
    main()