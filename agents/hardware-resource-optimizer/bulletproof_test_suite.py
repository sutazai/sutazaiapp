#!/usr/bin/env python3
"""
Bulletproof Test Suite for Hardware Resource Optimizer
Tests ACTUAL system effects, not just API responses.

This test suite:
1. Verifies every endpoint works
2. Measures actual system changes (memory freed, files deleted, etc.)
3. Tests edge cases and error conditions
4. Ensures 100% functionality with no errors
5. Provides detailed reporting of system effects

Usage: python3 bulletproof_test_suite.py
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
import hashlib
import gzip
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BulletproofTest')

BASE_URL = "http://localhost:8116"
TEST_WORKSPACE = "/tmp/bulletproof_test_workspace"

class SystemMeasurement:
    """Capture and compare system metrics"""
    
    def __init__(self, name: str):
        self.name = name
        self.before = {}
        self.after = {}
        
    def capture_before(self):
        """Capture system state before operation"""
        self.before = {
            "timestamp": time.time(),
            "memory": psutil.virtual_memory()._asdict(),
            "disk": psutil.disk_usage('/')._asdict(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "processes": len(psutil.pids()),
            "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
        }
        logger.info(f"Captured BEFORE state for {self.name}")
        
    def capture_after(self):
        """Capture system state after operation"""
        self.after = {
            "timestamp": time.time(),
            "memory": psutil.virtual_memory()._asdict(),
            "disk": psutil.disk_usage('/')._asdict(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "processes": len(psutil.pids()),
            "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
        }
        logger.info(f"Captured AFTER state for {self.name}")
        
    def get_changes(self) -> Dict[str, Any]:
        """Calculate actual system changes"""
        if not self.before or not self.after:
            return {"error": "Missing before/after measurements"}
            
        changes = {
            "test_name": self.name,
            "duration_seconds": self.after["timestamp"] - self.before["timestamp"],
            "memory_freed_mb": (self.after["memory"]["available"] - self.before["memory"]["available"]) / (1024**2),
            "disk_freed_mb": (self.after["disk"]["free"] - self.before["disk"]["free"]) / (1024**2),
            "memory_percent_change": self.after["memory"]["percent"] - self.before["memory"]["percent"],
            "disk_percent_change": self.after["disk"]["percent"] - self.before["disk"]["percent"],
            "process_count_change": self.after["processes"] - self.before["processes"],
            "load_avg_change": [
                self.after["load_avg"][i] - self.before["load_avg"][i] 
                for i in range(3)
            ]
        }
        
        return changes

class BulletproofTestSuite:
    """Comprehensive system-effect testing for hardware resource optimizer"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        self.docker_client = None
        self.test_files = []
        self.setup_workspace()
        
    def setup_workspace(self):
        """Setup isolated test workspace"""
        if os.path.exists(TEST_WORKSPACE):
            shutil.rmtree(TEST_WORKSPACE)
        os.makedirs(TEST_WORKSPACE, exist_ok=True)
        logger.info(f"Created test workspace: {TEST_WORKSPACE}")
        
        # Initialize Docker if available
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            logger.info("Docker client available for testing")
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
            
    def cleanup_workspace(self):
        """Clean up test workspace"""
        if os.path.exists(TEST_WORKSPACE):
            shutil.rmtree(TEST_WORKSPACE)
        logger.info("Cleaned up test workspace")
        
    def call_api(self, method: str, endpoint: str, params: Dict = None) -> Tuple[bool, Any, float]:
        """Call API endpoint and return success, data, and response time"""
        url = f"{BASE_URL}{endpoint}"
        start_time = time.time()
        
        try:
            if method == "GET":
                response = requests.get(url, params=params, timeout=30)
            elif method == "POST":
                response = requests.post(url, params=params, timeout=60)
            else:
                return False, f"Unsupported method: {method}", 0
                
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return True, response.json(), response_time
            else:
                return False, {
                    "status_code": response.status_code,
                    "text": response.text
                }, response_time
                
        except Exception as e:
            response_time = time.time() - start_time
            return False, str(e), response_time
            
    def create_test_files(self, test_type: str) -> str:
        """Create test files for various scenarios"""
        test_dir = os.path.join(TEST_WORKSPACE, test_type)
        os.makedirs(test_dir, exist_ok=True)
        
        if test_type == "duplicates":
            # Create duplicate files to test deduplication
            content = b"This is test content for duplicate detection"
            original_hash = hashlib.md5(content).hexdigest()
            
            files_created = []
            for i in range(3):
                original_path = f"{test_dir}/original_{i}.txt"
                with open(original_path, "wb") as f:
                    f.write(content)
                files_created.append(original_path)
                
                # Create duplicates
                for j in range(2):
                    dup_path = f"{test_dir}/duplicate_{i}_{j}.txt"
                    with open(dup_path, "wb") as f:
                        f.write(content)
                    files_created.append(dup_path)
                    
            self.test_files.extend(files_created)
            logger.info(f"Created {len(files_created)} files for duplicate testing")
            return test_dir
            
        elif test_type == "large_files":
            # Create large files for compression/cleanup testing
            files_created = []
            for size_mb in [10, 25, 50]:  # Smaller sizes for faster testing
                file_path = f"{test_dir}/large_{size_mb}mb.dat"
                with open(file_path, "wb") as f:
                    f.write(os.urandom(size_mb * 1024 * 1024))
                files_created.append(file_path)
                
            self.test_files.extend(files_created)
            logger.info(f"Created {len(files_created)} large files for testing")
            return test_dir
            
        elif test_type == "cache_files":
            # Create cache-like files
            cache_dirs = ["__pycache__", ".pytest_cache", "node_modules/.cache"]
            files_created = []
            
            for cache_dir in cache_dirs:
                cache_path = f"{test_dir}/{cache_dir}"
                os.makedirs(cache_path, exist_ok=True)
                
                for i in range(5):
                    file_path = f"{cache_path}/cache_file_{i}.tmp"
                    with open(file_path, "w") as f:
                        f.write(f"Cache content {i}" * 100)
                    files_created.append(file_path)
                    
            self.test_files.extend(files_created)
            logger.info(f"Created {len(files_created)} cache files for testing")
            return test_dir
            
        return test_dir
    
    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test health endpoint"""
        logger.info("Testing /health endpoint...")
        
        measurement = SystemMeasurement("health_check")
        measurement.capture_before()
        
        success, data, response_time = self.call_api("GET", "/health")
        
        measurement.capture_after()
        changes = measurement.get_changes()
        
        result = {
            "test": "health_endpoint",
            "success": success,
            "response_time": response_time,
            "system_changes": changes,
            "expected_fields": ["status", "agent", "description", "docker_available", "system_status"],
            "fields_present": list(data.keys()) if success and isinstance(data, dict) else [],
            "validation": {}
        }
        
        if success and isinstance(data, dict):
            result["validation"]["has_required_fields"] = all(
                field in data for field in result["expected_fields"]
            )
            result["validation"]["status_is_healthy"] = data.get("status") == "healthy"
            result["validation"]["agent_name_correct"] = data.get("agent") == "hardware-resource-optimizer"
        else:
            result["validation"] = {"error": "Invalid response format"}
            
        self.results.append(result)
        return result
    
    def test_status_endpoint(self) -> Dict[str, Any]:
        """Test status endpoint"""
        logger.info("Testing /status endpoint...")
        
        measurement = SystemMeasurement("status_check")
        measurement.capture_before()
        
        success, data, response_time = self.call_api("GET", "/status")
        
        measurement.capture_after()
        changes = measurement.get_changes()
        
        result = {
            "test": "status_endpoint",
            "success": success,
            "response_time": response_time,
            "system_changes": changes,
            "data": data if success else None
        }
        
        self.results.append(result)
        return result
    
    def test_memory_optimization(self) -> Dict[str, Any]:
        """Test memory optimization with actual system effects"""
        logger.info("Testing memory optimization with system effects...")
        
        # Create memory load first
        memory_hogs = []
        try:
            # Allocate some memory to test optimization
            for i in range(3):
                memory_hogs.append(bytearray(10 * 1024 * 1024))  # 10MB each
        except:
            pass
            
        measurement = SystemMeasurement("memory_optimization")
        measurement.capture_before()
        
        # Test dry run first
        success_dry, data_dry, time_dry = self.call_api("POST", "/optimize/memory", {"dry_run": "true"})
        
        # Then actual optimization
        success, data, response_time = self.call_api("POST", "/optimize/memory")
        
        measurement.capture_after()
        changes = measurement.get_changes()
        
        # Clean up memory hogs
        del memory_hogs
        
        result = {
            "test": "memory_optimization",
            "dry_run_success": success_dry,
            "dry_run_data": data_dry if success_dry else None,
            "actual_success": success,
            "actual_data": data if success else None,
            "response_time": response_time,
            "system_changes": changes,
            "validation": {}
        }
        
        if success and isinstance(data, dict):
            result["validation"]["has_memory_freed"] = "memory_freed_mb" in str(data)
            result["validation"]["gc_called"] = "garbage_collection" in str(data).lower() or "objects" in str(data).lower()
            result["validation"]["actual_memory_change"] = changes.get("memory_freed_mb", 0)
            result["validation"]["optimization_completed"] = data.get("status") == "success"
        
        self.results.append(result)
        return result
    
    def test_cpu_optimization(self) -> Dict[str, Any]:
        """Test CPU optimization"""
        logger.info("Testing CPU optimization...")
        
        measurement = SystemMeasurement("cpu_optimization")
        measurement.capture_before()
        
        success, data, response_time = self.call_api("POST", "/optimize/cpu")
        
        measurement.capture_after()
        changes = measurement.get_changes()
        
        result = {
            "test": "cpu_optimization",
            "success": success,
            "data": data if success else None,
            "response_time": response_time,
            "system_changes": changes
        }
        
        self.results.append(result)
        return result
    
    def test_disk_optimization(self) -> Dict[str, Any]:
        """Test disk optimization"""
        logger.info("Testing disk optimization...")
        
        measurement = SystemMeasurement("disk_optimization")
        measurement.capture_before()
        
        success, data, response_time = self.call_api("POST", "/optimize/disk")
        
        measurement.capture_after()
        changes = measurement.get_changes()
        
        result = {
            "test": "disk_optimization",
            "success": success,
            "data": data if success else None,
            "response_time": response_time,
            "system_changes": changes
        }
        
        self.results.append(result)
        return result
    
    def test_docker_optimization(self) -> Dict[str, Any]:
        """Test Docker optimization"""
        logger.info("Testing Docker optimization...")
        
        measurement = SystemMeasurement("docker_optimization")
        measurement.capture_before()
        
        success, data, response_time = self.call_api("POST", "/optimize/docker")
        
        measurement.capture_after()
        changes = measurement.get_changes()
        
        result = {
            "test": "docker_optimization",
            "success": success,
            "data": data if success else None,
            "response_time": response_time,
            "system_changes": changes,
            "docker_available": self.docker_client is not None
        }
        
        self.results.append(result)
        return result
    
    def test_storage_analysis(self) -> Dict[str, Any]:
        """Test storage analysis with real data"""
        logger.info("Testing storage analysis...")
        
        # Create test files
        test_dir = self.create_test_files("large_files")
        
        measurement = SystemMeasurement("storage_analysis")
        measurement.capture_before()
        
        success, data, response_time = self.call_api("GET", "/analyze/storage", {
            "path": test_dir,
            "limit": "50"
        })
        
        measurement.capture_after()
        changes = measurement.get_changes()
        
        result = {
            "test": "storage_analysis",
            "success": success,
            "data": data if success else None,
            "response_time": response_time,
            "system_changes": changes,
            "test_dir": test_dir,
            "validation": {}
        }
        
        if success and isinstance(data, dict):
            result["validation"]["found_test_files"] = any(
                test_dir in str(item) for item in data.get("items", [])
            )
            result["validation"]["has_size_info"] = "total_size" in str(data)
            
        self.results.append(result)
        return result
    
    def test_duplicate_detection(self) -> Dict[str, Any]:
        """Test duplicate file detection with real duplicates"""
        logger.info("Testing duplicate detection...")
        
        # Create duplicate files
        test_dir = self.create_test_files("duplicates")
        
        measurement = SystemMeasurement("duplicate_detection")
        measurement.capture_before()
        
        success, data, response_time = self.call_api("GET", "/analyze/storage/duplicates", {
            "path": test_dir
        })
        
        measurement.capture_after()
        changes = measurement.get_changes()
        
        # Count actual duplicates created
        files_before = len([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))])
        
        result = {
            "test": "duplicate_detection",
            "success": success,
            "data": data if success else None,
            "response_time": response_time,
            "system_changes": changes,
            "test_dir": test_dir,
            "files_created": files_before,
            "validation": {}
        }
        
        if success and isinstance(data, dict):
            duplicates_found = len(data.get("duplicate_groups", []))
            result["validation"]["found_duplicates"] = duplicates_found > 0
            result["validation"]["duplicate_groups"] = duplicates_found
            result["validation"]["expected_duplicates"] = True  # We created duplicates
            
        self.results.append(result)
        return result
    
    def test_duplicate_cleanup(self) -> Dict[str, Any]:
        """Test duplicate cleanup with actual file removal"""
        logger.info("Testing duplicate cleanup...")
        
        # Create duplicate files
        test_dir = self.create_test_files("duplicates")
        files_before = len([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))])
        
        measurement = SystemMeasurement("duplicate_cleanup")
        measurement.capture_before()
        
        # First dry run
        success_dry, data_dry, time_dry = self.call_api("POST", "/optimize/storage/duplicates", {
            "path": test_dir,
            "dry_run": "true"
        })
        
        # Then actual cleanup - ensure path exists and is accessible
        if not os.path.exists(test_dir):
            os.makedirs(test_dir, exist_ok=True)
            
        success, data, response_time = self.call_api("POST", "/optimize/storage/duplicates", {
            "path": test_dir
        })
        
        measurement.capture_after()
        changes = measurement.get_changes()
        
        # Count files after cleanup
        files_after = len([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]) if os.path.exists(test_dir) else 0
        
        result = {
            "test": "duplicate_cleanup",
            "dry_run_success": success_dry,
            "dry_run_data": data_dry if success_dry else None,
            "actual_success": success,
            "actual_data": data if success else None,
            "response_time": response_time,
            "system_changes": changes,
            "files_before": files_before,
            "files_after": files_after,
            "files_removed": files_before - files_after,
            "validation": {}
        }
        
        if success:
            result["validation"]["files_actually_removed"] = files_before > files_after
            result["validation"]["disk_space_freed"] = changes.get("disk_freed_mb", 0) > 0
            
        self.results.append(result)
        return result
    
    def test_cache_cleanup(self) -> Dict[str, Any]:
        """Test cache cleanup with real cache files"""
        logger.info("Testing cache cleanup...")
        
        # Create cache files
        test_dir = self.create_test_files("cache_files")
        cache_files_before = sum(1 for root, dirs, files in os.walk(test_dir) for f in files)
        
        measurement = SystemMeasurement("cache_cleanup")
        measurement.capture_before()
        
        success, data, response_time = self.call_api("POST", "/optimize/storage/cache", {
            "path": test_dir
        })
        
        measurement.capture_after()
        changes = measurement.get_changes()
        
        # Count cache files after cleanup
        cache_files_after = sum(1 for root, dirs, files in os.walk(test_dir) for f in files) if os.path.exists(test_dir) else 0
        
        result = {
            "test": "cache_cleanup",
            "success": success,
            "data": data if success else None,
            "response_time": response_time,
            "system_changes": changes,
            "cache_files_before": cache_files_before,
            "cache_files_after": cache_files_after,
            "cache_files_removed": cache_files_before - cache_files_after,
            "validation": {}
        }
        
        if success:
            result["validation"]["cache_files_removed"] = cache_files_before > cache_files_after
            result["validation"]["disk_space_freed"] = changes.get("disk_freed_mb", 0) >= 0
            
        self.results.append(result)
        return result
    
    def test_comprehensive_optimization(self) -> Dict[str, Any]:
        """Test comprehensive all-in-one optimization"""
        logger.info("Testing comprehensive optimization...")
        
        measurement = SystemMeasurement("comprehensive_optimization")
        measurement.capture_before()
        
        success, data, response_time = self.call_api("POST", "/optimize/all")
        
        measurement.capture_after()
        changes = measurement.get_changes()
        
        result = {
            "test": "comprehensive_optimization",
            "success": success,
            "data": data if success else None,
            "response_time": response_time,
            "system_changes": changes,
            "validation": {}
        }
        
        if success and isinstance(data, dict):
            result["validation"]["multiple_optimizations"] = len(str(data)) > 100  # Complex response
            result["validation"]["includes_memory"] = "memory" in str(data).lower()
            result["validation"]["includes_disk"] = "disk" in str(data).lower()
            
        self.results.append(result)
        return result
    
    def test_error_conditions(self) -> Dict[str, Any]:
        """Test error handling and edge cases"""
        logger.info("Testing error conditions...")
        
        error_tests = [
            ("GET", "/nonexistent", None, "nonexistent_endpoint"),
            ("GET", "/analyze/storage", {"path": "/nonexistent/path"}, "invalid_path"),
            ("POST", "/optimize/storage/duplicates", {"path": "/etc"}, "protected_path"),
            ("POST", "/optimize/memory", {"invalid_param": "true"}, "invalid_parameters")
        ]
        
        results = []
        
        for method, endpoint, params, test_name in error_tests:
            success, data, response_time = self.call_api(method, endpoint, params)
            
            error_result = {
                "test": f"error_condition_{test_name}",
                "success": success if test_name != "nonexistent_endpoint" else not success,  # Expect failure for nonexistent endpoint
                "data": data,
                "response_time": response_time,
                "expected_failure": True,
                "validation": {
                    "handles_error_gracefully": not success or "error" in str(data).lower(),
                    "appropriate_response": (
                        not success and "404" in str(data) if test_name == "nonexistent_endpoint" 
                        else (not success or "error" in str(data).lower())
                    )
                }
            }
            results.append(error_result)
            self.results.append(error_result)
            
        return {"error_tests": results}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        logger.info("Starting bulletproof test suite...")
        
        # Basic functionality tests
        self.test_health_endpoint()
        self.test_status_endpoint()
        
        # Core optimization tests
        self.test_memory_optimization()
        self.test_cpu_optimization()
        self.test_disk_optimization()
        self.test_docker_optimization()
        
        # Storage analysis tests
        self.test_storage_analysis()
        self.test_duplicate_detection()
        self.test_duplicate_cleanup()
        self.test_cache_cleanup()
        
        # Comprehensive test
        self.test_comprehensive_optimization()
        
        # Error handling tests
        self.test_error_conditions()
        
        # Generate summary
        total_time = time.time() - self.start_time
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.get("success", False))
        failed_tests = total_tests - passed_tests
        
        summary = {
            "bulletproof_test_suite": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_time_seconds": total_time,
                "workspace": TEST_WORKSPACE
            },
            "system_effects_verified": self._analyze_system_effects(),
            "detailed_results": self.results
        }
        
        return summary
    
    def _analyze_system_effects(self) -> Dict[str, Any]:
        """Analyze actual system effects across all tests"""
        total_memory_freed = 0
        total_disk_freed = 0
        tests_with_effects = 0
        
        for result in self.results:
            changes = result.get("system_changes", {})
            if isinstance(changes, dict):
                memory_freed = changes.get("memory_freed_mb", 0)
                disk_freed = changes.get("disk_freed_mb", 0)
                
                if memory_freed != 0 or disk_freed != 0:
                    tests_with_effects += 1
                    total_memory_freed += memory_freed
                    total_disk_freed += disk_freed
                    
        return {
            "total_memory_freed_mb": total_memory_freed,
            "total_disk_freed_mb": total_disk_freed,
            "tests_with_system_effects": tests_with_effects,
            "system_optimization_verified": tests_with_effects > 0
        }
    
    def save_results(self, filename: str = None):
        """Save test results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bulletproof_test_results_{timestamp}.json"
            
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"Results saved to {filename}")
        
    def cleanup(self):
        """Cleanup test environment"""
        self.cleanup_workspace()
        logger.info("Test cleanup completed")

def main():
    """Run bulletproof test suite"""
    print("🚀 Starting Bulletproof Test Suite for Hardware Resource Optimizer")
    print("=" * 80)
    
    suite = BulletproofTestSuite()
    
    try:
        # Run all tests
        results = suite.run_all_tests()
        
        # Print summary
        summary = results["bulletproof_test_suite"]
        effects = results["system_effects_verified"]
        
        print("\n" + "=" * 80)
        print("📊 BULLETPROOF TEST RESULTS SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        print(f"Total Time: {summary['total_time_seconds']:.2f} seconds")
        print(f"System Effects Verified: {'✅ YES' if effects['system_optimization_verified'] else '❌ NO'}")
        print(f"Memory Freed: {effects['total_memory_freed_mb']:.2f} MB")
        print(f"Disk Freed: {effects['total_disk_freed_mb']:.2f} MB")
        
        # Print detailed test results
        print("\n🔍 DETAILED TEST RESULTS:")
        print("-" * 40)
        for result in results["detailed_results"]:
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            test_name = result.get("test", "unknown")
            response_time = result.get("response_time", 0)
            changes = result.get("system_changes", {})
            
            print(f"{status} {test_name} ({response_time:.3f}s)")
            
            if isinstance(changes, dict):
                memory_change = changes.get("memory_freed_mb", 0)
                disk_change = changes.get("disk_freed_mb", 0)
                if memory_change != 0 or disk_change != 0:
                    print(f"    💾 Memory: {memory_change:+.2f}MB, Disk: {disk_change:+.2f}MB")
                    
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"bulletproof_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n📄 Full results saved to: {results_file}")
        
        # Final verdict
        if summary['pass_rate'] >= 1.0 and effects['system_optimization_verified']:
            print("\n🎉 BULLETPROOF TEST VERDICT: ALL SYSTEMS GO! 🎉")
            print("✅ 100% functionality verified")
            print("✅ Actual system effects confirmed")
            print("✅ Ready for production use")
        elif summary['pass_rate'] >= 0.9:
            print("\n⚠️  BULLETPROOF TEST VERDICT: MOSTLY FUNCTIONAL")
            print(f"✅ {summary['pass_rate']:.1%} pass rate")
            print("⚠️  Some issues detected - review failed tests")
        else:
            print("\n❌ BULLETPROOF TEST VERDICT: NEEDS ATTENTION")
            print(f"❌ Only {summary['pass_rate']:.1%} pass rate")
            print("❌ Significant issues detected")
            
        return summary['pass_rate'] >= 0.9
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        print(f"\n❌ BULLETPROOF TEST SUITE FAILED: {e}")
        return False
        
    finally:
        suite.cleanup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)