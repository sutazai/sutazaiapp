#!/usr/bin/env python3
"""
Comprehensive End-to-End Test Framework for Hardware Resource Optimizer
=======================================================================

This framework provides comprehensive testing for all agent endpoints with:
- Parameter validation and edge case testing
- Performance benchmarking and stress testing
- Error handling and recovery validation
- Integration testing with real system resources
- Concurrent load testing
- Safety mechanism verification
- Detailed reporting and metrics

Author: QA Team Lead
Version: 1.0.0
"""

import os
import sys
import json
import time
import asyncio
import aiohttp
import pytest
import tempfile
import shutil
import subprocess
import threading
import statistics
import hashlib
import gzip
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging
import psutil
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('E2E_Framework')

class E2ETestFramework:
    """Comprehensive End-to-End Testing Framework"""
    
    def __init__(self, base_url: str = "http://localhost:8116", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        
        # Test configuration
        self.test_results = []
        self.performance_metrics = {}
        self.error_scenarios = []
        self.safety_checks = []
        
        # Test data setup
        self.temp_test_dir = None
        self.setup_test_environment()
        
        logger.info(f"E2E Framework initialized for {base_url}")
    
    def setup_test_environment(self):
        """Setup isolated test environment"""
        self.temp_test_dir = tempfile.mkdtemp(prefix="e2e_test_")
        logger.info(f"Test environment created at: {self.temp_test_dir}")
        
        # Create test files and directories
        self._create_test_data()
    
    def _create_test_data(self):
        """Create comprehensive test data for all scenarios"""
        # Create directory structure
        test_dirs = [
            "temp_files", "cache_files", "log_files", 
            "large_files", "duplicate_files", "compressible_files"
        ]
        
        for dir_name in test_dirs:
            os.makedirs(os.path.join(self.temp_test_dir, dir_name), exist_ok=True)
        
        # Create various test files
        self._create_temp_files()
        self._create_cache_files()
        self._create_log_files()
        self._create_large_files()
        self._create_duplicate_files()
        self._create_compressible_files()
    
    def _create_temp_files(self):
        """Create temporary files for cleanup testing"""
        temp_dir = os.path.join(self.temp_test_dir, "temp_files")
        
        # Recent files (should not be deleted)
        with open(os.path.join(temp_dir, "recent_temp.txt"), "w") as f:
            f.write("Recent temporary file content")
        
        # Old files (should be deleted)
        old_file = os.path.join(temp_dir, "old_temp.txt")
        with open(old_file, "w") as f:
            f.write("Old temporary file content")
        
        # Set old modification time (4 days ago)
        old_time = time.time() - (4 * 24 * 3600)
        os.utime(old_file, (old_time, old_time))
    
    def _create_cache_files(self):
        """Create cache files for cache cleanup testing"""
        cache_dir = os.path.join(self.temp_test_dir, "cache_files")
        
        cache_files = ["browser_cache.dat", "app_cache.tmp", "thumbnail_cache.jpg"]
        for cache_file in cache_files:
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                f.write(b"Cache data content" * 100)
    
    def _create_log_files(self):
        """Create log files for log optimization testing"""
        log_dir = os.path.join(self.temp_test_dir, "log_files")
        
        # Recent log (should be compressed)
        with open(os.path.join(log_dir, "recent.log"), "w") as f:
            f.write("INFO: Recent log entries\n" * 1000)
        
        # Old log (should be deleted)
        old_log = os.path.join(log_dir, "old.log")
        with open(old_log, "w") as f:
            f.write("INFO: Old log entries\n" * 1000)
        
        # Set old modification time (100 days ago)
        old_time = time.time() - (100 * 24 * 3600)
        os.utime(old_log, (old_time, old_time))
    
    def _create_large_files(self):
        """Create large files for large file analysis"""
        large_dir = os.path.join(self.temp_test_dir, "large_files")
        
        # Create 150MB file
        large_file = os.path.join(large_dir, "large_file.bin")
        with open(large_file, "wb") as f:
            f.write(b"X" * (150 * 1024 * 1024))
        
        # Create 50MB file (below threshold)
        medium_file = os.path.join(large_dir, "medium_file.bin")
        with open(medium_file, "wb") as f:
            f.write(b"Y" * (50 * 1024 * 1024))
    
    def _create_duplicate_files(self):
        """Create duplicate files for duplicate analysis"""
        dup_dir = os.path.join(self.temp_test_dir, "duplicate_files")
        
        content1 = "Duplicate content 1\n" * 100
        content2 = "Duplicate content 2\n" * 200
        
        # Create duplicates of content1
        for i in range(3):
            with open(os.path.join(dup_dir, f"duplicate1_{i}.txt"), "w") as f:
                f.write(content1)
        
        # Create duplicates of content2
        for i in range(2):
            with open(os.path.join(dup_dir, f"duplicate2_{i}.txt"), "w") as f:
                f.write(content2)
        
        # Create unique file
        with open(os.path.join(dup_dir, "unique.txt"), "w") as f:
            f.write("Unique content")
    
    def _create_compressible_files(self):
        """Create files suitable for compression testing"""
        compress_dir = os.path.join(self.temp_test_dir, "compressible_files")
        
        # Create old compressible files
        files = [
            ("data.csv", "name,age,city\n" + "John,25,NYC\n" * 1000),
            ("log.txt", "INFO: Log entry\n" * 1000),
            ("config.json", json.dumps({"key_" + str(i): f"value_{i}" for i in range(100)}, indent=2))
        ]
        
        for filename, content in files:
            filepath = os.path.join(compress_dir, filename)
            with open(filepath, "w") as f:
                f.write(content)
            
            # Set old modification time (35 days ago)
            old_time = time.time() - (35 * 24 * 3600)
            os.utime(filepath, (old_time, old_time))
    
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.temp_test_dir and os.path.exists(self.temp_test_dir):
            shutil.rmtree(self.temp_test_dir)
            logger.info("Test environment cleaned up")
    
    # Core Testing Methods
    
    def test_endpoint(self, method: str, endpoint: str, params: Dict = None, 
                     data: Dict = None, expected_status: int = 200) -> Dict[str, Any]:
        """Test a single endpoint with comprehensive validation"""
        start_time = time.time()
        test_result = {
            "endpoint": endpoint,
            "method": method,
            "params": params,
            "data": data,
            "expected_status": expected_status,
            "start_time": start_time,
            "success": False,
            "response_data": None,
            "error": None,
            "duration": 0,
            "status_code": None
        }
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                response = self.session.get(url, params=params)
            elif method.upper() == "POST":
                response = self.session.post(url, params=params, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            test_result["status_code"] = response.status_code
            test_result["duration"] = time.time() - start_time
            
            if response.status_code == expected_status:
                try:
                    response_data = response.json()
                    test_result["response_data"] = response_data
                    test_result["success"] = True
                except json.JSONDecodeError:
                    test_result["error"] = "Invalid JSON response"
            else:
                test_result["error"] = f"Unexpected status code: {response.status_code}"
                test_result["response_data"] = response.text
        
        except Exception as e:
            test_result["error"] = str(e)
            test_result["duration"] = time.time() - start_time
        
        self.test_results.append(test_result)
        return test_result
    
    def run_health_tests(self) -> List[Dict[str, Any]]:
        """Test health and status endpoints"""
        logger.info("Running health tests...")
        
        tests = [
            ("GET", "/health", None, None, 200),
            ("GET", "/status", None, None, 200),
        ]
        
        results = []
        for method, endpoint, params, data, expected_status in tests:
            result = self.test_endpoint(method, endpoint, params, data, expected_status)
            results.append(result)
            
            # Additional validation for health endpoint
            if endpoint == "/health" and result["success"]:
                response_data = result["response_data"]
                assert response_data.get("status") == "healthy"
                assert response_data.get("agent") == "hardware-resource-optimizer"
                assert "system_status" in response_data
                assert "docker_available" in response_data
        
        return results
    
    def run_optimization_tests(self) -> List[Dict[str, Any]]:
        """Test all optimization endpoints"""
        logger.info("Running optimization tests...")
        
        optimization_endpoints = [
            "/optimize/memory",
            "/optimize/cpu", 
            "/optimize/disk",
            "/optimize/docker",
            "/optimize/all"
        ]
        
        results = []
        for endpoint in optimization_endpoints:
            # Test normal execution
            result = self.test_endpoint("POST", endpoint)
            results.append(result)
            
            # Validate response structure
            if result["success"]:
                response_data = result["response_data"]
                assert response_data.get("status") in ["success", "error"]
                assert "optimization_type" in response_data
                assert "actions_taken" in response_data
                assert "timestamp" in response_data
        
        return results
    
    def run_storage_analysis_tests(self) -> List[Dict[str, Any]]:
        """Test storage analysis endpoints"""
        logger.info("Running storage analysis tests...")
        
        results = []
        
        # Test storage analysis with different paths
        test_paths = ["/", "/tmp", self.temp_test_dir]
        for path in test_paths:
            result = self.test_endpoint("GET", "/analyze/storage", {"path": path})
            results.append(result)
            
            if result["success"]:
                response_data = result["response_data"]
                if response_data.get("status") == "success":
                    assert "total_files" in response_data
                    assert "total_size_bytes" in response_data
                    assert "extension_breakdown" in response_data
        
        # Test duplicate analysis
        result = self.test_endpoint("GET", "/analyze/storage/duplicates", 
                                   {"path": os.path.join(self.temp_test_dir, "duplicate_files")})
        results.append(result)
        
        if result["success"] and result["response_data"].get("status") == "success":
            response_data = result["response_data"]
            assert "duplicate_groups" in response_data
            assert "total_duplicates" in response_data
            assert "space_wasted_bytes" in response_data
        
        # Test large files analysis with different thresholds
        thresholds = [10, 100, 200]  # MB
        for threshold in thresholds:
            result = self.test_endpoint("GET", "/analyze/storage/large-files", 
                                       {"path": os.path.join(self.temp_test_dir, "large_files"), 
                                        "min_size_mb": threshold})
            results.append(result)
            
            if result["success"] and result["response_data"].get("status") == "success":
                response_data = result["response_data"]
                assert "large_files_count" in response_data
                assert "total_size_mb" in response_data
        
        # Test storage report
        result = self.test_endpoint("GET", "/analyze/storage/report")
        results.append(result)
        
        if result["success"] and result["response_data"].get("status") == "success":
            response_data = result["response_data"]
            assert "disk_usage" in response_data
            assert "path_analysis" in response_data
        
        return results
    
    def run_storage_optimization_tests(self) -> List[Dict[str, Any]]:
        """Test storage optimization endpoints"""
        logger.info("Running storage optimization tests...")
        
        results = []
        
        # Test comprehensive storage optimization (dry run first)
        result = self.test_endpoint("POST", "/optimize/storage", {"dry_run": True})
        results.append(result)
        
        if result["success"]:
            response_data = result["response_data"]
            assert response_data.get("dry_run") == True
            assert "actions_taken" in response_data
        
        # Test duplicate optimization (dry run)
        result = self.test_endpoint("POST", "/optimize/storage/duplicates", 
                                   {"path": os.path.join(self.temp_test_dir, "duplicate_files"), 
                                    "dry_run": True})
        results.append(result)
        
        # Test cache optimization
        result = self.test_endpoint("POST", "/optimize/storage/cache")
        results.append(result)
        
        # Test compression optimization
        result = self.test_endpoint("POST", "/optimize/storage/compress", 
                                   {"path": os.path.join(self.temp_test_dir, "compressible_files"), 
                                    "days_old": 30})
        results.append(result)
        
        # Test log optimization
        result = self.test_endpoint("POST", "/optimize/storage/logs")
        results.append(result)
        
        return results
    
    def run_parameter_validation_tests(self) -> List[Dict[str, Any]]:
        """Test parameter validation and edge cases"""
        logger.info("Running parameter validation tests...")
        
        results = []
        
        # Test invalid paths
        invalid_paths = ["/nonexistent", "/root/restricted", ""]
        for path in invalid_paths:
            result = self.test_endpoint("GET", "/analyze/storage", {"path": path})
            results.append(result)
        
        # Test invalid parameters
        result = self.test_endpoint("GET", "/analyze/storage/large-files", 
                                   {"path": "/", "min_size_mb": -1})
        results.append(result)
        
        result = self.test_endpoint("GET", "/analyze/storage/large-files", 
                                   {"path": "/", "min_size_mb": "invalid"})
        results.append(result)
        
        # Test missing required parameters
        result = self.test_endpoint("POST", "/optimize/storage/compress", {})
        results.append(result)
        
        return results
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        logger.info("Running performance tests...")
        
        performance_results = {}
        
        # Test response times for all endpoints
        endpoints = [
            ("GET", "/health"),
            ("GET", "/status"),
            ("POST", "/optimize/memory"),
            ("GET", "/analyze/storage", {"path": "/tmp"}),
            ("GET", "/analyze/storage/report")
        ]
        
        for method, endpoint, *params in endpoints:
            times = []
            for _ in range(5):  # Run 5 times for average
                params_dict = params[0] if params else None
                start_time = time.time()
                result = self.test_endpoint(method, endpoint, params_dict)
                duration = time.time() - start_time
                if result["success"]:
                    times.append(duration)
            
            if times:
                performance_results[f"{method} {endpoint}"] = {
                    "avg_response_time": statistics.mean(times),
                    "min_response_time": min(times),
                    "max_response_time": max(times),
                    "std_dev": statistics.stdev(times) if len(times) > 1 else 0
                }
        
        self.performance_metrics.update(performance_results)
        return performance_results
    
    def run_concurrent_load_tests(self, num_threads: int = 10, requests_per_thread: int = 5) -> Dict[str, Any]:
        """Test concurrent load handling"""
        logger.info(f"Running concurrent load tests with {num_threads} threads...")
        
        def make_requests():
            results = []
            for _ in range(requests_per_thread):
                result = self.test_endpoint("GET", "/health")
                results.append(result["success"])
            return results
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(make_requests) for _ in range(num_threads)]
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())
        
        total_time = time.time() - start_time
        success_rate = sum(all_results) / len(all_results) * 100
        
        load_test_results = {
            "total_requests": len(all_results),
            "successful_requests": sum(all_results),
            "success_rate_percent": success_rate,
            "total_duration": total_time,
            "requests_per_second": len(all_results) / total_time
        }
        
        return load_test_results
    
    def run_error_recovery_tests(self) -> List[Dict[str, Any]]:
        """Test error handling and recovery"""
        logger.info("Running error recovery tests...")
        
        results = []
        
        # Test with invalid JSON
        try:
            response = requests.post(f"{self.base_url}/optimize/memory", 
                                   data="invalid json", 
                                   headers={'Content-Type': 'application/json'})
            results.append({
                "test": "invalid_json",
                "status_code": response.status_code,
                "success": response.status_code in [400, 422]  # Expected error codes
            })
        except Exception as e:
            results.append({
                "test": "invalid_json",
                "error": str(e),
                "success": False
            })
        
        # Test with very large path parameter
        large_path = "/"+"a" * 1000
        result = self.test_endpoint("GET", "/analyze/storage", {"path": large_path})
        results.append({
            "test": "large_path_parameter",
            "success": not result["success"]  # Should fail gracefully
        })
        
        return results
    
    def run_safety_mechanism_tests(self) -> List[Dict[str, Any]]:
        """Test safety mechanisms and protection"""
        logger.info("Running safety mechanism tests...")
        
        results = []
        
        # Test protected path access
        protected_paths = ["/etc", "/boot", "/usr", "/bin"]
        for path in protected_paths:
            result = self.test_endpoint("GET", "/analyze/storage", {"path": path})
            results.append({
                "test": f"protected_path_{path}",
                "path": path,
                "result": result,
                "safe": result["response_data"].get("status") == "error" if result["success"] else True
            })
        
        # Test dry run safety
        result = self.test_endpoint("POST", "/optimize/storage", {"dry_run": True})
        results.append({
            "test": "dry_run_safety",
            "success": result["success"],
            "dry_run_respected": result["response_data"].get("dry_run") == True if result["success"] else False
        })
        
        return results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r["success"])
        
        # Calculate endpoint coverage
        unique_endpoints = set(r["endpoint"] for r in self.test_results)
        
        # Calculate average response times
        successful_results = [r for r in self.test_results if r["success"]]
        avg_response_time = statistics.mean([r["duration"] for r in successful_results]) if successful_results else 0
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                "unique_endpoints_tested": len(unique_endpoints),
                "average_response_time": avg_response_time
            },
            "endpoint_coverage": list(unique_endpoints),
            "performance_metrics": self.performance_metrics,
            "detailed_results": self.test_results,
            "test_timestamp": datetime.now().isoformat(),
            "agent_base_url": self.base_url
        }
        
        return report
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete comprehensive test suite"""
        logger.info("Starting comprehensive E2E test suite...")
        start_time = time.time()
        
        try:
            # 1. Health and Status Tests
            health_results = self.run_health_tests()
            
            # 2. Core Optimization Tests
            optimization_results = self.run_optimization_tests()
            
            # 3. Storage Analysis Tests
            storage_analysis_results = self.run_storage_analysis_tests()
            
            # 4. Storage Optimization Tests
            storage_optimization_results = self.run_storage_optimization_tests()
            
            # 5. Parameter Validation Tests
            validation_results = self.run_parameter_validation_tests()
            
            # 6. Performance Tests
            performance_results = self.run_performance_tests()
            
            # 7. Concurrent Load Tests
            load_test_results = self.run_concurrent_load_tests()
            
            # 8. Error Recovery Tests
            error_recovery_results = self.run_error_recovery_tests()
            
            # 9. Safety Mechanism Tests
            safety_results = self.run_safety_mechanism_tests()
            
            # Generate comprehensive report
            report = self.generate_comprehensive_report()
            
            # Add additional test categories to report
            report.update({
                "health_tests": health_results,
                "optimization_tests": optimization_results,
                "storage_analysis_tests": storage_analysis_results,
                "storage_optimization_tests": storage_optimization_results,
                "validation_tests": validation_results,
                "performance_tests": performance_results,
                "load_tests": load_test_results,
                "error_recovery_tests": error_recovery_results,
                "safety_tests": safety_results,
                "total_test_duration": time.time() - start_time
            })
            
            logger.info(f"Comprehensive test suite completed in {report['total_test_duration']:.2f} seconds")
            return report
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "partial_results": self.test_results
            }
        finally:
            self.cleanup_test_environment()

if __name__ == "__main__":
    # Run the comprehensive test suite
    framework = E2ETestFramework()
    report = framework.run_comprehensive_test_suite()
    
    # Save report to file
    report_file = f"comprehensive_e2e_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nComprehensive E2E Test Report:")
    print(f"==============================")
    print(f"Total Tests: {report['test_summary']['total_tests']}")
    print(f"Successful: {report['test_summary']['successful_tests']}")
    print(f"Failed: {report['test_summary']['failed_tests']}")
    print(f"Success Rate: {report['test_summary']['success_rate']:.2f}%")
    print(f"Endpoints Tested: {report['test_summary']['unique_endpoints_tested']}")
    print(f"Average Response Time: {report['test_summary']['average_response_time']:.3f}s")
    print(f"Test Duration: {report['total_test_duration']:.2f}s")
    print(f"\nDetailed report saved to: {report_file}")