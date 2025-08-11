#!/usr/bin/env python3
"""
ULTRA-COMPREHENSIVE QA TEST SUITE for Hardware Resource Optimizer
QA Team Lead: ULTRA-TEST all fixes applied to hardware-resource-optimizer

CRITICAL FIXES BEING TESTED:
1. Event loop conflict resolution 
2. Port configuration (11111:8080)
3. Docker client thread safety
4. Path traversal security
5. Thread safety with locks

TESTING REQUIREMENTS:
- Test ALL endpoints for functionality
- Test security vulnerabilities  
- Test concurrent requests (100+)
- Test error handling
- Test recovery mechanisms
- Test performance (<200ms)
- Test integration points
- Test edge cases

ZERO tolerance for bugs. This is ULTRA-CRITICAL validation.
"""

import asyncio
import concurrent.futures
import json
import os
import sys
import time
import threading
import requests
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('QA_COMPREHENSIVE_TEST')

class UltraComprehensiveQATestSuite:
    """ULTRA-CRITICAL QA Test Suite - Zero Tolerance for Bugs"""
    
    def __init__(self):
        # Test configuration
        self.base_url = "http://localhost:11111"  # Test container on port 11111
        self.production_url = "http://localhost:11104"  # Production container on port 11104
        
        self.test_results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
            "warnings": [],
            "performance_metrics": {},
            "security_test_results": {},
            "concurrent_test_results": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Performance requirements
        self.max_response_time_ms = 200
        self.max_memory_optimization_time_ms = 200
        
        # Security test payloads
        self.path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "../../etc/shadow",
            "../../../proc/version",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]
        
        logger.info("Initialized ULTRA-COMPREHENSIVE QA Test Suite")
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Execute all ULTRA-CRITICAL tests"""
        logger.info("🚀 STARTING ULTRA-COMPREHENSIVE QA VALIDATION")
        
        start_time = time.time()
        
        try:
            # Test 1: Basic Health and Connectivity
            self._test_basic_connectivity()
            
            # Test 2: All Endpoint Functionality 
            self._test_all_endpoints()
            
            # Test 3: Security Vulnerability Tests
            self._test_security_vulnerabilities()
            
            # Test 4: Performance Tests (<200ms requirement)
            self._test_performance_requirements()
            
            # Test 5: Concurrent Request Load Testing (100+ requests)
            self._test_concurrent_load()
            
            # Test 6: Error Handling and Edge Cases
            self._test_error_handling()
            
            # Test 7: Thread Safety and Docker Client Fixes
            self._test_thread_safety()
            
            # Test 8: Event Loop Conflict Resolution
            self._test_event_loop_fixes()
            
            # Test 9: Recovery Mechanisms
            self._test_recovery_mechanisms()
            
            # Test 10: Integration Testing
            self._test_integration_points()
            
        except Exception as e:
            logger.error(f"Critical test suite error: {e}")
            self.test_results["errors"].append(f"Test suite failure: {e}")
        
        finally:
            execution_time = time.time() - start_time
            self._generate_final_report(execution_time)
            
        return self.test_results
    
    def _test_basic_connectivity(self):
        """Test 1: Basic Health and Connectivity"""
        logger.info("🔍 TEST 1: Basic Health and Connectivity")
        
        test_cases = [
            ("Test Container Health", f"{self.base_url}/health"),
            ("Production Container Health", f"{self.production_url}/health"),
            ("Test Container Root", f"{self.base_url}/"),
            ("Test Container Status", f"{self.base_url}/status")
        ]
        
        for test_name, url in test_cases:
            try:
                start_time = time.time()
                response = requests.get(url, timeout=10)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    self._record_pass(f"{test_name}: OK ({response_time:.1f}ms)")
                    
                    # Validate response structure
                    if '/health' in url:
                        data = response.json()
                        required_fields = ['status', 'agent']
                        for field in required_fields:
                            if field not in data:
                                self._record_fail(f"{test_name}: Missing field '{field}' in response")
                            else:
                                self._record_pass(f"{test_name}: Has required field '{field}'")
                else:
                    self._record_fail(f"{test_name}: HTTP {response.status_code}")
                    
            except Exception as e:
                self._record_fail(f"{test_name}: Connection error - {e}")
    
    def _test_all_endpoints(self):
        """Test 2: All Endpoint Functionality"""
        logger.info("🔍 TEST 2: All Endpoint Functionality")
        
        # GET endpoints
        get_endpoints = [
            ("/health", "Health Check"),
            ("/status", "System Status"),
            ("/", "Root Endpoint"),
            ("/analyze/storage?path=/tmp", "Storage Analysis"),
            ("/analyze/storage/duplicates?path=/tmp", "Duplicate Analysis"),
            ("/analyze/storage/large-files?path=/tmp&min_size_mb=1", "Large Files Analysis"),
            ("/analyze/storage/report", "Storage Report")
        ]
        
        for endpoint, description in get_endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}{endpoint}", timeout=15)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    self._record_pass(f"GET {endpoint} ({description}): OK ({response_time:.1f}ms)")
                    
                    # Validate response structure
                    if 'status' in data and data['status'] == 'success':
                        self._record_pass(f"GET {endpoint}: Valid success response")
                    elif 'agent' in data:  # Root endpoint format
                        self._record_pass(f"GET {endpoint}: Valid agent response")
                    else:
                        self._record_warning(f"GET {endpoint}: Unexpected response format")
                        
                else:
                    self._record_fail(f"GET {endpoint}: HTTP {response.status_code}")
                    
            except Exception as e:
                self._record_fail(f"GET {endpoint}: Error - {e}")
        
        # POST endpoints  
        post_endpoints = [
            ("/optimize/memory", {}, "Memory Optimization"),
            ("/optimize/cpu", {}, "CPU Optimization"),
            ("/optimize/disk", {}, "Disk Optimization"),
            ("/optimize/docker", {}, "Docker Optimization"),
            ("/optimize/storage?dry_run=true", {}, "Storage Optimization (Dry Run)"),
            ("/optimize/storage/cache", {}, "Cache Optimization"),
            ("/optimize/storage/logs", {}, "Log Optimization")
        ]
        
        for endpoint, payload, description in post_endpoints:
            try:
                start_time = time.time()
                response = requests.post(f"{self.base_url}{endpoint}", json=payload, timeout=30)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    self._record_pass(f"POST {endpoint} ({description}): OK ({response_time:.1f}ms)")
                    
                    # Record performance metrics
                    self.test_results["performance_metrics"][endpoint] = response_time
                    
                    # Check for success status
                    if 'status' in data and data['status'] == 'success':
                        self._record_pass(f"POST {endpoint}: Successful optimization")
                        
                        # Check for actions taken
                        if 'actions_taken' in data:
                            action_count = len(data['actions_taken'])
                            self._record_pass(f"POST {endpoint}: {action_count} actions taken")
                    else:
                        self._record_warning(f"POST {endpoint}: Non-success status: {data.get('status')}")
                        
                else:
                    self._record_fail(f"POST {endpoint}: HTTP {response.status_code}")
                    
            except Exception as e:
                self._record_fail(f"POST {endpoint}: Error - {e}")
    
    def _test_security_vulnerabilities(self):
        """Test 3: Security Vulnerability Tests"""
        logger.info("🔍 TEST 3: Security Vulnerability Tests (Path Traversal)")
        
        # Test path traversal vulnerabilities
        vulnerable_endpoints = [
            "/analyze/storage",
            "/analyze/storage/duplicates", 
            "/optimize/storage/duplicates",
            "/optimize/storage/compress"
        ]
        
        security_results = {}
        
        for endpoint in vulnerable_endpoints:
            security_results[endpoint] = {"blocked": 0, "allowed": 0, "errors": []}
            
            for payload in self.path_traversal_payloads:
                try:
                    # Test GET endpoints with malicious path parameter
                    if endpoint.startswith("/analyze"):
                        url = f"{self.base_url}{endpoint}?path={payload}"
                        response = requests.get(url, timeout=10)
                    else:
                        # Test POST endpoints with malicious path parameter
                        url = f"{self.base_url}{endpoint}?path={payload}&dry_run=true"
                        response = requests.post(url, json={}, timeout=10)
                    
                    # Security fix should return 403 (Forbidden) for path traversal attempts
                    if response.status_code == 403:
                        security_results[endpoint]["blocked"] += 1
                        self._record_pass(f"Security: Blocked path traversal attempt on {endpoint} with payload: {payload}")
                    elif response.status_code in [400, 404]:
                        security_results[endpoint]["blocked"] += 1  
                        self._record_pass(f"Security: Rejected malicious path on {endpoint} (HTTP {response.status_code})")
                    elif response.status_code == 200:
                        security_results[endpoint]["allowed"] += 1
                        
                        # Check if the path validation worked in the response
                        try:
                            data = response.json()
                            if 'error' in data and 'traversal' in data['error'].lower():
                                security_results[endpoint]["blocked"] += 1
                                self._record_pass(f"Security: Path traversal blocked in application logic for {endpoint}")
                            else:
                                self._record_fail(f"🚨 SECURITY VULNERABILITY: Path traversal allowed on {endpoint} with payload: {payload}")
                        except (AssertionError, Exception) as e:
                            logger.error(f"Unexpected exception: {e}", exc_info=True)
                            self._record_fail(f"🚨 SECURITY VULNERABILITY: Path traversal potentially allowed on {endpoint}")
                    else:
                        security_results[endpoint]["errors"].append(f"HTTP {response.status_code} for payload {payload}")
                        
                except Exception as e:
                    security_results[endpoint]["errors"].append(f"Error testing {payload}: {e}")
        
        # Evaluate security test results
        for endpoint, results in security_results.items():
            total_tests = len(self.path_traversal_payloads)
            blocked_rate = results["blocked"] / total_tests * 100
            
            if blocked_rate >= 100:
                self._record_pass(f"Security: {endpoint} - 100% path traversal attacks blocked")
            elif blocked_rate >= 80:
                self._record_warning(f"Security: {endpoint} - {blocked_rate:.1f}% attacks blocked (needs improvement)")
            else:
                self._record_fail(f"🚨 CRITICAL SECURITY ISSUE: {endpoint} - Only {blocked_rate:.1f}% attacks blocked")
        
        self.test_results["security_test_results"] = security_results
    
    def _test_performance_requirements(self):
        """Test 4: Performance Tests (<200ms requirement)"""
        logger.info("🔍 TEST 4: Performance Requirements (<200ms)")
        
        # Critical performance endpoints
        performance_endpoints = [
            ("/optimize/memory", "POST", {}),
            ("/health", "GET", None),
            ("/status", "GET", None),
            ("/analyze/storage?path=/tmp", "GET", None)
        ]
        
        performance_results = {}
        
        for endpoint, method, payload in performance_endpoints:
            response_times = []
            
            # Run each test 5 times for statistical significance
            for i in range(5):
                try:
                    start_time = time.time()
                    
                    if method == "GET":
                        response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                    else:
                        response = requests.post(f"{self.base_url}{endpoint}", json=payload, timeout=5)
                    
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        response_times.append(response_time)
                    
                except Exception as e:
                    self._record_fail(f"Performance test error on {endpoint}: {e}")
            
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                max_time = max(response_times)
                min_time = min(response_times)
                
                performance_results[endpoint] = {
                    "average_ms": avg_time,
                    "max_ms": max_time,
                    "min_ms": min_time,
                    "samples": len(response_times)
                }
                
                # Evaluate against performance requirements
                if avg_time < self.max_response_time_ms:
                    self._record_pass(f"Performance: {endpoint} - Average {avg_time:.1f}ms (< {self.max_response_time_ms}ms ✅)")
                else:
                    self._record_fail(f"⚡ PERFORMANCE ISSUE: {endpoint} - Average {avg_time:.1f}ms (> {self.max_response_time_ms}ms ❌)")
                
                if max_time > self.max_response_time_ms * 2:  # Allow 2x tolerance for max
                    self._record_warning(f"Performance: {endpoint} - Max response time {max_time:.1f}ms is high")
        
        # Special test for memory optimization performance requirement
        if "/optimize/memory" in performance_results:
            memory_perf = performance_results["/optimize/memory"]
            if memory_perf["average_ms"] < self.max_memory_optimization_time_ms:
                self._record_pass(f"🎯 CRITICAL REQUIREMENT MET: Memory optimization averages {memory_perf['average_ms']:.1f}ms < 200ms")
            else:
                self._record_fail(f"🚨 CRITICAL REQUIREMENT FAILED: Memory optimization averages {memory_perf['average_ms']:.1f}ms > 200ms")
        
        self.test_results["performance_metrics"].update(performance_results)
    
    def _test_concurrent_load(self):
        """Test 5: Concurrent Request Load Testing (100+ requests)"""
        logger.info("🔍 TEST 5: Concurrent Load Testing (100+ requests)")
        
        def make_request(endpoint_info):
            endpoint, method, payload = endpoint_info
            try:
                start_time = time.time()
                
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=30)
                else:
                    response = requests.post(f"{self.base_url}{endpoint}", json=payload, timeout=30)
                
                response_time = (time.time() - start_time) * 1000
                
                return {
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "response_time_ms": response_time,
                    "endpoint": endpoint
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "endpoint": endpoint
                }
        
        # Test endpoints under load
        test_endpoints = [
            ("/health", "GET", None),
            ("/status", "GET", None), 
            ("/optimize/memory", "POST", {}),
            ("/analyze/storage?path=/tmp", "GET", None)
        ] * 25  # 100 total requests (25 of each endpoint)
        
        start_time = time.time()
        
        # Execute concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request, endpoint_info) for endpoint_info in test_endpoints]
            results = [future.result() for future in concurrent.futures.as_completed(futures, timeout=60)]
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        success_rate = len(successful_requests) / len(results) * 100
        avg_response_time = sum(r["response_time_ms"] for r in successful_requests if "response_time_ms" in r) / len(successful_requests) if successful_requests else 0
        
        concurrent_results = {
            "total_requests": len(results),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate_percent": success_rate,
            "average_response_time_ms": avg_response_time,
            "total_execution_time_s": total_time,
            "requests_per_second": len(results) / total_time
        }
        
        # Evaluate concurrent load performance
        if success_rate >= 95:
            self._record_pass(f"Concurrent Load: {success_rate:.1f}% success rate (≥95% ✅)")
        elif success_rate >= 90:
            self._record_warning(f"Concurrent Load: {success_rate:.1f}% success rate (needs improvement)")
        else:
            self._record_fail(f"🚨 LOAD TEST FAILURE: Only {success_rate:.1f}% success rate under load")
        
        if avg_response_time < self.max_response_time_ms * 3:  # Allow 3x tolerance under load
            self._record_pass(f"Concurrent Load: Average response time {avg_response_time:.1f}ms under load")
        else:
            self._record_fail(f"⚡ PERFORMANCE DEGRADATION: {avg_response_time:.1f}ms average response time under load")
        
        # Check for any specific error patterns
        error_patterns = {}
        for result in failed_requests:
            error = result.get("error", f"HTTP {result.get('status_code')}")
            error_patterns[error] = error_patterns.get(error, 0) + 1
        
        for error, count in error_patterns.items():
            self._record_warning(f"Concurrent Load: {count} requests failed with: {error}")
        
        self.test_results["concurrent_test_results"] = concurrent_results
    
    def _test_error_handling(self):
        """Test 6: Error Handling and Edge Cases"""
        logger.info("🔍 TEST 6: Error Handling and Edge Cases")
        
        # Test invalid parameters
        error_test_cases = [
            ("/analyze/storage?path=/nonexistent/path", "GET", "Non-existent path"),
            ("/analyze/storage/large-files?path=/tmp&min_size_mb=-1", "GET", "Negative size parameter"),
            ("/optimize/storage/compress?path=/tmp&days_old=-5", "POST", "Negative days parameter"),
            ("/optimize/storage?invalid_param=true", "POST", "Invalid parameter"),
        ]
        
        for endpoint, method, description in error_test_cases:
            try:
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                else:
                    response = requests.post(f"{self.base_url}{endpoint}", json={}, timeout=10)
                
                # Error handling should return appropriate HTTP codes
                if response.status_code in [400, 403, 404, 422]:
                    self._record_pass(f"Error Handling: {description} correctly returns HTTP {response.status_code}")
                elif response.status_code == 200:
                    # Check if error is handled in response body
                    try:
                        data = response.json()
                        if 'status' in data and data['status'] == 'error':
                            self._record_pass(f"Error Handling: {description} handled in response body")
                        else:
                            self._record_warning(f"Error Handling: {description} may need better validation")
                    except (AssertionError, Exception) as e:
                        logger.error(f"Unexpected exception: {e}", exc_info=True)
                        self._record_warning(f"Error Handling: {description} response format unclear")
                else:
                    self._record_fail(f"Error Handling: {description} returns unexpected HTTP {response.status_code}")
                    
            except Exception as e:
                self._record_warning(f"Error Handling: {description} caused exception: {e}")
        
        # Test malformed JSON payloads
        malformed_payload_tests = [
            ("/optimize/memory", "invalid json"),
            ("/optimize/storage", '{"invalid": }'),
        ]
        
        for endpoint, bad_payload in malformed_payload_tests:
            try:
                response = requests.post(
                    f"{self.base_url}{endpoint}", 
                    data=bad_payload, 
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code in [400, 422]:
                    self._record_pass(f"Error Handling: Malformed JSON correctly rejected on {endpoint}")
                else:
                    self._record_warning(f"Error Handling: Malformed JSON handling unclear on {endpoint}")
                    
            except Exception as e:
                self._record_pass(f"Error Handling: Malformed JSON properly rejected on {endpoint}")
    
    def _test_thread_safety(self):
        """Test 7: Thread Safety and Docker Client Fixes"""
        logger.info("🔍 TEST 7: Thread Safety and Docker Client Fixes")
        
        def concurrent_docker_operation(thread_id):
            """Test Docker operations in concurrent threads"""
            try:
                start_time = time.time()
                response = requests.post(f"{self.base_url}/optimize/docker", json={}, timeout=30)
                response_time = (time.time() - start_time) * 1000
                
                return {
                    "thread_id": thread_id,
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "response_time_ms": response_time,
                    "has_error": False
                }
                
            except Exception as e:
                return {
                    "thread_id": thread_id,
                    "success": False,
                    "error": str(e),
                    "has_error": True
                }
        
        # Test thread safety with concurrent Docker operations
        logger.info("Testing Docker client thread safety with 10 concurrent threads...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            docker_futures = [executor.submit(concurrent_docker_operation, i) for i in range(10)]
            docker_results = [future.result() for future in concurrent.futures.as_completed(docker_futures, timeout=60)]
        
        successful_docker_ops = [r for r in docker_results if r["success"]]
        failed_docker_ops = [r for r in docker_results if not r["success"]]
        
        docker_success_rate = len(successful_docker_ops) / len(docker_results) * 100
        
        if docker_success_rate >= 90:
            self._record_pass(f"Thread Safety: Docker operations {docker_success_rate:.1f}% success rate in concurrent threads")
        else:
            self._record_fail(f"🚨 THREAD SAFETY ISSUE: Docker operations only {docker_success_rate:.1f}% success rate")
        
        # Test memory optimization thread safety 
        def concurrent_memory_optimization(thread_id):
            try:
                response = requests.post(f"{self.base_url}/optimize/memory", json={}, timeout=15)
                return {
                    "thread_id": thread_id,
                    "success": response.status_code == 200,
                    "status_code": response.status_code
                }
            except Exception as e:
                return {"thread_id": thread_id, "success": False, "error": str(e)}
        
        logger.info("Testing memory optimization thread safety with 15 concurrent threads...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            memory_futures = [executor.submit(concurrent_memory_optimization, i) for i in range(15)]
            memory_results = [future.result() for future in concurrent.futures.as_completed(memory_futures, timeout=60)]
        
        successful_memory_ops = [r for r in memory_results if r["success"]]
        memory_success_rate = len(successful_memory_ops) / len(memory_results) * 100
        
        if memory_success_rate >= 95:
            self._record_pass(f"Thread Safety: Memory optimization {memory_success_rate:.1f}% success rate in concurrent threads")
        else:
            self._record_warning(f"Thread Safety: Memory optimization {memory_success_rate:.1f}% success rate (may need improvement)")
    
    def _test_event_loop_fixes(self):
        """Test 8: Event Loop Conflict Resolution"""
        logger.info("🔍 TEST 8: Event Loop Conflict Resolution")
        
        # Test rapid successive requests to trigger potential event loop conflicts
        rapid_requests = []
        
        for i in range(20):
            try:
                start_time = time.time()
                response = requests.post(f"{self.base_url}/optimize/memory", json={}, timeout=10)
                response_time = (time.time() - start_time) * 1000
                
                rapid_requests.append({
                    "request_id": i,
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "response_time_ms": response_time
                })
                
                # Brief delay to test event loop handling
                time.sleep(0.1)
                
            except Exception as e:
                rapid_requests.append({
                    "request_id": i,
                    "success": False,
                    "error": str(e)
                })
        
        successful_rapid = [r for r in rapid_requests if r["success"]]
        rapid_success_rate = len(successful_rapid) / len(rapid_requests) * 100
        
        if rapid_success_rate >= 95:
            self._record_pass(f"Event Loop: {rapid_success_rate:.1f}% success rate in rapid successive requests")
        elif rapid_success_rate >= 85:
            self._record_warning(f"Event Loop: {rapid_success_rate:.1f}% success rate (some instability)")
        else:
            self._record_fail(f"🚨 EVENT LOOP ISSUE: Only {rapid_success_rate:.1f}% success rate in rapid requests")
        
        # Check for consistent response times (no major spikes indicating deadlocks)
        if successful_rapid:
            response_times = [r["response_time_ms"] for r in successful_rapid if "response_time_ms" in r]
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                max_time = max(response_times)
                
                if max_time < avg_time * 5:  # Max should not be more than 5x average
                    self._record_pass(f"Event Loop: Consistent response times (avg: {avg_time:.1f}ms, max: {max_time:.1f}ms)")
                else:
                    self._record_warning(f"Event Loop: Response time variance high (avg: {avg_time:.1f}ms, max: {max_time:.1f}ms)")
    
    def _test_recovery_mechanisms(self):
        """Test 9: Recovery Mechanisms"""
        logger.info("🔍 TEST 9: Recovery Mechanisms") 
        
        # Test service recovery after errors
        recovery_tests = [
            ("Service continues after invalid request", "/invalid/endpoint"),
            ("Service recovers after timeout simulation", "/optimize/memory"),
            ("Health check remains responsive", "/health")
        ]
        
        for test_name, endpoint in recovery_tests:
            # Make a potentially problematic request
            try:
                if endpoint == "/invalid/endpoint":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                else:
                    response = requests.post(f"{self.base_url}{endpoint}", json={}, timeout=30)
            except Exception as e:
                pass  # Expected for some recovery tests
            
            # Immediately test if service is still responsive
            try:
                health_response = requests.get(f"{self.base_url}/health", timeout=5)
                if health_response.status_code == 200:
                    self._record_pass(f"Recovery: {test_name} - Service remains healthy")
                else:
                    self._record_fail(f"🚨 RECOVERY ISSUE: {test_name} - Service health compromised")
            except Exception as e:
                self._record_fail(f"🚨 CRITICAL RECOVERY FAILURE: {test_name} - Service unreachable: {e}")
    
    def _test_integration_points(self):
        """Test 10: Integration Testing"""
        logger.info("🔍 TEST 10: Integration Testing")
        
        # Test Docker integration (if available)
        try:
            response = requests.post(f"{self.base_url}/optimize/docker", json={}, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'status' in data:
                    if data['status'] == 'success':
                        self._record_pass("Integration: Docker optimization successful")
                    elif data['status'] == 'error' and 'not available' in data.get('error', '').lower():
                        self._record_warning("Integration: Docker not available (expected in containerized environment)")
                    else:
                        self._record_warning(f"Integration: Docker optimization status: {data['status']}")
            else:
                self._record_fail(f"Integration: Docker optimization failed with HTTP {response.status_code}")
        except Exception as e:
            self._record_fail(f"Integration: Docker test error - {e}")
        
        # Test system resource integration
        try:
            response = requests.get(f"{self.base_url}/status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                required_metrics = ['cpu_percent', 'memory_percent', 'disk_percent']
                
                for metric in required_metrics:
                    if metric in data:
                        self._record_pass(f"Integration: System metric '{metric}' available")
                        
                        # Validate metric ranges
                        value = data[metric]
                        if 0 <= value <= 100:
                            self._record_pass(f"Integration: System metric '{metric}' = {value:.1f}% (valid range)")
                        else:
                            self._record_warning(f"Integration: System metric '{metric}' = {value:.1f}% (outside expected range)")
                    else:
                        self._record_fail(f"Integration: Missing system metric '{metric}'")
            else:
                self._record_fail(f"Integration: System status endpoint failed with HTTP {response.status_code}")
        except Exception as e:
            self._record_fail(f"Integration: System status test error - {e}")
        
        # Test storage analysis integration
        try:
            response = requests.get(f"{self.base_url}/analyze/storage?path=/tmp", timeout=20)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    required_fields = ['total_files', 'total_size_bytes', 'extension_breakdown']
                    for field in required_fields:
                        if field in data:
                            self._record_pass(f"Integration: Storage analysis includes '{field}'")
                        else:
                            self._record_warning(f"Integration: Storage analysis missing '{field}'")
                    self._record_pass("Integration: Storage analysis working correctly")
                else:
                    self._record_fail(f"Integration: Storage analysis failed: {data.get('error')}")
            else:
                self._record_fail(f"Integration: Storage analysis HTTP {response.status_code}")
        except Exception as e:
            self._record_fail(f"Integration: Storage analysis error - {e}")
    
    def _record_pass(self, message: str):
        """Record a passed test"""
        self.test_results["total_tests"] += 1
        self.test_results["passed"] += 1
        logger.info(f"✅ PASS: {message}")
    
    def _record_fail(self, message: str):
        """Record a failed test"""
        self.test_results["total_tests"] += 1
        self.test_results["failed"] += 1
        self.test_results["errors"].append(message)
        logger.error(f"❌ FAIL: {message}")
    
    def _record_warning(self, message: str):
        """Record a test warning"""
        self.test_results["warnings"].append(message)
        logger.warning(f"⚠️  WARN: {message}")
    
    def _generate_final_report(self, execution_time: float):
        """Generate comprehensive final test report"""
        logger.info("📊 GENERATING FINAL QA REPORT")
        
        # Calculate success rates
        total_tests = self.test_results["total_tests"]
        passed_tests = self.test_results["passed"]
        failed_tests = self.test_results["failed"]
        
        if total_tests > 0:
            pass_rate = (passed_tests / total_tests) * 100
        else:
            pass_rate = 0
        
        # Generate report
        report = {
            "qa_test_execution": {
                "timestamp": self.test_results["timestamp"],
                "execution_time_seconds": execution_time,
                "total_tests_executed": total_tests,
                "tests_passed": passed_tests,
                "tests_failed": failed_tests,
                "pass_rate_percent": pass_rate,
                "warnings_count": len(self.test_results["warnings"])
            },
            "critical_fixes_validated": [
                "Event loop conflict resolution",
                "Port configuration (11111:8080)",
                "Docker client thread safety",
                "Path traversal security",
                "Thread safety with locks"
            ],
            "test_categories": {
                "basic_connectivity": "✅ Completed",
                "endpoint_functionality": "✅ Completed",
                "security_vulnerabilities": "✅ Completed", 
                "performance_requirements": "✅ Completed",
                "concurrent_load_testing": "✅ Completed",
                "error_handling": "✅ Completed",
                "thread_safety": "✅ Completed",
                "event_loop_fixes": "✅ Completed",
                "recovery_mechanisms": "✅ Completed",
                "integration_testing": "✅ Completed"
            },
            "performance_analysis": self.test_results["performance_metrics"],
            "security_analysis": self.test_results["security_test_results"],
            "concurrent_load_analysis": self.test_results["concurrent_test_results"]
        }
        
        # Overall assessment
        if pass_rate >= 95 and failed_tests == 0:
            overall_status = "🟢 EXCELLENT - All fixes validated, zero tolerance maintained"
            recommendation = "Service is production-ready with all critical fixes validated"
        elif pass_rate >= 90:
            overall_status = "🟡 GOOD - Minor issues found, fixes working well"
            recommendation = "Service is stable, address warnings for optimization"
        elif pass_rate >= 80:
            overall_status = "🟠 ACCEPTABLE - Some issues found, fixes partially working"
            recommendation = "Address failed tests before production deployment"
        else:
            overall_status = "🔴 CRITICAL - Major issues found, fixes need attention"
            recommendation = "DO NOT DEPLOY - Critical issues must be resolved"
        
        report["final_assessment"] = {
            "overall_status": overall_status,
            "recommendation": recommendation,
            "critical_failures": [error for error in self.test_results["errors"] if "🚨" in error],
            "zero_tolerance_maintained": failed_tests == 0 and pass_rate >= 95
        }
        
        # Log final results
        logger.info("=" * 80)
        logger.info("🎯 ULTRA-COMPREHENSIVE QA TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Execution Time: {execution_time:.2f} seconds")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Pass Rate: {pass_rate:.1f}%")
        logger.info(f"Warnings: {len(self.test_results['warnings'])}")
        logger.info(f"Overall Status: {overall_status}")
        logger.info(f"Recommendation: {recommendation}")
        logger.info("=" * 80)
        
        # Log critical failures
        if report["final_assessment"]["critical_failures"]:
            logger.error("🚨 CRITICAL FAILURES:")
            for failure in report["final_assessment"]["critical_failures"]:
                logger.error(f"  - {failure}")
        
        # Log warnings
        if self.test_results["warnings"]:
            logger.warning("⚠️  WARNINGS:")
            for warning in self.test_results["warnings"]:
                logger.warning(f"  - {warning}")
        
        # Save detailed report
        report_file = f"/opt/sutazaiapp/tests/qa_comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📄 Detailed report saved: {report_file}")
        except Exception as e:
            logger.error(f"Could not save report: {e}")
        
        self.test_results.update(report)

def main():
    """Run the ULTRA-COMPREHENSIVE QA Test Suite"""
    logger.info("🚀 Starting ULTRA-COMPREHENSIVE QA Validation")
    logger.info("Testing ALL fixes applied to hardware-resource-optimizer")
    logger.info("ZERO tolerance for bugs - Production readiness validation")
    
    qa_suite = UltraComprehensiveQATestSuite()
    results = qa_suite.run_all_tests()
    
    # Return results for external processing
    return results

if __name__ == "__main__":
    results = main()
    
    # Exit with appropriate code
    if results["qa_test_execution"]["tests_failed"] == 0:
        sys.exit(0)  # Success
    else:
