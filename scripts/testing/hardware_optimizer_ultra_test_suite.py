#!/usr/bin/env python3
"""
ULTRA-CRITICAL AUTOMATED TESTING SPECIALIST - HARDWARE RESOURCE OPTIMIZER
==========================================================================

Ultra-comprehensive automated testing suite for hardware-resource-optimizer service.
This test suite implements ZERO-TOLERANCE testing with complete automation.

SCOPE:
- 16 endpoints with comprehensive load testing
- 6 concurrent user levels: 1, 5, 10, 25, 50, 100
- Performance SLA validation: <200ms for 95% requests, >99.5% success rate
- Memory leak detection and resource monitoring
- Security boundary testing and error injection
- Automated stress, spike, endurance, and volume testing
- Complete recovery testing and failover scenarios

Author: Ultra-Critical Automated Testing Specialist
Version: 1.0.0
Compliance: COMPREHENSIVE CODEBASE RULES 1-19
"""

import asyncio
import aiohttp
import json
import time
import statistics
import psutil
import threading
import tempfile
import shutil
import os
import sys
import hashlib
import gzip
import logging
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
import requests
import pytest
import subprocess
import signal
import contextlib
import gc
import traceback
import random
import string
import resource
import socket
from functools import wraps
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/tests/ultra_hardware_optimizer_tests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('UltraHardwareOptimizerTests')

@dataclass
class EndpointSpec:
    """Specification for an endpoint to test"""
    method: str
    path: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    requires_dry_run: bool = False
    timeout: int = 30

@dataclass
class TestResult:
    """Result of a single test request"""
    endpoint: str
    method: str
    success: bool
    response_time_ms: float
    status_code: int
    response_size: int
    memory_usage_mb: float
    cpu_percent: float
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class LoadTestMetrics:
    """Aggregated metrics for load testing"""
    endpoint: str
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    avg_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    throughput_rps: float
    total_duration_s: float
    memory_peak_mb: float
    memory_leak_detected: bool
    sla_compliance: bool
    errors_by_type: Dict[str, int] = field(default_factory=dict)

@dataclass
class SecurityTestResult:
    """Result of security boundary testing"""
    test_type: str
    endpoint: str
    success: bool
    vulnerability_detected: bool
    details: str
    severity: str = "LOW"

class SystemMonitor:
    """Continuous system performance monitoring during tests"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.metrics: List[Dict[str, float]] = []
        self.monitor_thread: Optional[threading.Thread] = None
        self.process = psutil.Process()
        
    def start(self) -> None:
        """Start system monitoring"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")
        
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return summary"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        return self._calculate_summary()
    
    def _monitor_loop(self) -> None:
        """Continuous monitoring loop"""
        while self.monitoring:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Network connections
                connections = len(psutil.net_connections())
                
                # Process-specific metrics
                process_memory = self.process.memory_info()
                process_cpu = self.process.cpu_percent()
                
                metric = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': (disk.used / disk.total) * 100,
                    'disk_free_gb': disk.free / (1024**3),
                    'connections': connections,
                    'process_memory_mb': process_memory.rss / (1024*1024),
                    'process_cpu_percent': process_cpu,
                    'load_avg_1m': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
                }
                
                self.metrics.append(metric)
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                break
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics from metrics"""
        if not self.metrics:
            return {}
        
        # Extract time series data
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_percent'] for m in self.metrics]
        process_memory_values = [m['process_memory_mb'] for m in self.metrics]
        connection_values = [m['connections'] for m in self.metrics]
        
        # Calculate statistics
        summary = {
            'duration_seconds': self.metrics[-1]['timestamp'] - self.metrics[0]['timestamp'],
            'sample_count': len(self.metrics),
            'cpu_stats': {
                'avg': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'median': statistics.median(cpu_values)
            },
            'memory_stats': {
                'avg': statistics.mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'median': statistics.median(memory_values)
            },
            'process_memory_stats': {
                'avg': statistics.mean(process_memory_values),
                'max': max(process_memory_values),
                'min': min(process_memory_values),
                'peak_mb': max(process_memory_values),
                'leak_detected': max(process_memory_values) - min(process_memory_values) > 50  # >50MB growth
            },
            'connection_stats': {
                'avg': statistics.mean(connection_values),
                'max': max(connection_values),
                'min': min(connection_values)
            }
        }
        
        return summary

class UltraHardwareOptimizerTester:
    """Ultra-comprehensive automated testing framework for hardware-resource-optimizer"""
    
    # All 16 endpoints discovered from OpenAPI spec
    ENDPOINTS = [
        EndpointSpec("GET", "/health", "Health check endpoint"),
        EndpointSpec("GET", "/status", "Get current system resource status"),
        EndpointSpec("POST", "/optimize/memory", "Optimize memory usage"),
        EndpointSpec("POST", "/optimize/cpu", "Optimize CPU scheduling"),
        EndpointSpec("POST", "/optimize/disk", "Clean up disk space"),
        EndpointSpec("POST", "/optimize/docker", "Clean up Docker resources"),
        EndpointSpec("POST", "/optimize/all", "Run all optimizations"),
        EndpointSpec("GET", "/analyze/storage", "Analyze storage usage", {"path": "/tmp"}),
        EndpointSpec("GET", "/analyze/storage/duplicates", "Find duplicate files", {"path": "/tmp"}),
        EndpointSpec("GET", "/analyze/storage/large-files", "Find large files", {"path": "/tmp", "min_size_mb": 10}),
        EndpointSpec("GET", "/analyze/storage/report", "Generate storage report"),
        EndpointSpec("POST", "/optimize/storage", "Storage optimization", {"dry_run": True}, requires_dry_run=True),
        EndpointSpec("POST", "/optimize/storage/duplicates", "Remove duplicates", {"path": "/tmp", "dry_run": True}, requires_dry_run=True),
        EndpointSpec("POST", "/optimize/storage/cache", "Clear caches"),
        EndpointSpec("POST", "/optimize/storage/compress", "Compress files", {"path": "/tmp", "days_old": 1}),
        EndpointSpec("POST", "/optimize/storage/logs", "Log rotation and cleanup")
    ]
    
    # Load testing configurations
    LOAD_LEVELS = [1, 5, 10, 25, 50, 100]
    TEST_DURATION = 60  # seconds
    SLA_RESPONSE_TIME_MS = 200  # <200ms for 95% of requests
    SLA_SUCCESS_RATE = 99.5  # >99.5% success rate
    MAX_MEMORY_USAGE_MB = 200  # Must not exceed 200MB during testing
    
    def __init__(self, base_url: str = "http://localhost:11110", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Test results storage
        self.load_test_results: List[LoadTestMetrics] = []
        self.security_test_results: List[SecurityTestResult] = []
        self.stress_test_results: List[Dict[str, Any]] = []
        self.error_injection_results: List[Dict[str, Any]] = []
        
        # Test environment
        self.temp_test_dir: Optional[str] = None
        self.system_monitor = SystemMonitor()
        
        # Setup test environment
        self._setup_test_environment()
        
    def _setup_test_environment(self) -> None:
        """Setup comprehensive test environment"""
        # Create temporary test directory
        self.temp_test_dir = tempfile.mkdtemp(prefix="ultra_hardware_optimizer_test_")
        logger.info(f"Test environment created: {self.temp_test_dir}")
        
        # Create test data
        self._create_test_data()
        
        # Verify service availability
        self._verify_service_health()
    
    def _create_test_data(self) -> None:
        """Create comprehensive test data for all test scenarios"""
        if not self.temp_test_dir:
            return
            
        # Create directory structure
        test_dirs = [
            "temp_files", "cache_files", "log_files", 
            "large_files", "duplicate_files", "compressible_files",
            "edge_case_files", "security_test_files"
        ]
        
        for dir_name in test_dirs:
            os.makedirs(os.path.join(self.temp_test_dir, dir_name), exist_ok=True)
        
        # Create test files for different scenarios
        self._create_temp_files()
        self._create_large_files()
        self._create_duplicate_files() 
        self._create_compressible_files()
        self._create_edge_case_files()
        
        logger.info("Test data creation completed")
    
    def _create_temp_files(self) -> None:
        """Create temporary files for cleanup testing"""
        temp_dir = os.path.join(self.temp_test_dir, "temp_files")
        
        # Recent files (should not be deleted)
        for i in range(5):
            with open(os.path.join(temp_dir, f"recent_temp_{i}.tmp"), 'w') as f:
                f.write(f"Recent temp file {i} content\n" * 100)
        
        # Old files (should be deleted)
        old_time = time.time() - (4 * 24 * 3600)  # 4 days old
        for i in range(10):
            filepath = os.path.join(temp_dir, f"old_temp_{i}.tmp")
            with open(filepath, 'w') as f:
                f.write(f"Old temp file {i} content\n" * 50)
            os.utime(filepath, (old_time, old_time))
    
    def _create_large_files(self) -> None:
        """Create large files for testing"""
        large_dir = os.path.join(self.temp_test_dir, "large_files")
        
        # Create files of various sizes
        sizes_mb = [15, 25, 50, 100]  # MB
        
        for size_mb in sizes_mb:
            filepath = os.path.join(large_dir, f"large_file_{size_mb}mb.dat")
            with open(filepath, 'w') as f:
                # Write random data to reach target size
                chunk_size = 1024  # 1KB chunks
                chunks_needed = (size_mb * 1024 * 1024) // chunk_size
                chunk_data = 'x' * chunk_size
                
                for _ in range(chunks_needed):
                    f.write(chunk_data)
    
    def _create_duplicate_files(self) -> None:
        """Create duplicate files for deduplication testing"""
        dup_dir = os.path.join(self.temp_test_dir, "duplicate_files")
        
        # Create identical files
        original_content = "This is duplicate file content\n" * 1000
        
        for i in range(5):
            with open(os.path.join(dup_dir, f"duplicate_{i}.txt"), 'w') as f:
                f.write(original_content)
        
        # Create another set of identical files
        other_content = "Different duplicate content\n" * 500
        
        for i in range(3):
            with open(os.path.join(dup_dir, f"other_duplicate_{i}.txt"), 'w') as f:
                f.write(other_content)
    
    def _create_compressible_files(self) -> None:
        """Create files suitable for compression testing"""
        comp_dir = os.path.join(self.temp_test_dir, "compressible_files")
        
        # Create log-like files with repetitive content
        log_content = """
[INFO] 2024-08-10 10:00:00 - Application started
[INFO] 2024-08-10 10:00:01 - Loading configuration
[INFO] 2024-08-10 10:00:02 - Database connection established
[ERROR] 2024-08-10 10:00:03 - Failed to load module xyz
[INFO] 2024-08-10 10:00:04 - Retrying module load
[INFO] 2024-08-10 10:00:05 - Module loaded successfully
""" * 500
        
        # Create old log files (suitable for compression)
        old_time = time.time() - (40 * 24 * 3600)  # 40 days old
        
        for i in range(5):
            filepath = os.path.join(comp_dir, f"old_application_{i}.log")
            with open(filepath, 'w') as f:
                f.write(log_content)
            os.utime(filepath, (old_time, old_time))
        
        # Create JSON files with repetitive structure
        json_content = json.dumps({
            "timestamp": "2024-08-10T10:00:00Z",
            "level": "INFO",
            "message": "Repetitive log message",
            "service": "test-service",
            "metadata": {"key": "value", "repeated": True}
        }) + "\n"
        
        for i in range(3):
            filepath = os.path.join(comp_dir, f"data_{i}.json")
            with open(filepath, 'w') as f:
                f.write(json_content * 1000)  # Repeat 1000 times
            os.utime(filepath, (old_time, old_time))
    
    def _create_edge_case_files(self) -> None:
        """Create edge case files for robust testing"""
        edge_dir = os.path.join(self.temp_test_dir, "edge_case_files")
        
        # Empty file
        open(os.path.join(edge_dir, "empty_file.txt"), 'w').close()
        
        # File with special characters in name
        with open(os.path.join(edge_dir, "special_chars_@#$%.txt"), 'w') as f:
            f.write("File with special characters in name\n")
        
        # Very long filename
        long_name = "very_long_filename_" + "x" * 100 + ".txt"
        with open(os.path.join(edge_dir, long_name), 'w') as f:
            f.write("File with very long name\n")
        
        # Binary-like file
        with open(os.path.join(edge_dir, "binary_like.dat"), 'wb') as f:
            f.write(bytes(range(256)) * 100)
        
        # File with only whitespace
        with open(os.path.join(edge_dir, "whitespace_only.txt"), 'w') as f:
            f.write(" " * 1000 + "\n" * 500 + "\t" * 200)
    
    def _verify_service_health(self) -> None:
        """Verify that the hardware optimizer service is available and healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("status") == "healthy":
                    logger.info("Service health verification passed")
                    return
            
            raise Exception(f"Service not healthy: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Service health check failed: {e}")
            raise RuntimeError("Hardware optimizer service is not available for testing")
    
    async def _make_request(self, endpoint: EndpointSpec, session: aiohttp.ClientSession, 
                          monitor: SystemMonitor, request_id: int = 0) -> TestResult:
        """Make a single HTTP request and measure performance"""
        start_time = time.time()
        
        try:
            # Prepare request parameters
            url = f"{self.base_url}{endpoint.path}"
            
            if endpoint.method == "GET":
                async with session.get(url, params=endpoint.params, 
                                     timeout=aiohttp.ClientTimeout(total=endpoint.timeout)) as response:
                    response_data = await response.text()
                    response_size = len(response_data)
                    status_code = response.status
            else:  # POST
                async with session.post(url, params=endpoint.params,
                                      timeout=aiohttp.ClientTimeout(total=endpoint.timeout)) as response:
                    response_data = await response.text()
                    response_size = len(response_data)
                    status_code = response.status
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Get current system metrics
            current_metrics = monitor.metrics[-1] if monitor.metrics else {}
            memory_usage_mb = current_metrics.get('process_memory_mb', 0.0)
            cpu_percent = current_metrics.get('cpu_percent', 0.0)
            
            success = 200 <= status_code < 300
            
            return TestResult(
                endpoint=endpoint.path,
                method=endpoint.method,
                success=success,
                response_time_ms=response_time_ms,
                status_code=status_code,
                response_size=response_size,
                memory_usage_mb=memory_usage_mb,
                cpu_percent=cpu_percent,
                error_message=None if success else f"HTTP {status_code}: {response_data[:200]}"
            )
            
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return TestResult(
                endpoint=endpoint.path,
                method=endpoint.method,
                success=False,
                response_time_ms=response_time_ms,
                status_code=0,
                response_size=0,
                memory_usage_mb=0.0,
                cpu_percent=0.0,
                error_message=str(e)
            )
    
    async def _run_load_test_for_endpoint(self, endpoint: EndpointSpec, 
                                        concurrent_users: int) -> LoadTestMetrics:
        """Run load test for a single endpoint with specified concurrency"""
        logger.info(f"Starting load test: {endpoint.method} {endpoint.path} with {concurrent_users} users")
        
        # Start system monitoring
        monitor = SystemMonitor()
        monitor.start()
        
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=concurrent_users * 2,
            limit_per_host=concurrent_users * 2,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=endpoint.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Calculate test parameters
            requests_per_user = max(1, (self.TEST_DURATION * concurrent_users) // 10)  # Reasonable request rate
            total_requests = concurrent_users * requests_per_user
            
            logger.info(f"Test parameters: {total_requests} total requests, {requests_per_user} per user")
            
            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(concurrent_users)
            
            async def limited_request(request_id: int) -> TestResult:
                async with semaphore:
                    return await self._make_request(endpoint, session, monitor, request_id)
            
            # Start test
            start_time = time.time()
            
            # Create all tasks
            tasks = [limited_request(i) for i in range(total_requests)]
            
            # Execute with controlled concurrency
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            total_duration = end_time - start_time
            
        # Stop monitoring and get summary
        monitor_summary = monitor.stop()
        
        # Process results
        valid_results = [r for r in results if isinstance(r, TestResult)]
        exception_count = len(results) - len(valid_results)
        
        if not valid_results:
            logger.error(f"No valid results for {endpoint.path}")
            return self._create_empty_metrics(endpoint, concurrent_users)
        
        # Calculate metrics
        successful_results = [r for r in valid_results if r.success]
        failed_results = [r for r in valid_results if not r.success]
        
        response_times = [r.response_time_ms for r in valid_results]
        success_rate = (len(successful_results) / len(valid_results)) * 100
        
        # Calculate percentiles
        response_times.sort()
        p95_index = int(len(response_times) * 0.95)
        p99_index = int(len(response_times) * 0.99)
        
        # Memory leak detection
        memory_peak = monitor_summary.get('process_memory_stats', {}).get('peak_mb', 0)
        memory_leak_detected = monitor_summary.get('process_memory_stats', {}).get('leak_detected', False)
        
        # SLA compliance check
        p95_response_time = response_times[p95_index] if response_times else 0
        sla_compliance = (success_rate >= self.SLA_SUCCESS_RATE and 
                         p95_response_time <= self.SLA_RESPONSE_TIME_MS and
                         memory_peak <= self.MAX_MEMORY_USAGE_MB)
        
        # Error analysis
        errors_by_type = {}
        for result in failed_results:
            if result.error_message:
                error_type = result.error_message.split(':')[0]
                errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
        
        if exception_count > 0:
            errors_by_type['Exception'] = exception_count
        
        metrics = LoadTestMetrics(
            endpoint=endpoint.path,
            concurrent_users=concurrent_users,
            total_requests=len(valid_results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            success_rate=success_rate,
            avg_response_time_ms=statistics.mean(response_times) if response_times else 0,
            median_response_time_ms=statistics.median(response_times) if response_times else 0,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=response_times[p99_index] if response_times else 0,
            min_response_time_ms=min(response_times) if response_times else 0,
            max_response_time_ms=max(response_times) if response_times else 0,
            throughput_rps=len(valid_results) / total_duration if total_duration > 0 else 0,
            total_duration_s=total_duration,
            memory_peak_mb=memory_peak,
            memory_leak_detected=memory_leak_detected,
            sla_compliance=sla_compliance,
            errors_by_type=errors_by_type
        )
        
        logger.info(f"Load test completed: {endpoint.path} - Success: {success_rate:.1f}%, "
                   f"P95: {p95_response_time:.1f}ms, SLA: {'PASS' if sla_compliance else 'FAIL'}")
        
        return metrics
    
    def _create_empty_metrics(self, endpoint: EndpointSpec, concurrent_users: int) -> LoadTestMetrics:
        """Create empty metrics for failed tests"""
        return LoadTestMetrics(
            endpoint=endpoint.path,
            concurrent_users=concurrent_users,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            success_rate=0.0,
            avg_response_time_ms=0.0,
            median_response_time_ms=0.0,
            p95_response_time_ms=0.0,
            p99_response_time_ms=0.0,
            min_response_time_ms=0.0,
            max_response_time_ms=0.0,
            throughput_rps=0.0,
            total_duration_s=0.0,
            memory_peak_mb=0.0,
            memory_leak_detected=False,
            sla_compliance=False,
            errors_by_type={"TestFailure": 1}
        )
    
    async def run_comprehensive_load_tests(self) -> List[LoadTestMetrics]:
        """Run comprehensive load tests across all endpoints and load levels"""
        logger.info("Starting comprehensive load tests for all endpoints")
        
        all_results = []
        
        for endpoint in self.ENDPOINTS:
            for concurrent_users in self.LOAD_LEVELS:
                try:
                    # Small delay between tests to allow system recovery
                    if all_results:  # Not the first test
                        await asyncio.sleep(5)
                        gc.collect()  # Force garbage collection
                    
                    metrics = await self._run_load_test_for_endpoint(endpoint, concurrent_users)
                    all_results.append(metrics)
                    self.load_test_results.append(metrics)
                    
                except Exception as e:
                    logger.error(f"Load test failed for {endpoint.path} with {concurrent_users} users: {e}")
                    empty_metrics = self._create_empty_metrics(endpoint, concurrent_users)
                    all_results.append(empty_metrics)
                    self.load_test_results.append(empty_metrics)
        
        logger.info(f"Comprehensive load tests completed: {len(all_results)} test scenarios")
        return all_results
    
    def run_security_boundary_tests(self) -> List[SecurityTestResult]:
        """Run security boundary and authentication tests"""
        logger.info("Starting security boundary tests")
        
        security_results = []
        
        # Test 1: Path traversal attempts
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "../../../../../../../../etc/hosts",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
        
        for payload in path_traversal_payloads:
            try:
                response = requests.get(
                    f"{self.base_url}/analyze/storage",
                    params={"path": payload},
                    timeout=10
                )
                
                # Should not access system files
                vulnerability_detected = (
                    response.status_code == 200 and 
                    "root:" in response.text.lower()
                )
                
                security_results.append(SecurityTestResult(
                    test_type="Path Traversal",
                    endpoint="/analyze/storage",
                    success=True,
                    vulnerability_detected=vulnerability_detected,
                    details=f"Payload: {payload}, Status: {response.status_code}",
                    severity="HIGH" if vulnerability_detected else "LOW"
                ))
                
            except Exception as e:
                security_results.append(SecurityTestResult(
                    test_type="Path Traversal",
                    endpoint="/analyze/storage",
                    success=False,
                    vulnerability_detected=False,
                    details=f"Test failed: {e}",
                    severity="LOW"
                ))
        
        # Test 2: Large parameter injection
        large_params = {
            "path": "/tmp/" + "A" * 10000,  # Very long path
            "min_size_mb": 999999999,  # Extremely large number
            "days_old": -1  # Negative number
        }
        
        endpoints_to_test = [
            "/analyze/storage",
            "/analyze/storage/duplicates", 
            "/analyze/storage/large-files",
            "/optimize/storage/compress"
        ]
        
        for endpoint in endpoints_to_test:
            for param_name, param_value in large_params.items():
                try:
                    response = requests.get(
                        f"{self.base_url}{endpoint}",
                        params={param_name: param_value},
                        timeout=30
                    )
                    
                    # Should handle invalid parameters gracefully
                    vulnerability_detected = response.status_code == 500
                    
                    security_results.append(SecurityTestResult(
                        test_type="Parameter Injection",
                        endpoint=endpoint,
                        success=True,
                        vulnerability_detected=vulnerability_detected,
                        details=f"Param: {param_name}={param_value}, Status: {response.status_code}",
                        severity="MEDIUM" if vulnerability_detected else "LOW"
                    ))
                    
                except Exception as e:
                    security_results.append(SecurityTestResult(
                        test_type="Parameter Injection",
                        endpoint=endpoint,
                        success=False,
                        vulnerability_detected=False,
                        details=f"Test failed: {e}",
                        severity="LOW"
                    ))
        
        # Test 3: Unauthorized optimization attempts
        destructive_endpoints = [
            "/optimize/storage",
            "/optimize/storage/duplicates",
            "/optimize/disk",
            "/optimize/all"
        ]
        
        for endpoint in destructive_endpoints:
            try:
                # Try without dry_run parameter (should be safe by default)
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    timeout=30
                )
                
                # Should use dry_run by default or require explicit permission
                if response.status_code == 200:
                    response_data = response.json()
                    dry_run_used = response_data.get("dry_run", True)
                    vulnerability_detected = not dry_run_used
                else:
                    vulnerability_detected = False
                
                security_results.append(SecurityTestResult(
                    test_type="Unauthorized Optimization",
                    endpoint=endpoint,
                    success=True,
                    vulnerability_detected=vulnerability_detected,
                    details=f"Status: {response.status_code}, Dry run: {dry_run_used if response.status_code == 200 else 'N/A'}",
                    severity="HIGH" if vulnerability_detected else "LOW"
                ))
                
            except Exception as e:
                security_results.append(SecurityTestResult(
                    test_type="Unauthorized Optimization", 
                    endpoint=endpoint,
                    success=False,
                    vulnerability_detected=False,
                    details=f"Test failed: {e}",
                    severity="LOW"
                ))
        
        self.security_test_results = security_results
        logger.info(f"Security boundary tests completed: {len(security_results)} tests")
        
        return security_results
    
    def run_stress_tests(self) -> List[Dict[str, Any]]:
        """Run stress testing scenarios"""
        logger.info("Starting stress testing scenarios")
        
        stress_results = []
        
        # Stress Test 1: Spike Testing - Sudden load increase
        logger.info("Running spike stress test")
        spike_result = self._run_spike_test()
        stress_results.append(spike_result)
        
        # Stress Test 2: Endurance Testing - Sustained load
        logger.info("Running endurance stress test")
        endurance_result = self._run_endurance_test()
        stress_results.append(endurance_result)
        
        # Stress Test 3: Volume Testing - Large data processing
        logger.info("Running volume stress test")
        volume_result = self._run_volume_test()
        stress_results.append(volume_result)
        
        # Stress Test 4: Memory Exhaustion Testing
        logger.info("Running memory exhaustion test")
        memory_result = self._run_memory_exhaustion_test()
        stress_results.append(memory_result)
        
        self.stress_test_results = stress_results
        logger.info(f"Stress testing completed: {len(stress_results)} scenarios")
        
        return stress_results
    
    def _run_spike_test(self) -> Dict[str, Any]:
        """Simulate sudden spike in load"""
        try:
            # Start with low load
            low_load_results = []
            for _ in range(10):
                response = requests.get(f"{self.base_url}/health", timeout=10)
                low_load_results.append(response.elapsed.total_seconds() * 1000)
                time.sleep(0.1)
            
            # Sudden spike to high load
            spike_start = time.time()
            spike_results = []
            
            # Create 50 simultaneous requests
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = []
                for _ in range(50):
                    future = executor.submit(self._make_spike_request)
                    futures.append(future)
                
                for future in as_completed(futures, timeout=60):
                    try:
                        result = future.result()
                        spike_results.append(result)
                    except Exception as e:
                        spike_results.append({"success": False, "error": str(e)})
            
            spike_end = time.time()
            
            # Calculate recovery time
            recovery_start = time.time()
            recovery_responses = []
            
            while time.time() - recovery_start < 30:  # 30 second recovery window
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=5)
                    recovery_time = response.elapsed.total_seconds() * 1000
                    recovery_responses.append(recovery_time)
                    
                    # If response time is back to normal, service recovered
                    if recovery_time < statistics.mean(low_load_results) * 2:
                        break
                        
                except Exception:
                    pass
                
                time.sleep(1)
            
            recovery_time = time.time() - recovery_start
            
            # Analysis
            successful_spike = len([r for r in spike_results if r.get("success", False)])
            success_rate = (successful_spike / len(spike_results)) * 100
            
            return {
                "test_type": "Spike Test",
                "spike_duration_s": spike_end - spike_start,
                "spike_requests": len(spike_results),
                "successful_requests": successful_spike,
                "success_rate": success_rate,
                "recovery_time_s": recovery_time,
                "baseline_response_ms": statistics.mean(low_load_results),
                "passed": success_rate > 90 and recovery_time < 30,
                "details": f"Service handled {success_rate:.1f}% of spike load, recovered in {recovery_time:.1f}s"
            }
            
        except Exception as e:
            return {
                "test_type": "Spike Test",
                "passed": False,
                "error": str(e),
                "details": "Spike test failed with exception"
            }
    
    def _make_spike_request(self) -> Dict[str, Any]:
        """Make a single request during spike test"""
        try:
            start_time = time.time()
            response = requests.post(f"{self.base_url}/optimize/memory", timeout=30)
            end_time = time.time()
            
            return {
                "success": 200 <= response.status_code < 300,
                "response_time_ms": (end_time - start_time) * 1000,
                "status_code": response.status_code
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time_ms": 0
            }
    
    def _run_endurance_test(self) -> Dict[str, Any]:
        """Run sustained load over extended period"""
        try:
            logger.info("Starting 5-minute endurance test")
            
            # Monitor system during test
            monitor = SystemMonitor(interval=1.0)
            monitor.start()
            
            # Run sustained load for 5 minutes
            endurance_start = time.time()
            endurance_duration = 300  # 5 minutes
            
            results = []
            error_count = 0
            
            while time.time() - endurance_start < endurance_duration:
                try:
                    response = requests.get(f"{self.base_url}/status", timeout=10)
                    success = 200 <= response.status_code < 300
                    results.append({
                        "timestamp": time.time(),
                        "success": success,
                        "response_time_ms": response.elapsed.total_seconds() * 1000
                    })
                    
                    if not success:
                        error_count += 1
                    
                    # Small delay to avoid overwhelming
                    time.sleep(2)
                    
                except Exception as e:
                    error_count += 1
                    results.append({
                        "timestamp": time.time(),
                        "success": False,
                        "error": str(e)
                    })
                    time.sleep(2)
            
            # Stop monitoring
            monitor_summary = monitor.stop()
            
            # Analysis
            total_requests = len(results)
            successful_requests = len([r for r in results if r.get("success", False)])
            success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0
            
            response_times = [r.get("response_time_ms", 0) for r in results if r.get("success", False)]
            avg_response_time = statistics.mean(response_times) if response_times else 0
            
            # Check for performance degradation over time
            first_half = response_times[:len(response_times)//2]
            second_half = response_times[len(response_times)//2:]
            
            degradation = False
            if first_half and second_half:
                first_avg = statistics.mean(first_half)
                second_avg = statistics.mean(second_half)
                degradation = second_avg > first_avg * 1.5  # 50% degradation threshold
            
            return {
                "test_type": "Endurance Test",
                "duration_s": endurance_duration,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": success_rate,
                "avg_response_time_ms": avg_response_time,
                "performance_degradation": degradation,
                "system_stats": monitor_summary,
                "passed": success_rate > 95 and not degradation,
                "details": f"Sustained {success_rate:.1f}% success rate over {endurance_duration}s"
            }
            
        except Exception as e:
            return {
                "test_type": "Endurance Test",
                "passed": False,
                "error": str(e),
                "details": "Endurance test failed with exception"
            }
    
    def _run_volume_test(self) -> Dict[str, Any]:
        """Test handling of large data volumes"""
        try:
            # Create large test dataset
            large_data_dir = os.path.join(self.temp_test_dir, "volume_test")
            os.makedirs(large_data_dir, exist_ok=True)
            
            # Create many files for volume testing
            file_count = 1000
            files_created = 0
            
            for i in range(file_count):
                try:
                    with open(os.path.join(large_data_dir, f"volume_file_{i:04d}.txt"), 'w') as f:
                        f.write(f"Volume test file {i}\n" * 100)
                    files_created += 1
                except Exception:
                    break
            
            logger.info(f"Created {files_created} files for volume testing")
            
            # Test storage analysis with large dataset
            start_time = time.time()
            
            try:
                response = requests.get(
                    f"{self.base_url}/analyze/storage",
                    params={"path": large_data_dir},
                    timeout=120  # Longer timeout for large data
                )
                
                analysis_time = time.time() - start_time
                success = response.status_code == 200
                
                if success:
                    data = response.json()
                    files_analyzed = data.get("total_files", 0)
                    
                    # Verify all files were processed
                    complete_analysis = files_analyzed >= files_created * 0.9  # Allow 10% tolerance
                else:
                    files_analyzed = 0
                    complete_analysis = False
                
            except Exception as e:
                analysis_time = time.time() - start_time
                success = False
                files_analyzed = 0
                complete_analysis = False
            
            # Cleanup
            try:
                shutil.rmtree(large_data_dir)
            except Exception:
                pass
            
            return {
                "test_type": "Volume Test",
                "files_created": files_created,
                "files_analyzed": files_analyzed,
                "analysis_time_s": analysis_time,
                "analysis_success": success,
                "complete_analysis": complete_analysis,
                "passed": success and complete_analysis and analysis_time < 60,
                "details": f"Analyzed {files_analyzed}/{files_created} files in {analysis_time:.1f}s"
            }
            
        except Exception as e:
            return {
                "test_type": "Volume Test",
                "passed": False,
                "error": str(e),
                "details": "Volume test failed with exception"
            }
    
    def _run_memory_exhaustion_test(self) -> Dict[str, Any]:
        """Test behavior under memory pressure"""
        try:
            # Get initial memory state
            initial_memory = psutil.virtual_memory()
            process = psutil.Process()
            initial_process_memory = process.memory_info().rss / (1024*1024)  # MB
            
            # Start memory monitoring
            monitor = SystemMonitor(interval=0.5)
            monitor.start()
            
            # Simulate memory pressure by making many concurrent requests
            memory_test_results = []
            
            with ThreadPoolExecutor(max_workers=20) as executor:
                # Submit many requests to potentially cause memory buildup
                futures = []
                
                for i in range(100):  # 100 concurrent requests
                    future = executor.submit(self._memory_test_request, i)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures, timeout=180):  # 3 minute timeout
                    try:
                        result = future.result()
                        memory_test_results.append(result)
                    except Exception as e:
                        memory_test_results.append({
                            "success": False,
                            "error": str(e)
                        })
            
            # Stop monitoring
            monitor_summary = monitor.stop()
            
            # Analyze memory usage
            final_memory = psutil.virtual_memory()
            final_process_memory = process.memory_info().rss / (1024*1024)  # MB
            
            memory_growth_mb = final_process_memory - initial_process_memory
            peak_memory_mb = monitor_summary.get('process_memory_stats', {}).get('peak_mb', 0)
            
            # Check for memory leaks
            memory_leak_detected = memory_growth_mb > 100  # >100MB growth indicates potential leak
            
            # Success metrics
            successful_requests = len([r for r in memory_test_results if r.get("success", False)])
            success_rate = (successful_requests / len(memory_test_results)) * 100 if memory_test_results else 0
            
            return {
                "test_type": "Memory Exhaustion Test",
                "total_requests": len(memory_test_results),
                "successful_requests": successful_requests,
                "success_rate": success_rate,
                "initial_memory_mb": initial_process_memory,
                "final_memory_mb": final_process_memory,
                "memory_growth_mb": memory_growth_mb,
                "peak_memory_mb": peak_memory_mb,
                "memory_leak_detected": memory_leak_detected,
                "system_memory_stats": {
                    "initial_percent": initial_memory.percent,
                    "final_percent": final_memory.percent
                },
                "passed": success_rate > 90 and not memory_leak_detected and peak_memory_mb < self.MAX_MEMORY_USAGE_MB,
                "details": f"Memory growth: {memory_growth_mb:.1f}MB, Peak: {peak_memory_mb:.1f}MB, Success: {success_rate:.1f}%"
            }
            
        except Exception as e:
            return {
                "test_type": "Memory Exhaustion Test",
                "passed": False,
                "error": str(e),
                "details": "Memory exhaustion test failed with exception"
            }
    
    def _memory_test_request(self, request_id: int) -> Dict[str, Any]:
        """Make a single request during memory testing"""
        try:
            # Use different endpoints to vary memory usage patterns
            endpoints = [
                "/health",
                "/status", 
                "/analyze/storage?path=/tmp",
                "/analyze/storage/report"
            ]
            
            endpoint = endpoints[request_id % len(endpoints)]
            
            start_time = time.time()
            response = requests.get(f"{self.base_url}{endpoint}", timeout=30)
            end_time = time.time()
            
            return {
                "request_id": request_id,
                "endpoint": endpoint,
                "success": 200 <= response.status_code < 300,
                "response_time_ms": (end_time - start_time) * 1000,
                "status_code": response.status_code,
                "response_size": len(response.content) if hasattr(response, 'content') else 0
            }
            
        except Exception as e:
            return {
                "request_id": request_id,
                "success": False,
                "error": str(e)
            }
    
    def run_error_injection_tests(self) -> List[Dict[str, Any]]:
        """Run error injection and recovery tests"""
        logger.info("Starting error injection and recovery tests")
        
        error_injection_results = []
        
        # Error Injection Test 1: Invalid paths
        invalid_paths = [
            "/nonexistent/path/that/does/not/exist",
            "/root",  # Should be protected
            "/proc/self/mem",  # Should be blocked
            "",  # Empty path
            None  # Null path
        ]
        
        for path in invalid_paths:
            try:
                params = {"path": path} if path is not None else {}
                response = requests.get(
                    f"{self.base_url}/analyze/storage",
                    params=params,
                    timeout=10
                )
                
                # Should handle gracefully without crashing
                graceful_handling = response.status_code in [400, 422, 404]  # Expected error codes
                
                error_injection_results.append({
                    "test_type": "Invalid Path Injection",
                    "input": str(path),
                    "status_code": response.status_code,
                    "graceful_handling": graceful_handling,
                    "passed": graceful_handling,
                    "details": f"Path: {path}, Status: {response.status_code}"
                })
                
            except Exception as e:
                error_injection_results.append({
                    "test_type": "Invalid Path Injection",
                    "input": str(path),
                    "passed": False,
                    "error": str(e),
                    "details": f"Exception during test: {e}"
                })
        
        # Error Injection Test 2: Malformed parameters
        malformed_params = [
            {"min_size_mb": "not_a_number"},
            {"days_old": "invalid"},
            {"dry_run": "maybe"},
            {"path": "x" * 100000},  # Extremely long path
        ]
        
        for params in malformed_params:
            try:
                response = requests.get(
                    f"{self.base_url}/analyze/storage/large-files",
                    params=params,
                    timeout=10
                )
                
                # Should handle parameter validation
                validation_handling = response.status_code in [400, 422]
                
                error_injection_results.append({
                    "test_type": "Malformed Parameter Injection",
                    "input": str(params),
                    "status_code": response.status_code,
                    "validation_handling": validation_handling,
                    "passed": validation_handling,
                    "details": f"Params: {params}, Status: {response.status_code}"
                })
                
            except Exception as e:
                error_injection_results.append({
                    "test_type": "Malformed Parameter Injection",
                    "input": str(params),
                    "passed": False,
                    "error": str(e),
                    "details": f"Exception during test: {e}"
                })
        
        # Error Injection Test 3: Service recovery after errors
        logger.info("Testing service recovery after error injection")
        
        # Inject a series of bad requests
        for _ in range(20):
            try:
                requests.get(
                    f"{self.base_url}/analyze/storage",
                    params={"path": "/invalid/path/that/should/fail"},
                    timeout=5
                )
            except Exception:
                pass  # Expected to fail
        
        # Test if service still responds normally after error injection
        recovery_test_start = time.time()
        recovery_successful = False
        
        for attempt in range(10):  # Try for up to 10 attempts
            try:
                response = requests.get(f"{self.base_url}/health", timeout=10)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("status") == "healthy":
                        recovery_successful = True
                        break
            except Exception:
                pass
            
            time.sleep(1)  # Wait before retry
        
        recovery_time = time.time() - recovery_test_start
        
        error_injection_results.append({
            "test_type": "Service Recovery After Errors",
            "recovery_successful": recovery_successful,
            "recovery_time_s": recovery_time,
            "passed": recovery_successful and recovery_time < 30,
            "details": f"Service recovered: {recovery_successful}, Time: {recovery_time:.1f}s"
        })
        
        self.error_injection_results = error_injection_results
        logger.info(f"Error injection tests completed: {len(error_injection_results)} tests")
        
        return error_injection_results
    
    def generate_performance_charts(self) -> None:
        """Generate performance visualization charts"""
        logger.info("Generating performance charts")
        
        try:
            # Create output directory
            charts_dir = "/opt/sutazaiapp/tests/performance_charts"
            os.makedirs(charts_dir, exist_ok=True)
            
            # Chart 1: Response Time vs Concurrent Users
            self._create_response_time_chart(charts_dir)
            
            # Chart 2: Success Rate vs Load Level
            self._create_success_rate_chart(charts_dir)
            
            # Chart 3: Throughput Analysis
            self._create_throughput_chart(charts_dir)
            
            # Chart 4: Memory Usage Analysis
            self._create_memory_usage_chart(charts_dir)
            
            logger.info(f"Performance charts generated in {charts_dir}")
            
        except Exception as e:
            logger.error(f"Error generating performance charts: {e}")
    
    def _create_response_time_chart(self, output_dir: str) -> None:
        """Create response time vs concurrent users chart"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group results by endpoint
        endpoint_data = {}
        for result in self.load_test_results:
            if result.endpoint not in endpoint_data:
                endpoint_data[result.endpoint] = {"users": [], "p95_times": []}
            endpoint_data[result.endpoint]["users"].append(result.concurrent_users)
            endpoint_data[result.endpoint]["p95_times"].append(result.p95_response_time_ms)
        
        # Plot lines for each endpoint
        for endpoint, data in endpoint_data.items():
            if data["users"] and data["p95_times"]:
                ax.plot(data["users"], data["p95_times"], marker='o', label=endpoint[:30])
        
        # Add SLA line
        ax.axhline(y=self.SLA_RESPONSE_TIME_MS, color='red', linestyle='--', 
                  label=f'SLA Limit ({self.SLA_RESPONSE_TIME_MS}ms)')
        
        ax.set_xlabel('Concurrent Users')
        ax.set_ylabel('95th Percentile Response Time (ms)')
        ax.set_title('Response Time vs Concurrent Users (P95)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'response_time_vs_users.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_success_rate_chart(self, output_dir: str) -> None:
        """Create success rate vs load level chart"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group results by endpoint
        endpoint_data = {}
        for result in self.load_test_results:
            if result.endpoint not in endpoint_data:
                endpoint_data[result.endpoint] = {"users": [], "success_rates": []}
            endpoint_data[result.endpoint]["users"].append(result.concurrent_users)
            endpoint_data[result.endpoint]["success_rates"].append(result.success_rate)
        
        # Plot lines for each endpoint
        for endpoint, data in endpoint_data.items():
            if data["users"] and data["success_rates"]:
                ax.plot(data["users"], data["success_rates"], marker='s', label=endpoint[:30])
        
        # Add SLA line
        ax.axhline(y=self.SLA_SUCCESS_RATE, color='red', linestyle='--',
                  label=f'SLA Limit ({self.SLA_SUCCESS_RATE}%)')
        
        ax.set_xlabel('Concurrent Users')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate vs Concurrent Users')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'success_rate_vs_users.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_throughput_chart(self, output_dir: str) -> None:
        """Create throughput analysis chart"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group results by endpoint
        endpoint_data = {}
        for result in self.load_test_results:
            if result.endpoint not in endpoint_data:
                endpoint_data[result.endpoint] = {"users": [], "throughput": []}
            endpoint_data[result.endpoint]["users"].append(result.concurrent_users)
            endpoint_data[result.endpoint]["throughput"].append(result.throughput_rps)
        
        # Plot lines for each endpoint
        for endpoint, data in endpoint_data.items():
            if data["users"] and data["throughput"]:
                ax.plot(data["users"], data["throughput"], marker='^', label=endpoint[:30])
        
        ax.set_xlabel('Concurrent Users')
        ax.set_ylabel('Throughput (Requests/Second)')
        ax.set_title('Throughput vs Concurrent Users')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'throughput_vs_users.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_memory_usage_chart(self, output_dir: str) -> None:
        """Create memory usage analysis chart"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group results by concurrent users
        user_data = {}
        for result in self.load_test_results:
            if result.concurrent_users not in user_data:
                user_data[result.concurrent_users] = []
            user_data[result.concurrent_users].append(result.memory_peak_mb)
        
        # Calculate statistics for each user level
        users = sorted(user_data.keys())
        max_memory = [max(user_data[u]) if user_data[u] else 0 for u in users]
        avg_memory = [statistics.mean(user_data[u]) if user_data[u] else 0 for u in users]
        
        ax.plot(users, max_memory, marker='o', label='Peak Memory Usage', color='red')
        ax.plot(users, avg_memory, marker='s', label='Average Memory Usage', color='blue')
        
        # Add memory limit line
        ax.axhline(y=self.MAX_MEMORY_USAGE_MB, color='orange', linestyle='--',
                  label=f'Memory Limit ({self.MAX_MEMORY_USAGE_MB}MB)')
        
        ax.set_xlabel('Concurrent Users')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Usage vs Concurrent Users')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'memory_usage_vs_users.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive testing report with all metrics and analysis"""
        logger.info("Generating comprehensive testing report")
        
        # Calculate overall statistics
        total_tests = len(self.load_test_results)
        sla_compliant_tests = len([r for r in self.load_test_results if r.sla_compliance])
        overall_sla_compliance = (sla_compliant_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Memory leak analysis
        memory_leaks_detected = len([r for r in self.load_test_results if r.memory_leak_detected])
        
        # Security analysis
        security_vulnerabilities = len([r for r in self.security_test_results if r.vulnerability_detected])
        high_severity_vulns = len([r for r in self.security_test_results 
                                 if r.vulnerability_detected and r.severity == "HIGH"])
        
        # Stress test analysis
        stress_tests_passed = len([r for r in self.stress_test_results if r.get("passed", False)])
        
        # Error injection analysis
        error_tests_passed = len([r for r in self.error_injection_results if r.get("passed", False)])
        
        # Overall assessment
        critical_issues = []
        if overall_sla_compliance < 95:
            critical_issues.append(f"SLA compliance below 95%: {overall_sla_compliance:.1f}%")
        if memory_leaks_detected > 0:
            critical_issues.append(f"Memory leaks detected in {memory_leaks_detected} tests")
        if high_severity_vulns > 0:
            critical_issues.append(f"High severity security vulnerabilities: {high_severity_vulns}")
        if stress_tests_passed < len(self.stress_test_results):
            critical_issues.append("Some stress tests failed")
        
        overall_assessment = "PASS" if not critical_issues else "FAIL"
        
        report = {
            "test_execution_summary": {
                "timestamp": datetime.now().isoformat(),
                "test_duration_hours": time.time() / 3600,  # Approximate
                "overall_assessment": overall_assessment,
                "critical_issues": critical_issues
            },
            "load_testing_summary": {
                "total_load_tests": total_tests,
                "sla_compliant_tests": sla_compliant_tests,
                "overall_sla_compliance_percent": overall_sla_compliance,
                "endpoints_tested": len(self.ENDPOINTS),
                "load_levels_tested": len(self.LOAD_LEVELS),
                "memory_leaks_detected": memory_leaks_detected,
                "performance_metrics": {
                    "avg_response_time_ms": statistics.mean([r.avg_response_time_ms for r in self.load_test_results]) if self.load_test_results else 0,
                    "avg_success_rate_percent": statistics.mean([r.success_rate for r in self.load_test_results]) if self.load_test_results else 0,
                    "avg_throughput_rps": statistics.mean([r.throughput_rps for r in self.load_test_results]) if self.load_test_results else 0,
                    "peak_memory_usage_mb": max([r.memory_peak_mb for r in self.load_test_results]) if self.load_test_results else 0
                }
            },
            "security_testing_summary": {
                "total_security_tests": len(self.security_test_results),
                "vulnerabilities_detected": security_vulnerabilities,
                "high_severity_vulnerabilities": high_severity_vulns,
                "security_assessment": "FAIL" if high_severity_vulns > 0 else "PASS"
            },
            "stress_testing_summary": {
                "total_stress_tests": len(self.stress_test_results),
                "stress_tests_passed": stress_tests_passed,
                "stress_test_success_rate": (stress_tests_passed / len(self.stress_test_results)) * 100 if self.stress_test_results else 0
            },
            "error_injection_summary": {
                "total_error_injection_tests": len(self.error_injection_results),
                "error_tests_passed": error_tests_passed,
                "error_recovery_success_rate": (error_tests_passed / len(self.error_injection_results)) * 100 if self.error_injection_results else 0
            },
            "detailed_results": {
                "load_test_results": [asdict(result) for result in self.load_test_results],
                "security_test_results": [asdict(result) for result in self.security_test_results],
                "stress_test_results": self.stress_test_results,
                "error_injection_results": self.error_injection_results
            },
            "sla_validation": {
                "response_time_sla_ms": self.SLA_RESPONSE_TIME_MS,
                "success_rate_sla_percent": self.SLA_SUCCESS_RATE,
                "memory_usage_limit_mb": self.MAX_MEMORY_USAGE_MB,
                "sla_compliance_by_endpoint": [
                    {
                        "endpoint": result.endpoint,
                        "concurrent_users": result.concurrent_users,
                        "sla_compliant": result.sla_compliance,
                        "p95_response_time_ms": result.p95_response_time_ms,
                        "success_rate_percent": result.success_rate,
                        "memory_peak_mb": result.memory_peak_mb
                    }
                    for result in self.load_test_results
                ]
            },
            "recommendations": self._generate_recommendations(critical_issues)
        }
        
        # Save report to file
        report_file = f"/opt/sutazaiapp/tests/ultra_hardware_optimizer_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive report saved to {report_file}")
        return report
    
    def _generate_recommendations(self, critical_issues: List[str]) -> List[str]:
        """Generate actionable recommendations based on test results"""
        recommendations = []
        
        if any("SLA compliance" in issue for issue in critical_issues):
            recommendations.append("Optimize endpoint performance to meet response time SLAs")
            recommendations.append("Consider implementing request queuing and rate limiting")
            recommendations.append("Review and optimize database queries and resource usage")
        
        if any("Memory leaks" in issue for issue in critical_issues):
            recommendations.append("Implement proper memory management and garbage collection")
            recommendations.append("Review object lifecycle and ensure proper cleanup")
            recommendations.append("Add memory monitoring and alerts in production")
        
        if any("security vulnerabilities" in issue.lower() for issue in critical_issues):
            recommendations.append("Fix identified security vulnerabilities immediately")
            recommendations.append("Implement input validation and sanitization")
            recommendations.append("Add security scanning to CI/CD pipeline")
        
        if any("stress tests failed" in issue for issue in critical_issues):
            recommendations.append("Improve error handling and recovery mechanisms")
            recommendations.append("Implement circuit breakers and bulkhead patterns")
            recommendations.append("Add auto-scaling capabilities for high load scenarios")
        
        # General recommendations
        recommendations.extend([
            "Implement comprehensive monitoring and alerting in production",
            "Set up automated performance regression testing",
            "Create detailed runbooks for operational scenarios",
            "Establish performance baselines and SLA monitoring"
        ])
        
        return recommendations
    
    def cleanup_test_environment(self) -> None:
        """Cleanup test environment and temporary files"""
        try:
            if self.temp_test_dir and os.path.exists(self.temp_test_dir):
                shutil.rmtree(self.temp_test_dir)
                logger.info(f"Test environment cleaned up: {self.temp_test_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up test environment: {e}")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests and generate final report"""
        logger.info("=" * 80)
        logger.info("STARTING ULTRA-COMPREHENSIVE HARDWARE OPTIMIZER TESTING")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Phase 1: Load Testing
            logger.info("Phase 1: Running comprehensive load tests")
            await self.run_comprehensive_load_tests()
            
            # Phase 2: Security Testing
            logger.info("Phase 2: Running security boundary tests")
            self.run_security_boundary_tests()
            
            # Phase 3: Stress Testing
            logger.info("Phase 3: Running stress tests")
            self.run_stress_tests()
            
            # Phase 4: Error Injection Testing
            logger.info("Phase 4: Running error injection tests")
            self.run_error_injection_tests()
            
            # Phase 5: Generate Visualizations
            logger.info("Phase 5: Generating performance charts")
            self.generate_performance_charts()
            
            # Phase 6: Generate Comprehensive Report
            logger.info("Phase 6: Generating comprehensive report")
            final_report = self.generate_comprehensive_report()
            
            total_time = time.time() - start_time
            
            logger.info("=" * 80)
            logger.info(f"ULTRA-COMPREHENSIVE TESTING COMPLETED IN {total_time:.1f} SECONDS")
            logger.info(f"OVERALL ASSESSMENT: {final_report['test_execution_summary']['overall_assessment']}")
            logger.info("=" * 80)
            
            return final_report
            
        except Exception as e:
            logger.error(f"Critical error during testing: {e}")
            logger.error(traceback.format_exc())
            raise
        
        finally:
            # Always cleanup
            self.cleanup_test_environment()

# Test execution functions for pytest compatibility
class TestHardwareOptimizerUltra:
    """Pytest test class for ultra-comprehensive testing"""
    
    @pytest.fixture(scope="class")
    def tester(self):
        """Create tester instance"""
        return UltraHardwareOptimizerTester()
    
    @pytest.mark.asyncio
    async def test_comprehensive_load_testing(self, tester):
        """Run comprehensive load testing"""
        results = await tester.run_comprehensive_load_tests()
        
        # Verify all endpoints were tested
        assert len(results) == len(tester.ENDPOINTS) * len(tester.LOAD_LEVELS)
        
        # Check SLA compliance
        sla_compliant = [r for r in results if r.sla_compliance]
        sla_compliance_rate = len(sla_compliant) / len(results) * 100
        
        assert sla_compliance_rate >= 90, f"SLA compliance rate {sla_compliance_rate:.1f}% below 90%"
    
    def test_security_boundary_testing(self, tester):
        """Run security boundary testing"""
        results = tester.run_security_boundary_tests()
        
        # Check for high severity vulnerabilities
        high_severity = [r for r in results if r.vulnerability_detected and r.severity == "HIGH"]
        assert len(high_severity) == 0, f"High severity vulnerabilities detected: {len(high_severity)}"
    
    def test_stress_testing_scenarios(self, tester):
        """Run stress testing scenarios"""
        results = tester.run_stress_tests()
        
        # Verify all stress tests passed
        passed_tests = [r for r in results if r.get("passed", False)]
        assert len(passed_tests) == len(results), f"Stress tests failed: {len(results) - len(passed_tests)}"
    
    def test_error_injection_and_recovery(self, tester):
        """Run error injection and recovery tests"""
        results = tester.run_error_injection_tests()
        
        # Verify error handling
        recovery_test = [r for r in results if r.get("test_type") == "Service Recovery After Errors"]
        assert len(recovery_test) > 0 and recovery_test[0].get("passed", False), "Service recovery test failed"

def run_standalone_testing():
    """Run testing as standalone script"""
    import asyncio
    
    async def main():
        logger.info("Starting standalone ultra-comprehensive testing")
        
        tester = UltraHardwareOptimizerTester()
        
        try:
            final_report = await tester.run_all_tests()
            
            logger.info("\n" + "=" * 80)
            logger.info("ULTRA-COMPREHENSIVE TESTING COMPLETED")
            logger.info("=" * 80)
            logger.info(f"Overall Assessment: {final_report['test_execution_summary']['overall_assessment']}")
            logger.info(f"Load Tests: {final_report['load_testing_summary']['sla_compliant_tests']}/{final_report['load_testing_summary']['total_load_tests']} SLA compliant")
            logger.info(f"Security Tests: {final_report['security_testing_summary']['vulnerabilities_detected']} vulnerabilities detected")
            logger.info(f"Stress Tests: {final_report['stress_testing_summary']['stress_tests_passed']}/{final_report['stress_testing_summary']['total_stress_tests']} passed")
            logger.info("=" * 80)
            
            if final_report['test_execution_summary']['critical_issues']:
                logger.error("CRITICAL ISSUES:")
                for issue in final_report['test_execution_summary']['critical_issues']:
                    logger.info(f"  - {issue}")
                logger.info("=" * 80)
            
            return final_report
            
        except Exception as e:
            logger.error(f"Testing failed: {e}")
            raise

    return asyncio.run(main())

if __name__ == "__main__":
    run_standalone_testing()