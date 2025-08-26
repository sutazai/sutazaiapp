#!/usr/bin/env python3
"""
Performance Benchmarks and Stress Tests for Hardware Resource Optimizer
=======================================================================

This module provides comprehensive performance testing and stress testing:
- Load testing with varying concurrent users
- Response time benchmarking
- Memory usage monitoring during tests
- Throughput measurement
- Resource exhaustion testing
- Recovery time measurement
- Performance regression detection

Author: QA Team Lead
Version: 1.0.0
"""

import os
import sys
import json
import time
import asyncio
import aiohttp
import threading
import statistics
import psutil
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PerformanceTests')

class PerformanceMonitor:
    """Monitor system performance during tests"""
    
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
        return self.get_summary()
    
    def _monitor_loop(self):
        """Monitor system metrics continuously"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                self.metrics.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': (disk.used / disk.total) * 100,
                    'disk_free_gb': disk.free / (1024**3)
                })
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                break
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary statistics"""
        if not self.metrics:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_percent'] for m in self.metrics]
        
        return {
            'duration': self.metrics[-1]['timestamp'] - self.metrics[0]['timestamp'],
            'samples': len(self.metrics),
            'cpu': {
                'avg': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'std': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            },
            'memory': {
                'avg': statistics.mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'std': statistics.stdev(memory_values) if len(memory_values) > 1 else 0
            },
            'peak_memory_gb': max(m['memory_used_gb'] for m in self.metrics)
        }

class StressTestScenarios:
    """Comprehensive stress testing scenarios"""
    
    def __init__(self, base_url: str = "http://localhost:8116", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
    
    def single_request_benchmark(self, method: str, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """Benchmark a single request multiple times"""
        times = []
        errors = []
        
        for i in range(50):  # 50 iterations for statistical significance
            try:
                start_time = time.time()
                
                if method.upper() == "GET":
                    response = self.session.get(f"{self.base_url}{endpoint}", params=params)
                elif method.upper() == "POST":
                    response = self.session.post(f"{self.base_url}{endpoint}", params=params)
                
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    times.append(duration)
                else:
                    errors.append(f"Status {response.status_code}")
                    
            except Exception as e:
                errors.append(str(e))
        
        if times:
            return {
                'endpoint': endpoint,
                'method': method,
                'successful_requests': len(times),
                'failed_requests': len(errors),
                'avg_response_time': statistics.mean(times),
                'min_response_time': min(times),
                'max_response_time': max(times),
                'median_response_time': statistics.median(times),
                'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
                'p95_response_time': np.percentile(times, 95) if times else 0,
                'p99_response_time': np.percentile(times, 99) if times else 0,
                'errors': errors[:10]  # First 10 errors
            }
        else:
            return {
                'endpoint': endpoint,
                'method': method,
                'successful_requests': 0,
                'failed_requests': len(errors),
                'error': 'All requests failed',
                'errors': errors[:10]
            }
    
    def concurrent_load_test(self, endpoint: str, method: str = "GET", 
                           concurrent_users: int = 10, requests_per_user: int = 5,
                           params: Dict = None) -> Dict[str, Any]:
        """Test with concurrent load"""
        logger.info(f"Running concurrent load test: {concurrent_users} users, {requests_per_user} requests each")
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        results = []
        start_time = time.time()
        
        def user_session():
            """Simulate a user session"""
            session_results = []
            for _ in range(requests_per_user):
                try:
                    req_start = time.time()
                    
                    if method.upper() == "GET":
                        response = self.session.get(f"{self.base_url}{endpoint}", params=params)
                    elif method.upper() == "POST":
                        response = self.session.post(f"{self.base_url}{endpoint}", params=params)
                    
                    duration = time.time() - req_start
                    
                    session_results.append({
                        'success': response.status_code == 200,
                        'duration': duration,
                        'status_code': response.status_code
                    })
                    
                except Exception as e:
                    session_results.append({
                        'success': False,
                        'duration': 0,
                        'error': str(e)
                    })
            
            return session_results
        
        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_session) for _ in range(concurrent_users)]
            
            for future in as_completed(futures):
                try:
                    user_results = future.result()
                    results.extend(user_results)
                except Exception as e:
                    logger.error(f"User session failed: {e}")
        
        total_time = time.time() - start_time
        system_metrics = monitor.stop_monitoring()
        
        # Calculate statistics
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        response_times = [r['duration'] for r in successful_requests]
        
        return {
            'test_type': 'concurrent_load',
            'endpoint': endpoint,
            'method': method,
            'concurrent_users': concurrent_users,
            'requests_per_user': requests_per_user,
            'total_requests': len(results),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / len(results) * 100 if results else 0,
            'total_duration': total_time,
            'requests_per_second': len(results) / total_time if total_time > 0 else 0,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'p95_response_time': np.percentile(response_times, 95) if response_times else 0,
            'p99_response_time': np.percentile(response_times, 99) if response_times else 0,
            'system_metrics': system_metrics
        }
    
    def escalating_load_test(self, endpoint: str, method: str = "GET", 
                           max_users: int = 50, step_size: int = 5, 
                           step_duration: int = 30) -> Dict[str, Any]:
        """Test with escalating load to find breaking point"""
        logger.info(f"Running escalating load test up to {max_users} users")
        
        results = []
        current_users = step_size
        
        while current_users <= max_users:
            logger.info(f"Testing with {current_users} concurrent users...")
            
            step_result = self.concurrent_load_test(
                endpoint, method, current_users, 
                requests_per_user=max(1, step_duration // 2)  # Adjust requests based on duration
            )
            
            step_result['user_count'] = current_users
            results.append(step_result)
            
            # Stop if success rate drops below 80%
            if step_result['success_rate'] < 80:
                logger.warning(f"Success rate dropped to {step_result['success_rate']:.1f}% at {current_users} users")
                break
            
            current_users += step_size
            time.sleep(5)  # Brief pause between steps
        
        # Find breaking point
        breaking_point = None
        for result in results:
            if result['success_rate'] < 95:
                breaking_point = result['user_count']
                break
        
        return {
            'test_type': 'escalating_load',
            'endpoint': endpoint,
            'method': method,
            'max_users_tested': current_users - step_size,
            'breaking_point_users': breaking_point,
            'step_results': results,
            'peak_rps': max(r['requests_per_second'] for r in results),
            'peak_success_rate': max(r['success_rate'] for r in results)
        }
    
    def sustained_load_test(self, endpoint: str, method: str = "GET",
                          concurrent_users: int = 20, duration_minutes: int = 10) -> Dict[str, Any]:
        """Test sustained load over time"""
        logger.info(f"Running sustained load test: {concurrent_users} users for {duration_minutes} minutes")
        
        end_time = time.time() + (duration_minutes * 60)
        monitor = PerformanceMonitor(interval=1.0)  # 1 second intervals for sustained test
        monitor.start_monitoring()
        
        results = []
        
        def continuous_requests():
            """Make continuous requests until end time"""
            session_results = []
            while time.time() < end_time:
                try:
                    start_time = time.time()
                    
                    if method.upper() == "GET":
                        response = self.session.get(f"{self.base_url}{endpoint}")
                    elif method.upper() == "POST":
                        response = self.session.post(f"{self.base_url}{endpoint}")
                    
                    duration = time.time() - start_time
                    
                    session_results.append({
                        'timestamp': start_time,
                        'success': response.status_code == 200,
                        'duration': duration,
                        'status_code': response.status_code
                    })
                    
                    # Small delay to prevent overwhelming
                    time.sleep(0.1)
                    
                except Exception as e:
                    session_results.append({
                        'timestamp': time.time(),
                        'success': False,
                        'duration': 0,
                        'error': str(e)
                    })
            
            return session_results
        
        # Start concurrent sustained load
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(continuous_requests) for _ in range(concurrent_users)]
            
            for future in as_completed(futures):
                try:
                    user_results = future.result()
                    results.extend(user_results)
                except Exception as e:
                    logger.error(f"Sustained test user failed: {e}")
        
        system_metrics = monitor.stop_monitoring()
        
        # Analyze results over time
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        response_times = [r['duration'] for r in successful_requests]
        
        # Calculate performance over time intervals
        interval_stats = []
        start_timestamp = min(r['timestamp'] for r in results) if results else time.time()
        
        for i in range(duration_minutes):
            interval_start = start_timestamp + (i * 60)
            interval_end = interval_start + 60
            
            interval_requests = [
                r for r in results 
                if interval_start <= r['timestamp'] < interval_end
            ]
            
            if interval_requests:
                interval_successful = [r for r in interval_requests if r['success']]
                interval_times = [r['duration'] for r in interval_successful]
                
                interval_stats.append({
                    'minute': i + 1,
                    'total_requests': len(interval_requests),
                    'successful_requests': len(interval_successful),
                    'success_rate': len(interval_successful) / len(interval_requests) * 100,
                    'avg_response_time': statistics.mean(interval_times) if interval_times else 0,
                    'requests_per_second': len(interval_requests) / 60
                })
        
        return {
            'test_type': 'sustained_load',
            'endpoint': endpoint,
            'method': method,
            'concurrent_users': concurrent_users,
            'duration_minutes': duration_minutes,
            'total_requests': len(results),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'overall_success_rate': len(successful_requests) / len(results) * 100 if results else 0,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'p95_response_time': np.percentile(response_times, 95) if response_times else 0,
            'p99_response_time': np.percentile(response_times, 99) if response_times else 0,
            'avg_requests_per_second': len(results) / (duration_minutes * 60),
            'interval_stats': interval_stats,
            'system_metrics': system_metrics
        }
    
    def memory_stress_test(self) -> Dict[str, Any]:
        """Test memory-intensive operations"""
        logger.info("Running memory stress test...")
        
        monitor = PerformanceMonitor(interval=0.5)
        monitor.start_monitoring()
        
        # Test memory optimization under load
        results = []
        
        for i in range(10):  # 10 iterations
            try:
                start_time = time.time()
                response = self.session.post(f"{self.base_url}/optimize/memory")
                duration = time.time() - start_time
                
                results.append({
                    'iteration': i + 1,
                    'success': response.status_code == 200,
                    'duration': duration,
                    'response_data': response.json() if response.status_code == 200 else None
                })
                
                # Brief pause
                time.sleep(2)
                
            except Exception as e:
                results.append({
                    'iteration': i + 1,
                    'success': False,
                    'error': str(e)
                })
        
        system_metrics = monitor.stop_monitoring()
        
        successful_results = [r for r in results if r['success']]
        
        return {
            'test_type': 'memory_stress',
            'iterations': len(results),
            'successful_iterations': len(successful_results),
            'success_rate': len(successful_results) / len(results) * 100,
            'avg_duration': statistics.mean([r['duration'] for r in successful_results]) if successful_results else 0,
            'system_metrics': system_metrics,
            'results': results
        }
    
    def storage_analysis_stress_test(self) -> Dict[str, Any]:
        """Test storage analysis under various conditions"""
        logger.info("Running storage analysis stress test...")
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Test different paths and parameters
        test_scenarios = [
            ("/tmp", {}),
            ("/var", {}),
            ("/", {}),
            ("/opt", {}),
            ("/tmp", {"path": "/tmp"}),  # Duplicate analysis
        ]
        
        results = []
        
        for i, (path, params) in enumerate(test_scenarios):
            logger.info(f"Testing storage analysis scenario {i+1}/{len(test_scenarios)}: {path}")
            
            # Test storage analysis
            start_time = time.time()
            try:
                response = self.session.get(f"{self.base_url}/analyze/storage", 
                                          params={"path": path, **params})
                duration = time.time() - start_time
                
                result = {
                    'scenario': i + 1,
                    'path': path,
                    'params': params,
                    'success': response.status_code == 200,
                    'duration': duration,
                    'status_code': response.status_code
                }
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        result.update({
                            'total_files': data.get('total_files', 0),
                            'total_size_mb': data.get('total_size_mb', 0)
                        })
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'scenario': i + 1,
                    'path': path,
                    'params': params,
                    'success': False,
                    'error': str(e)
                })
            
            time.sleep(1)  # Brief pause between scenarios
        
        system_metrics = monitor.stop_monitoring()
        
        successful_results = [r for r in results if r['success']]
        
        return {
            'test_type': 'storage_analysis_stress',
            'scenarios_tested': len(results),
            'successful_scenarios': len(successful_results),
            'success_rate': len(successful_results) / len(results) * 100,
            'avg_duration': statistics.mean([r['duration'] for r in successful_results]) if successful_results else 0,
            'max_duration': max([r['duration'] for r in successful_results]) if successful_results else 0,
            'system_metrics': system_metrics,
            'scenario_results': results
        }

class PerformanceBenchmarkSuite:
    """Complete performance benchmark suite"""
    
    def __init__(self, base_url: str = "http://localhost:8116"):
        self.base_url = base_url
        self.stress_tester = StressTestScenarios(base_url)
        self.results = {}
    
    def run_single_endpoint_benchmarks(self) -> Dict[str, Any]:
        """Benchmark all endpoints individually"""
        logger.info("Running single endpoint benchmarks...")
        
        endpoints = [
            ("GET", "/health"),
            ("GET", "/status"),
            ("POST", "/optimize/memory"),
            ("POST", "/optimize/cpu"),
            ("POST", "/optimize/disk"),
            ("GET", "/analyze/storage", {"path": "/tmp"}),
            ("GET", "/analyze/storage/report"),
            ("POST", "/optimize/storage", {"dry_run": True})
        ]
        
        benchmarks = {}
        
        for method, endpoint, *params in endpoints:
            logger.info(f"Benchmarking {method} {endpoint}...")
            params_dict = params[0] if params else None
            
            benchmark = self.stress_tester.single_request_benchmark(method, endpoint, params_dict)
            benchmarks[f"{method} {endpoint}"] = benchmark
            
            time.sleep(2)  # Brief pause between benchmarks
        
        return benchmarks
    
    def run_load_tests(self) -> Dict[str, Any]:
        """Run various load tests"""
        logger.info("Running load tests...")
        
        load_tests = {}
        
        # Light load test
        load_tests['light_load'] = self.stress_tester.concurrent_load_test(
            "/health", "GET", concurrent_users=5, requests_per_user=10
        )
        
        # Medium load test
        load_tests['medium_load'] = self.stress_tester.concurrent_load_test(
            "/status", "GET", concurrent_users=15, requests_per_user=8
        )
        
        # Heavy load test
        load_tests['heavy_load'] = self.stress_tester.concurrent_load_test(
            "/analyze/storage", "GET", concurrent_users=10, requests_per_user=5,
            params={"path": "/tmp"}
        )
        
        return load_tests
    
    def run_stress_tests(self) -> Dict[str, Any]:
        """Run stress tests"""
        logger.info("Running stress tests...")
        
        stress_tests = {}
        
        # Escalating load test
        stress_tests['escalating_load'] = self.stress_tester.escalating_load_test(
            "/health", "GET", max_users=30, step_size=5
        )
        
        # Sustained load test
        stress_tests['sustained_load'] = self.stress_tester.sustained_load_test(
            "/status", "GET", concurrent_users=10, duration_minutes=5
        )
        
        # Memory stress test
        stress_tests['memory_stress'] = self.stress_tester.memory_stress_test()
        
        # Storage analysis stress test
        stress_tests['storage_stress'] = self.stress_tester.storage_analysis_stress_test()
        
        return stress_tests
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'agent_url': self.base_url,
            'benchmarks': self.results.get('benchmarks', {}),
            'load_tests': self.results.get('load_tests', {}),
            'stress_tests': self.results.get('stress_tests', {}),
            'summary': self._generate_summary()
        }
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        summary = {}
        
        # Benchmark summary
        if 'benchmarks' in self.results:
            response_times = []
            success_rates = []
            
            for endpoint, data in self.results['benchmarks'].items():
                if 'avg_response_time' in data:
                    response_times.append(data['avg_response_time'])
                    
                if 'successful_requests' in data and 'failed_requests' in data:
                    total = data['successful_requests'] + data['failed_requests']
                    success_rate = data['successful_requests'] / total * 100 if total > 0 else 0
                    success_rates.append(success_rate)
            
            if response_times:
                summary['benchmark_summary'] = {
                    'avg_response_time': statistics.mean(response_times),
                    'max_response_time': max(response_times),
                    'min_response_time': min(response_times),
                    'avg_success_rate': statistics.mean(success_rates) if success_rates else 0
                }
        
        # Load test summary
        if 'load_tests' in self.results:
            load_summary = {}
            for test_name, data in self.results['load_tests'].items():
                if 'success_rate' in data and 'requests_per_second' in data:
                    load_summary[test_name] = {
                        'success_rate': data['success_rate'],
                        'requests_per_second': data['requests_per_second'],
                        'avg_response_time': data.get('avg_response_time', 0)
                    }
            
            summary['load_test_summary'] = load_summary
        
        # Stress test summary
        if 'stress_tests' in self.results:
            stress_summary = {}
            for test_name, data in self.results['stress_tests'].items():
                if test_name == 'escalating_load':
                    stress_summary[test_name] = {
                        'breaking_point_users': data.get('breaking_point_users'),
                        'peak_rps': data.get('peak_rps', 0)
                    }
                elif 'success_rate' in data:
                    stress_summary[test_name] = {
                        'success_rate': data['success_rate']
                    }
            
            summary['stress_test_summary'] = stress_summary
        
        return summary
    
    def save_report_to_file(self, report: Dict[str, Any], filename: str = None):
        """Save performance report to file"""
        if not filename:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to: {filename}")
        return filename
    
    def run_comprehensive_performance_suite(self) -> Dict[str, Any]:
        """Run the complete performance test suite"""
        logger.info("Starting comprehensive performance test suite...")
        start_time = time.time()
        
        try:
            # Run benchmarks
            self.results['benchmarks'] = self.run_single_endpoint_benchmarks()
            
            # Run load tests
            self.results['load_tests'] = self.run_load_tests()
            
            # Run stress tests
            self.results['stress_tests'] = self.run_stress_tests()
            
            # Generate report
            report = self.generate_performance_report()
            report['total_test_duration'] = time.time() - start_time
            
            # Save report
            filename = self.save_report_to_file(report)
            report['report_file'] = filename
            
            logger.info(f"Performance test suite completed in {report['total_test_duration']:.2f} seconds")
            return report
            
        except Exception as e:
            logger.error(f"Performance test suite failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'partial_results': self.results
            }

def main():
    """Main entry point for performance testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hardware Optimizer Performance Testing")
    parser.add_argument("--url", default="http://localhost:8116", help="Agent URL")
    parser.add_argument("--test", choices=["benchmark", "load", "stress", "all"], 
                       default="all", help="Test type to run")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    suite = PerformanceBenchmarkSuite(args.url)
    
    if args.test == "benchmark":
        results = suite.run_single_endpoint_benchmarks()
    elif args.test == "load":
        results = suite.run_load_tests()
    elif args.test == "stress":
        results = suite.run_stress_tests()
    else:  # all
        results = suite.run_comprehensive_performance_suite()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")
    else:
        logger.info(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()