#!/usr/bin/env python3
"""
Comprehensive Performance Testing for MCP Monitoring System
Tests load handling, response times, resource usage, and system limits
"""

import asyncio
import aiohttp
import time
import json
import statistics
import psutil
import sys
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/opt/sutazaiapp/scripts/mcp/automation/tests/performance_test_results.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance test metrics"""
    endpoint: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    min_response_time: float
    max_response_time: float
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    total_duration: float
    error_rate: float
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class MonitoringPerformanceTest:
    """Comprehensive performance testing for MCP Monitoring System"""
    
    def __init__(self, base_url: str = "http://localhost:10250"):
        self.base_url = base_url
        self.endpoints = {
            'health': '/health',
            'health_detailed': '/health/detailed',
            'metrics': '/metrics',
            'dashboard': '/',
        }
        self.results: List[PerformanceMetrics] = []
        self.resource_metrics = []
        
    async def test_endpoint_response_time(self, endpoint: str, session: aiohttp.ClientSession) -> float:
        """Test single endpoint response time"""
        start_time = time.perf_counter()
        try:
            async with session.get(f"{self.base_url}{endpoint}") as response:
                await response.text()
                response_time = time.perf_counter() - start_time
                if response.status == 200:
                    return response_time
                else:
                    logger.warning(f"Non-200 status for {endpoint}: {response.status}")
                    return -1
        except Exception as e:
            logger.error(f"Error testing {endpoint}: {e}")
            return -1
    
    async def load_test_endpoint(self, endpoint: str, concurrent_requests: int, 
                                 duration_seconds: int = 10) -> PerformanceMetrics:
        """Perform load test on specific endpoint"""
        logger.info(f"Starting load test for {endpoint} with {concurrent_requests} concurrent requests for {duration_seconds}s")
        
        response_times = []
        errors = 0
        start_time = time.time()
        total_requests = 0
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < duration_seconds:
                tasks = []
                for _ in range(concurrent_requests):
                    tasks.append(self.test_endpoint_response_time(endpoint, session))
                
                results = await asyncio.gather(*tasks)
                
                for response_time in results:
                    total_requests += 1
                    if response_time == -1:
                        errors += 1
                    else:
                        response_times.append(response_time)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
        
        total_duration = time.time() - start_time
        
        if response_times:
            response_times.sort()
            metrics = PerformanceMetrics(
                endpoint=endpoint,
                total_requests=total_requests,
                successful_requests=len(response_times),
                failed_requests=errors,
                min_response_time=min(response_times) * 1000,  # Convert to ms
                max_response_time=max(response_times) * 1000,
                avg_response_time=statistics.mean(response_times) * 1000,
                p50_response_time=response_times[int(len(response_times) * 0.50)] * 1000,
                p95_response_time=response_times[int(len(response_times) * 0.95)] * 1000,
                p99_response_time=response_times[int(len(response_times) * 0.99)] * 1000,
                requests_per_second=total_requests / total_duration,
                total_duration=total_duration,
                error_rate=(errors / total_requests) * 100 if total_requests > 0 else 0
            )
        else:
            metrics = PerformanceMetrics(
                endpoint=endpoint,
                total_requests=total_requests,
                successful_requests=0,
                failed_requests=errors,
                min_response_time=0,
                max_response_time=0,
                avg_response_time=0,
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                total_duration=total_duration,
                error_rate=100
            )
        
        self.results.append(metrics)
        return metrics
    
    def monitor_system_resources(self) -> Dict[str, float]:
        """Monitor system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Try to find the monitoring process
            monitoring_process = None
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and 'monitoring_server.py' in ' '.join(cmdline):
                        monitoring_process = psutil.Process(proc.info['pid'])
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            resources = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
            }
            
            if monitoring_process:
                try:
                    resources['monitoring_cpu_percent'] = monitoring_process.cpu_percent()
                    resources['monitoring_memory_mb'] = monitoring_process.memory_info().rss / (1024**2)
                    resources['monitoring_threads'] = monitoring_process.num_threads()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            return resources
        except Exception as e:
            logger.error(f"Error monitoring resources: {e}")
            return {}
    
    async def stress_test(self, max_concurrent: int = 1000, step: int = 100) -> List[PerformanceMetrics]:
        """Gradually increase load to find breaking point"""
        logger.info(f"Starting stress test up to {max_concurrent} concurrent requests")
        stress_results = []
        
        for concurrent in range(step, max_concurrent + 1, step):
            logger.info(f"Testing with {concurrent} concurrent requests")
            
            # Test each endpoint
            for endpoint_name, endpoint_path in self.endpoints.items():
                # Monitor resources before test
                resources_before = self.monitor_system_resources()
                
                # Run load test
                metrics = await self.load_test_endpoint(endpoint_path, concurrent, duration_seconds=5)
                
                # Monitor resources after test
                resources_after = self.monitor_system_resources()
                
                # Log performance
                logger.info(f"  {endpoint_name}: {metrics.requests_per_second:.2f} req/s, "
                          f"p95: {metrics.p95_response_time:.2f}ms, "
                          f"errors: {metrics.error_rate:.2f}%")
                
                # Store resource metrics
                self.resource_metrics.append({
                    'concurrent_requests': concurrent,
                    'endpoint': endpoint_path,
                    'before': resources_before,
                    'after': resources_after
                })
                
                # Stop if error rate is too high or response time exceeds threshold
                if metrics.error_rate > 10 or metrics.p95_response_time > 5000:  # 5 seconds
                    logger.warning(f"Performance degradation detected at {concurrent} concurrent requests")
                    stress_results.append(metrics)
                    return stress_results
            
            stress_results.extend(self.results[-len(self.endpoints):])
            
            # Small delay between test levels
            await asyncio.sleep(2)
        
        return stress_results
    
    async def baseline_performance_test(self) -> Dict[str, PerformanceMetrics]:
        """Establish baseline performance metrics"""
        logger.info("Establishing baseline performance metrics")
        baseline = {}
        
        for endpoint_name, endpoint_path in self.endpoints.items():
            # Test with low concurrent load (10 users)
            metrics = await self.load_test_endpoint(endpoint_path, 10, duration_seconds=10)
            baseline[endpoint_name] = metrics
            
            logger.info(f"Baseline for {endpoint_name}:")
            logger.info(f"  Avg response time: {metrics.avg_response_time:.2f}ms")
            logger.info(f"  P95 response time: {metrics.p95_response_time:.2f}ms")
            logger.info(f"  Throughput: {metrics.requests_per_second:.2f} req/s")
        
        return baseline
    
    async def sustained_load_test(self, concurrent: int = 100, duration_minutes: int = 2) -> List[PerformanceMetrics]:
        """Test sustained load over extended period"""
        logger.info(f"Starting sustained load test: {concurrent} concurrent for {duration_minutes} minutes")
        sustained_results = []
        
        duration_seconds = duration_minutes * 60
        interval_seconds = 30  # Check performance every 30 seconds
        
        for i in range(0, duration_seconds, interval_seconds):
            logger.info(f"Sustained test progress: {i}/{duration_seconds} seconds")
            
            # Test all endpoints
            for endpoint_name, endpoint_path in self.endpoints.items():
                metrics = await self.load_test_endpoint(endpoint_path, concurrent, duration_seconds=interval_seconds)
                sustained_results.append(metrics)
                
                # Check for performance degradation
                if metrics.error_rate > 5:
                    logger.warning(f"High error rate detected: {metrics.error_rate:.2f}%")
        
        return sustained_results
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_tests': len(self.results),
                'endpoints_tested': list(self.endpoints.keys()),
            },
            'baseline_metrics': {},
            'load_test_results': [],
            'resource_usage': self.resource_metrics,
            'recommendations': []
        }
        
        # Analyze results by endpoint
        endpoint_metrics = {}
        for metric in self.results:
            if metric.endpoint not in endpoint_metrics:
                endpoint_metrics[metric.endpoint] = []
            endpoint_metrics[metric.endpoint].append(metric)
        
        # Calculate summary statistics for each endpoint
        for endpoint, metrics_list in endpoint_metrics.items():
            avg_response_times = [m.avg_response_time for m in metrics_list]
            p95_response_times = [m.p95_response_time for m in metrics_list]
            error_rates = [m.error_rate for m in metrics_list]
            throughputs = [m.requests_per_second for m in metrics_list]
            
            endpoint_summary = {
                'endpoint': endpoint,
                'test_count': len(metrics_list),
                'avg_response_time': {
                    'min': min(avg_response_times),
                    'max': max(avg_response_times),
                    'mean': statistics.mean(avg_response_times)
                },
                'p95_response_time': {
                    'min': min(p95_response_times),
                    'max': max(p95_response_times),
                    'mean': statistics.mean(p95_response_times)
                },
                'throughput': {
                    'min': min(throughputs),
                    'max': max(throughputs),
                    'mean': statistics.mean(throughputs)
                },
                'error_rate': {
                    'min': min(error_rates),
                    'max': max(error_rates),
                    'mean': statistics.mean(error_rates)
                }
            }
            
            report['load_test_results'].append(endpoint_summary)
            
            # Add recommendations based on performance
            if endpoint_summary['p95_response_time']['mean'] > 100:
                report['recommendations'].append(
                    f"Consider optimizing {endpoint} - P95 response time ({endpoint_summary['p95_response_time']['mean']:.2f}ms) exceeds 100ms target"
                )
            
            if endpoint_summary['error_rate']['max'] > 1:
                report['recommendations'].append(
                    f"Investigate error handling for {endpoint} - max error rate {endpoint_summary['error_rate']['max']:.2f}%"
                )
            
            if endpoint_summary['throughput']['mean'] < 1000:
                report['recommendations'].append(
                    f"Consider scaling {endpoint} - throughput ({endpoint_summary['throughput']['mean']:.2f} req/s) below 1000 req/s target"
                )
        
        # Performance requirements validation
        report['requirements_validation'] = {
            'api_response_time_p95_under_100ms': all(m.p95_response_time < 100 for m in self.results),
            'throughput_over_1000_rps': any(m.requests_per_second > 1000 for m in self.results),
            'error_rate_under_1_percent': all(m.error_rate < 1 for m in self.results),
            'supports_100_concurrent': any(m.total_requests > 100 and m.error_rate < 5 for m in self.results)
        }
        
        return report
    
    def save_results(self, filename: str = None):
        """Save test results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/opt/sutazaiapp/scripts/mcp/automation/tests/performance_report_{timestamp}.json"
        
        report = self.generate_performance_report()
        
        # Convert dataclass instances to dict for JSON serialization
        report['raw_results'] = [asdict(r) for r in self.results]
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {filename}")
        return filename
    
    def print_summary(self):
        """Print performance test summary"""
        report = self.generate_performance_report()
        
        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE TEST SUMMARY")
        logger.info("="*80)
        
        logger.info(f"\nTest completed at: {report['timestamp']}")
        logger.info(f"Total tests run: {report['summary']['total_tests']}")
        
        logger.info("\n" + "-"*40)
        logger.info("ENDPOINT PERFORMANCE METRICS")
        logger.info("-"*40)
        
        for result in report['load_test_results']:
            logger.info(f"\nEndpoint: {result['endpoint']}")
            logger.info(f"  Response Time (avg): {result['avg_response_time']['mean']:.2f}ms")
            logger.info(f"  Response Time (P95): {result['p95_response_time']['mean']:.2f}ms")
            logger.info(f"  Throughput: {result['throughput']['mean']:.2f} req/s")
            logger.error(f"  Error Rate: {result['error_rate']['mean']:.2f}%")
        
        logger.info("\n" + "-"*40)
        logger.info("REQUIREMENTS VALIDATION")
        logger.info("-"*40)
        
        for requirement, passed in report['requirements_validation'].items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {requirement}: {status}")
        
        if report['recommendations']:
            logger.info("\n" + "-"*40)
            logger.info("RECOMMENDATIONS")
            logger.info("-"*40)
            for rec in report['recommendations']:
                logger.info(f"  - {rec}")
        
        logger.info("\n" + "="*80)


async def main():
    """Main test execution"""
    tester = MonitoringPerformanceTest()
    
    try:
        # 1. Baseline Performance Test
        logger.info("\n[1/5] Running baseline performance tests...")
        baseline = await tester.baseline_performance_test()
        
        # 2. Load Testing with different concurrent levels
        logger.info("\n[2/5] Running load tests with increasing concurrency...")
        load_levels = [10, 50, 100, 500]
        for level in load_levels:
            logger.info(f"  Testing with {level} concurrent requests...")
            for endpoint_name, endpoint_path in tester.endpoints.items():
                await tester.load_test_endpoint(endpoint_path, level, duration_seconds=10)
        
        # 3. Stress Testing to find breaking point
        logger.info("\n[3/5] Running stress test to find system limits...")
        stress_results = await tester.stress_test(max_concurrent=500, step=50)
        
        # 4. Sustained Load Test (reduced duration for demo)
        logger.info("\n[4/5] Running sustained load test...")
        sustained_results = await tester.sustained_load_test(concurrent=100, duration_minutes=1)
        
        # 5. Generate and save report
        logger.info("\n[5/5] Generating performance report...")
        report_file = tester.save_results()
        
        # Print summary
        tester.print_summary()
        
        logger.info(f"\nDetailed report saved to: {report_file}")
        
        # Return exit code based on requirements validation
        report = tester.generate_performance_report()
        all_passed = all(report['requirements_validation'].values())
        
        if all_passed:
            logger.info("\n✅ All performance requirements met!")
            return 0
        else:
            logger.info("\n⚠️ Some performance requirements not met. See recommendations above.")
            return 1
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)