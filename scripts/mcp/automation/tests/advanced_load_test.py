#!/usr/bin/env python3
"""
Advanced Load Testing Tool for MCP Monitoring System
Comprehensive performance analysis with detailed metrics and visualizations
"""

import asyncio
import aiohttp
import time
import json
import statistics
import sys
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import logging
import psutil
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    endpoint: str
    start_time: float
    end_time: float
    response_time_ms: float
    status_code: int
    success: bool
    error_message: str = ""
    
@dataclass
class LoadTestResult:
    """Results from a load test scenario"""
    scenario_name: str
    endpoint: str
    concurrent_users: int
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # Response time metrics (in milliseconds)
    min_response_time: float
    max_response_time: float
    mean_response_time: float
    median_response_time: float
    p50_response_time: float
    p75_response_time: float
    p90_response_time: float
    p95_response_time: float
    p99_response_time: float
    
    # Throughput metrics
    requests_per_second: float
    
    # Error metrics
    error_rate: float
    
    # Optional fields with defaults
    bytes_per_second: float = 0
    status_codes: Dict[int, int] = field(default_factory=dict)
    
    # Resource metrics
    avg_cpu_percent: float = 0
    avg_memory_mb: float = 0
    peak_cpu_percent: float = 0
    peak_memory_mb: float = 0
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class SystemMonitor:
    """Monitor system resources during tests"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        
    def start(self):
        """Start monitoring system resources"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop monitoring and return metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        return self.metrics
        
    def _monitor_loop(self):
        """Monitor loop that runs in background thread"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                metric = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory.used / (1024 * 1024),
                    'memory_percent': memory.percent
                }
                
                # Try to find monitoring process
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = proc.info.get('cmdline', [])
                        if cmdline and 'monitoring_server.py' in ' '.join(cmdline):
                            p = psutil.Process(proc.info['pid'])
                            metric['process_cpu'] = p.cpu_percent()
                            metric['process_memory_mb'] = p.memory_info().rss / (1024 * 1024)
                            break
                    except:
                        continue
                        
                self.metrics.append(metric)
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")

class AdvancedLoadTester:
    """Advanced load testing with detailed metrics"""
    
    def __init__(self, base_url: str = "http://localhost:10250"):
        self.base_url = base_url
        self.results: List[LoadTestResult] = []
        self.system_monitor = SystemMonitor()
        
    async def make_request(self, session: aiohttp.ClientSession, endpoint: str) -> RequestMetrics:
        """Make a single request and capture metrics"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.perf_counter()
        
        try:
            async with session.get(url) as response:
                content = await response.read()
                end_time = time.perf_counter()
                
                return RequestMetrics(
                    endpoint=endpoint,
                    start_time=start_time,
                    end_time=end_time,
                    response_time_ms=(end_time - start_time) * 1000,
                    status_code=response.status,
                    success=(response.status == 200),
                    error_message=""
                )
        except Exception as e:
            end_time = time.perf_counter()
            return RequestMetrics(
                endpoint=endpoint,
                start_time=start_time,
                end_time=end_time,
                response_time_ms=(end_time - start_time) * 1000,
                status_code=0,
                success=False,
                error_message=str(e)
            )
    
    async def run_load_test(self, 
                          scenario_name: str,
                          endpoint: str, 
                          concurrent_users: int,
                          duration_seconds: int,
                          ramp_up_seconds: int = 0) -> LoadTestResult:
        """Run a load test scenario"""
        
        logger.info(f"Starting scenario: {scenario_name}")
        logger.info(f"  Endpoint: {endpoint}")
        logger.info(f"  Concurrent users: {concurrent_users}")
        logger.info(f"  Duration: {duration_seconds}s")
        
        # Start system monitoring
        self.system_monitor.start()
        
        all_metrics: List[RequestMetrics] = []
        start_time = time.time()
        
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(limit=concurrent_users, limit_per_host=concurrent_users)
        async with aiohttp.ClientSession(connector=connector) as session:
            
            # Ramp-up phase
            if ramp_up_seconds > 0:
                logger.info(f"  Ramping up over {ramp_up_seconds}s...")
                ramp_up_steps = min(10, concurrent_users)
                step_duration = ramp_up_seconds / ramp_up_steps
                users_per_step = concurrent_users // ramp_up_steps
                
                for step in range(ramp_up_steps):
                    current_users = users_per_step * (step + 1)
                    tasks = [self.make_request(session, endpoint) for _ in range(current_users)]
                    results = await asyncio.gather(*tasks)
                    all_metrics.extend(results)
                    await asyncio.sleep(step_duration)
            
            # Main test phase
            end_time = start_time + duration_seconds
            request_count = 0
            
            while time.time() < end_time:
                # Launch concurrent requests
                tasks = []
                for _ in range(concurrent_users):
                    tasks.append(self.make_request(session, endpoint))
                
                # Wait for all requests to complete
                batch_results = await asyncio.gather(*tasks)
                all_metrics.extend(batch_results)
                request_count += len(batch_results)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
                
                # Progress indicator
                if request_count % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = request_count / elapsed if elapsed > 0 else 0
                    logger.info(f"    Progress: {request_count} requests, {rate:.1f} req/s")
        
        # Stop monitoring
        system_metrics = self.system_monitor.stop()
        
        # Calculate statistics
        actual_duration = time.time() - start_time
        response_times = [m.response_time_ms for m in all_metrics if m.success]
        
        if response_times:
            response_times.sort()
            
            # Calculate percentiles
            def percentile(data, p):
                idx = int(len(data) * p / 100)
                return data[min(idx, len(data)-1)]
            
            # Count status codes
            status_codes = defaultdict(int)
            for m in all_metrics:
                status_codes[m.status_code] += 1
            
            # Calculate resource metrics
            cpu_values = [m['cpu_percent'] for m in system_metrics if 'cpu_percent' in m]
            memory_values = [m['memory_mb'] for m in system_metrics if 'memory_mb' in m]
            
            result = LoadTestResult(
                scenario_name=scenario_name,
                endpoint=endpoint,
                concurrent_users=concurrent_users,
                duration_seconds=actual_duration,
                total_requests=len(all_metrics),
                successful_requests=sum(1 for m in all_metrics if m.success),
                failed_requests=sum(1 for m in all_metrics if not m.success),
                min_response_time=min(response_times),
                max_response_time=max(response_times),
                mean_response_time=statistics.mean(response_times),
                median_response_time=statistics.median(response_times),
                p50_response_time=percentile(response_times, 50),
                p75_response_time=percentile(response_times, 75),
                p90_response_time=percentile(response_times, 90),
                p95_response_time=percentile(response_times, 95),
                p99_response_time=percentile(response_times, 99),
                requests_per_second=len(all_metrics) / actual_duration,
                error_rate=(sum(1 for m in all_metrics if not m.success) / len(all_metrics) * 100),
                status_codes=dict(status_codes),
                avg_cpu_percent=statistics.mean(cpu_values) if cpu_values else 0,
                avg_memory_mb=statistics.mean(memory_values) if memory_values else 0,
                peak_cpu_percent=max(cpu_values) if cpu_values else 0,
                peak_memory_mb=max(memory_values) if memory_values else 0
            )
        else:
            # All requests failed
            result = LoadTestResult(
                scenario_name=scenario_name,
                endpoint=endpoint,
                concurrent_users=concurrent_users,
                duration_seconds=actual_duration,
                total_requests=len(all_metrics),
                successful_requests=0,
                failed_requests=len(all_metrics),
                min_response_time=0,
                max_response_time=0,
                mean_response_time=0,
                median_response_time=0,
                p50_response_time=0,
                p75_response_time=0,
                p90_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=len(all_metrics) / actual_duration if actual_duration > 0 else 0,
                error_rate=100,
                status_codes={0: len(all_metrics)}
            )
        
        self.results.append(result)
        return result
    
    def print_result(self, result: LoadTestResult):
        """Print formatted test result"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Scenario: {result.scenario_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Endpoint: {result.endpoint}")
        logger.info(f"Concurrent Users: {result.concurrent_users}")
        logger.info(f"Duration: {result.duration_seconds:.1f}s")
        logger.info(f"\nRequests:")
        logger.info(f"  Total: {result.total_requests}")
        logger.info(f"  Successful: {result.successful_requests}")
        logger.error(f"  Failed: {result.failed_requests}")
        logger.error(f"  Error Rate: {result.error_rate:.2f}%")
        
        if result.successful_requests > 0:
            logger.info(f"\nResponse Times (ms):")
            logger.info(f"  Min: {result.min_response_time:.2f}")
            logger.info(f"  Mean: {result.mean_response_time:.2f}")
            logger.info(f"  Median: {result.median_response_time:.2f}")
            logger.info(f"  P75: {result.p75_response_time:.2f}")
            logger.info(f"  P90: {result.p90_response_time:.2f}")
            logger.info(f"  P95: {result.p95_response_time:.2f}")
            logger.info(f"  P99: {result.p99_response_time:.2f}")
            logger.info(f"  Max: {result.max_response_time:.2f}")
        
        logger.info(f"\nThroughput:")
        logger.info(f"  Requests/sec: {result.requests_per_second:.2f}")
        
        if result.avg_cpu_percent > 0:
            logger.info(f"\nSystem Resources:")
            logger.info(f"  Avg CPU: {result.avg_cpu_percent:.1f}%")
            logger.info(f"  Peak CPU: {result.peak_cpu_percent:.1f}%")
            logger.info(f"  Avg Memory: {result.avg_memory_mb:.1f} MB")
            logger.info(f"  Peak Memory: {result.peak_memory_mb:.1f} MB")
        
        logger.info(f"\nStatus Codes:")
        for code, count in sorted(result.status_codes.items()):
            logger.info(f"  {code}: {count}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'base_url': self.base_url,
            'total_scenarios': len(self.results),
            'scenarios': [],
            'performance_summary': {},
            'requirements_validation': {},
            'recommendations': []
        }
        
        # Add scenario results
        for result in self.results:
            report['scenarios'].append(asdict(result))
        
        # Performance summary
        if self.results:
            best_throughput = max(r.requests_per_second for r in self.results)
            avg_error_rate = statistics.mean(r.error_rate for r in self.results)
            
            report['performance_summary'] = {
                'best_throughput_rps': best_throughput,
                'average_error_rate': avg_error_rate,
                'scenarios_tested': len(self.results),
                'total_requests': sum(r.total_requests for r in self.results)
            }
            
            # Requirements validation
            report['requirements_validation'] = {
                'response_time_under_100ms_p95': any(r.p95_response_time < 100 for r in self.results),
                'throughput_over_1000_rps': best_throughput > 1000,
                'supports_100_concurrent_users': any(
                    r.concurrent_users >= 100 and r.error_rate < 5 
                    for r in self.results
                ),
                'error_rate_under_1_percent': all(r.error_rate < 1 for r in self.results),
                'cpu_usage_under_80_percent': all(
                    r.peak_cpu_percent < 80 for r in self.results 
                    if r.peak_cpu_percent > 0
                )
            }
            
            # Generate recommendations
            for result in self.results:
                if result.p95_response_time > 100:
                    report['recommendations'].append(
                        f"Optimize {result.endpoint} - P95 response time "
                        f"({result.p95_response_time:.1f}ms) exceeds 100ms target"
                    )
                
                if result.error_rate > 1:
                    report['recommendations'].append(
                        f"Investigate errors on {result.endpoint} - "
                        f"error rate {result.error_rate:.1f}%"
                    )
        
        return report
    
    def save_report(self, filename: str = None) -> str:
        """Save report to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/opt/sutazaiapp/scripts/mcp/automation/tests/advanced_load_report_{timestamp}.json"
        
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to: {filename}")
        return filename

async def main():
    """Main test execution"""
    tester = AdvancedLoadTester()
    
    logger.info("\n" + "="*80)
    logger.info("ADVANCED LOAD TESTING - MCP MONITORING SYSTEM")
    logger.info("="*80)
    
    # Test scenarios
    scenarios = [
        # Baseline tests
        ("Baseline - Light Load", "/health", 10, 10, 0),
        ("Baseline - Light Load", "/metrics", 10, 10, 0),
        
        # Normal load tests
        ("Normal Load", "/health", 50, 15, 2),
        ("Normal Load", "/metrics", 50, 15, 2),
        
        # High load tests
        ("High Load", "/health", 100, 20, 3),
        ("High Load", "/metrics", 100, 20, 3),
        
        # Stress tests
        ("Stress Test", "/health", 500, 15, 5),
        
        # Spike test
        ("Spike Test", "/health", 1000, 10, 0),
    ]
    
    # Run all scenarios
    for scenario_name, endpoint, users, duration, ramp_up in scenarios:
        result = await tester.run_load_test(scenario_name, endpoint, users, duration, ramp_up)
        tester.print_result(result)
        
        # Small delay between tests
        await asyncio.sleep(2)
    
    # Generate final report
    report = tester.generate_report()
    report_file = tester.save_report()
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE TEST SUMMARY")
    logger.info("="*80)
    
    logger.info(f"\nPerformance Highlights:")
    logger.info(f"  Best Throughput: {report['performance_summary']['best_throughput_rps']:.2f} req/s")
    logger.error(f"  Average Error Rate: {report['performance_summary']['average_error_rate']:.2f}%")
    logger.info(f"  Total Requests: {report['performance_summary']['total_requests']}")
    
    logger.info(f"\nRequirements Validation:")
    for req, passed in report['requirements_validation'].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {req}: {status}")
    
    if report['recommendations']:
        logger.info(f"\nRecommendations:")
        for rec in report['recommendations']:
            logger.info(f"  • {rec}")
    
    logger.info(f"\n📊 Detailed report saved to: {report_file}")
    logger.info("="*80)
    
    # Return success if all critical requirements pass
    critical_requirements = [
        report['requirements_validation'].get('response_time_under_100ms_p95', False),
        report['requirements_validation'].get('throughput_over_1000_rps', False),
        report['requirements_validation'].get('supports_100_concurrent_users', False)
    ]
    
    return 0 if all(critical_requirements) else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)