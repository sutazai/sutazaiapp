#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
ULTRAPERFORMANCE Load Testing Suite
Complete performance testing with k6-style metrics
"""

import asyncio
import aiohttp
import time
import statistics
import json
from datetime import datetime
from typing import Dict, List, Any
import random
import psutil
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


@dataclass
class LoadTestResult:
    """Load test result metrics"""
    endpoint: str
    duration_seconds: int
    virtual_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    error_rate_percent: float
    throughput_mbps: float
    cpu_usage_percent: float
    memory_usage_mb: float
    timestamp: str


class UltraPerformanceLoadTester:
    """ULTRAPERFORMANCE load testing engine"""
    
    def __init__(self):
        self.base_url = "http://localhost:10010"
        self.results = []
        self.response_times = []
        self.errors = []
        self.bytes_transferred = 0
        
    def test_endpoint_sync(self, url: str, method: str = "GET", 
                           payload: dict = None, headers: dict = None) -> dict:
        """Single synchronous request for load testing"""
        start_time = time.time()
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=payload, headers=headers, timeout=10)
            else:
                response = requests.request(method, url, json=payload, headers=headers, timeout=10)
                
            elapsed_ms = (time.time() - start_time) * 1000
            
            return {
                'success': response.status_code < 400,
                'status_code': response.status_code,
                'response_time_ms': elapsed_ms,
                'bytes': len(response.content)
            }
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return {
                'success': False,
                'status_code': 0,
                'response_time_ms': elapsed_ms,
                'bytes': 0,
                'error': str(e)
            }
            
    def ramp_up_load_test(self, endpoint: str, max_users: int = 100, 
                          ramp_duration: int = 30, test_duration: int = 60) -> LoadTestResult:
        """
        Ramp-up load test: gradually increase users
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"RAMP-UP LOAD TEST: {endpoint}")
        logger.info(f"Max Users: {max_users}, Ramp: {ramp_duration}s, Test: {test_duration}s")
        logger.info(f"{'='*60}")
        
        url = f"{self.base_url}{endpoint}"
        all_results = []
        start_time = time.time()
        
        # CPU/Memory baseline
        process = psutil.Process()
        cpu_samples = []
        memory_samples = []
        
        def worker(user_id: int, start_delay: float):
            """Worker function for each virtual user"""
            time.sleep(start_delay)  # Ramp-up delay
            
            user_results = []
            user_start = time.time()
            
            while time.time() - start_time < test_duration:
                result = self.test_endpoint_sync(url)
                user_results.append(result)
                
                # Think time between requests (10-100ms)
                time.sleep(random.uniform(0.01, 0.1))
                
            return user_results
            
        # Calculate ramp-up delays
        user_delays = []
        for i in range(max_users):
            delay = (i / max_users) * ramp_duration
            user_delays.append(delay)
            
        # Execute load test
        with ThreadPoolExecutor(max_workers=max_users) as executor:
            futures = []
            
            for i in range(max_users):
                future = executor.submit(worker, i, user_delays[i])
                futures.append(future)
                
            # Monitor while test runs
            monitor_interval = 1.0
            next_monitor = time.time() + monitor_interval
            
            while any(not f.done() for f in futures):
                if time.time() >= next_monitor:
                    cpu_samples.append(process.cpu_percent())
                    memory_samples.append(process.memory_info().rss / 1024 / 1024)
                    
                    # Print progress
                    elapsed = time.time() - start_time
                    active_users = sum(1 for f in futures if not f.done())
                    logger.info(f"  [{elapsed:5.1f}s] Active users: {active_users:3d}, "
                          f"CPU: {cpu_samples[-1]:5.1f}%, "
                          f"Memory: {memory_samples[-1]:6.0f}MB")
                    
                    next_monitor += monitor_interval
                    
                time.sleep(0.1)
                
            # Collect results
            for future in as_completed(futures):
                user_results = future.result()
                all_results.extend(user_results)
                
        # Calculate metrics
        return self._calculate_metrics(endpoint, test_duration, max_users, 
                                      all_results, cpu_samples, memory_samples)
                                      
    def spike_test(self, endpoint: str, normal_users: int = 10, 
                   spike_users: int = 100, spike_duration: int = 10) -> LoadTestResult:
        """
        Spike test: sudden increase in load
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"SPIKE TEST: {endpoint}")
        logger.info(f"Normal: {normal_users} users, Spike: {spike_users} users for {spike_duration}s")
        logger.info(f"{'='*60}")
        
        url = f"{self.base_url}{endpoint}"
        all_results = []
        
        def spike_worker():
            """Worker for spike load"""
            results = []
            end_time = time.time() + spike_duration
            
            while time.time() < end_time:
                result = self.test_endpoint_sync(url)
                results.append(result)
                time.sleep(random.uniform(0.01, 0.05))  # Aggressive requests
                
            return results
            
        # Execute spike
        with ThreadPoolExecutor(max_workers=spike_users) as executor:
            futures = [executor.submit(spike_worker) for _ in range(spike_users)]
            
            for future in as_completed(futures):
                all_results.extend(future.result())
                
        return self._calculate_metrics(endpoint, spike_duration, spike_users, 
                                      all_results, [], [])
                                      
    def stress_test(self, endpoint: str, initial_users: int = 10, 
                   user_increment: int = 10, max_response_time: float = 1000) -> LoadTestResult:
        """
        Stress test: find breaking point
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"STRESS TEST: {endpoint}")
        logger.info(f"Finding breaking point (max response time: {max_response_time}ms)")
        logger.info(f"{'='*60}")
        
        url = f"{self.base_url}{endpoint}"
        current_users = initial_users
        breaking_point_found = False
        all_results = []
        
        while not breaking_point_found and current_users <= 500:
            logger.info(f"\nTesting with {current_users} users...")
            
            round_results = []
            
            def stress_worker():
                results = []
                end_time = time.time() + 10  # 10 second rounds
                
                while time.time() < end_time:
                    result = self.test_endpoint_sync(url)
                    results.append(result)
                    time.sleep(0.05)
                    
                return results
                
            with ThreadPoolExecutor(max_workers=current_users) as executor:
                futures = [executor.submit(stress_worker) for _ in range(current_users)]
                
                for future in as_completed(futures):
                    round_results.extend(future.result())
                    
            # Check if breaking point reached
            response_times = [r['response_time_ms'] for r in round_results]
            avg_response = statistics.mean(response_times) if response_times else 0
            error_rate = sum(1 for r in round_results if not r['success']) / len(round_results) * 100
            
            logger.error(f"  Avg response: {avg_response:.0f}ms, Error rate: {error_rate:.1f}%")
            
            all_results.extend(round_results)
            
            if avg_response > max_response_time or error_rate > 10:
                breaking_point_found = True
                logger.info(f"\n🔴 Breaking point found at {current_users} users!")
            else:
                current_users += user_increment
                
        return self._calculate_metrics(endpoint, 10, current_users, 
                                      all_results, [], [])
                                      
    def endurance_test(self, endpoint: str, users: int = 50, 
                       duration_minutes: int = 5) -> LoadTestResult:
        """
        Endurance/Soak test: sustained load over time
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"ENDURANCE TEST: {endpoint}")
        logger.info(f"{users} users for {duration_minutes} minutes")
        logger.info(f"{'='*60}")
        
        url = f"{self.base_url}{endpoint}"
        duration_seconds = duration_minutes * 60
        all_results = []
        start_time = time.time()
        
        def endurance_worker():
            results = []
            
            while time.time() - start_time < duration_seconds:
                result = self.test_endpoint_sync(url)
                results.append(result)
                time.sleep(random.uniform(0.1, 0.5))  # Sustainable pace
                
            return results
            
        # Progress monitoring
        def monitor_progress():
            while time.time() - start_time < duration_seconds:
                elapsed = time.time() - start_time
                progress = (elapsed / duration_seconds) * 100
                logger.info(f"  Progress: {progress:.1f}% ({elapsed:.0f}/{duration_seconds}s)")
                time.sleep(10)
                
        # Start monitoring thread
        import threading
        monitor_thread = threading.Thread(target=monitor_progress)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Execute endurance test
        with ThreadPoolExecutor(max_workers=users) as executor:
            futures = [executor.submit(endurance_worker) for _ in range(users)]
            
            for future in as_completed(futures):
                all_results.extend(future.result())
                
        return self._calculate_metrics(endpoint, duration_seconds, users, 
                                      all_results, [], [])
                                      
    def _calculate_metrics(self, endpoint: str, duration: int, users: int,
                          results: List[dict], cpu_samples: List[float],
                          memory_samples: List[float]) -> LoadTestResult:
        """Calculate comprehensive metrics from results"""
        
        if not results:
            return LoadTestResult(
                endpoint=endpoint,
                duration_seconds=duration,
                virtual_users=users,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                requests_per_second=0,
                avg_response_time_ms=0,
                min_response_time_ms=0,
                max_response_time_ms=0,
                p50_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                error_rate_percent=100,
                throughput_mbps=0,
                cpu_usage_percent=0,
                memory_usage_mb=0,
                timestamp=datetime.now().isoformat()
            )
            
        # Calculate basic metrics
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r['success'])
        failed_requests = total_requests - successful_requests
        
        # Response times
        response_times = [r['response_time_ms'] for r in results]
        sorted_times = sorted(response_times)
        
        # Percentiles
        p50_idx = int(len(sorted_times) * 0.50)
        p95_idx = int(len(sorted_times) * 0.95)
        p99_idx = int(len(sorted_times) * 0.99)
        
        # Throughput
        total_bytes = sum(r.get('bytes', 0) for r in results)
        throughput_mbps = (total_bytes / duration / 1024 / 1024 * 8) if duration > 0 else 0
        
        return LoadTestResult(
            endpoint=endpoint,
            duration_seconds=duration,
            virtual_users=users,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            requests_per_second=total_requests / duration if duration > 0 else 0,
            avg_response_time_ms=statistics.mean(response_times),
            min_response_time_ms=min(response_times),
            max_response_time_ms=max(response_times),
            p50_response_time_ms=sorted_times[p50_idx] if p50_idx < len(sorted_times) else 0,
            p95_response_time_ms=sorted_times[p95_idx] if p95_idx < len(sorted_times) else 0,
            p99_response_time_ms=sorted_times[p99_idx] if p99_idx < len(sorted_times) else 0,
            error_rate_percent=(failed_requests / total_requests * 100) if total_requests > 0 else 0,
            throughput_mbps=throughput_mbps,
            cpu_usage_percent=statistics.mean(cpu_samples) if cpu_samples else 0,
            memory_usage_mb=statistics.mean(memory_samples) if memory_samples else 0,
            timestamp=datetime.now().isoformat()
        )
        
    def run_complete_test_suite(self) -> Dict[str, Any]:
        """Run complete ULTRAPERFORMANCE test suite"""
        
        logger.info("=" * 80)
        logger.info("ULTRAPERFORMANCE LOAD TESTING SUITE")
        logger.info("=" * 80)
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'summary': {},
            'recommendations': []
        }
        
        # Test scenarios
        test_scenarios = [
            # (endpoint, test_type, params)
            ("/health", "ramp", {"max_users": 50, "ramp_duration": 10, "test_duration": 30}),
            ("/api/v1/models/", "spike", {"normal_users": 5, "spike_users": 100, "spike_duration": 10}),
            ("/health", "stress", {"initial_users": 10, "user_increment": 10, "max_response_time": 500}),
            # ("/health", "endurance", {"users": 20, "duration_minutes": 2}),  # Shorter for demo
        ]
        
        for endpoint, test_type, params in test_scenarios:
            logger.info(f"\n{'='*60}")
            
            if test_type == "ramp":
                result = self.ramp_up_load_test(endpoint, **params)
            elif test_type == "spike":
                result = self.spike_test(endpoint, **params)
            elif test_type == "stress":
                result = self.stress_test(endpoint, **params)
            elif test_type == "endurance":
                result = self.endurance_test(endpoint, **params)
                
            test_results['tests'].append({
                'type': test_type,
                'result': asdict(result)
            })
            
            # Print results
            logger.info(f"\nResults:")
            logger.info(f"  Total Requests: {result.total_requests}")
            logger.error(f"  Success Rate: {100 - result.error_rate_percent:.1f}%")
            logger.info(f"  Requests/Second: {result.requests_per_second:.1f}")
            logger.info(f"  Avg Response: {result.avg_response_time_ms:.0f}ms")
            logger.info(f"  P95 Response: {result.p95_response_time_ms:.0f}ms")
            logger.info(f"  P99 Response: {result.p99_response_time_ms:.0f}ms")
            
        # Generate summary and recommendations
        test_results['summary'] = self._generate_summary(test_results['tests'])
        test_results['recommendations'] = self._generate_recommendations(test_results['summary'])
        
        return test_results
        
    def _generate_summary(self, tests: List[dict]) -> dict:
        """Generate test summary"""
        
        avg_response_times = []
        error_rates = []
        throughputs = []
        
        for test in tests:
            result = test['result']
            avg_response_times.append(result['avg_response_time_ms'])
            error_rates.append(result['error_rate_percent'])
            throughputs.append(result['requests_per_second'])
            
        performance_score = 100
        
        # Penalize high response times
        avg_response = statistics.mean(avg_response_times)
        if avg_response > 100:
            performance_score -= min(30, avg_response / 10)
            
        # Penalize errors
        avg_error_rate = statistics.mean(error_rates)
        if avg_error_rate > 1:
            performance_score -= min(30, avg_error_rate * 5)
            
        # Reward high throughput
        avg_throughput = statistics.mean(throughputs)
        if avg_throughput > 100:
            performance_score += min(10, avg_throughput / 50)
            
        return {
            'avg_response_time_ms': avg_response,
            'avg_error_rate_percent': avg_error_rate,
            'avg_throughput_rps': avg_throughput,
            'performance_score': max(0, min(100, performance_score)),
            'grade': 'A' if performance_score >= 90 else 'B' if performance_score >= 80 else 'C' if performance_score >= 70 else 'D' if performance_score >= 60 else 'F'
        }
        
    def _generate_recommendations(self, summary: dict) -> List[str]:
        """Generate performance recommendations"""
        
        recommendations = []
        
        if summary['avg_response_time_ms'] > 100:
            recommendations.append("🔴 HIGH PRIORITY: Optimize response times - implement caching and query optimization")
            
        if summary['avg_error_rate_percent'] > 1:
            recommendations.append("🔴 HIGH PRIORITY: Reduce error rate - improve error handling and retry logic")
            
        if summary['avg_throughput_rps'] < 50:
            recommendations.append("🟡 MEDIUM PRIORITY: Increase throughput - consider horizontal scaling")
            
        if summary['performance_score'] < 80:
            recommendations.append("🟡 MEDIUM PRIORITY: Overall performance needs improvement")
            
        # Always include best practices
        recommendations.extend([
            "✅ Implement request rate limiting to prevent overload",
            "✅ Use CDN for static assets",
            "✅ Enable HTTP/2 for better multiplexing",
            "✅ Implement circuit breakers for resilience",
            "✅ Monitor with APM tools for real-time insights"
        ])
        
        return recommendations


def main():
    """Run the complete ULTRAPERFORMANCE test suite"""
    
    tester = UltraPerformanceLoadTester()
    results = tester.run_complete_test_suite()
    
    # Print final summary
    logger.info("\n" + "=" * 80)
    logger.info("ULTRAPERFORMANCE TEST SUMMARY")
    logger.info("=" * 80)
    
    summary = results['summary']
    logger.info(f"\nPerformance Score: {summary['performance_score']}/100 (Grade: {summary['grade']})")
    logger.info(f"Average Response Time: {summary['avg_response_time_ms']:.0f}ms")
    logger.error(f"Average Error Rate: {summary['avg_error_rate_percent']:.1f}%")
    logger.info(f"Average Throughput: {summary['avg_throughput_rps']:.1f} req/s")
    
    logger.info("\nRECOMMENDATIONS:")
    for rec in results['recommendations']:
        logger.info(f"  {rec}")
        
    # Save results
    report_file = f"/opt/sutazaiapp/reports/load_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"\nFull report saved to: {report_file}")
    
    return results


if __name__ == "__main__":
    main()