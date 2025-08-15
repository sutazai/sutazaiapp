#!/usr/bin/env python3
"""
SutazAI Performance Testing Suite
================================

Comprehensive performance testing including:
- Load testing with configurable concurrent users
- Response time benchmarking
- Memory and resource usage monitoring
- Stress testing for system limits
- Throughput measurement
"""

import asyncio
import aiohttp
import time
import statistics
import json
from datetime import datetime
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    response_times: List[float]
    success_count: int
    error_count: int
    timeout_count: int
    total_requests: int
    start_time: float
    end_time: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        return (self.success_count / self.total_requests * 100) if self.total_requests > 0 else 0
    
    @property
    def requests_per_second(self) -> float:
        return self.total_requests / self.duration if self.duration > 0 else 0
    
    @property
    def avg_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0
    
    @property
    def median_response_time(self) -> float:
        return statistics.median(self.response_times) if self.response_times else 0
    
    @property
    def p95_response_time(self) -> float:
        if not self.response_times:
            return 0
        return statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile
    
    @property
    def p99_response_time(self) -> float:
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        index = int(0.99 * len(sorted_times))
        return sorted_times[index] if index < len(sorted_times) else sorted_times[-1]

class PerformanceTestSuite:
    """Comprehensive performance testing suite"""
    
    def __init__(self):
        self.base_url = "http://localhost:10010"
        self.endpoints = [
            "/health",
            "/api/v1",
            "/api/v1/agents",
            "/api/v1/agents/status",
            "/api/v1/health"
        ]
        
    async def run_load_test(self, concurrent_users: int = 10, duration_seconds: int = 30) -> PerformanceMetrics:
        """Run load test with specified concurrent users"""
        logger.info(f"Running load test: {concurrent_users} concurrent users for {duration_seconds} seconds")
        
        async def make_request(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
            """Make a single HTTP request"""
            start_time = time.time()
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    await response.text()  # Read response body
                    end_time = time.time()
                    return {
                        "success": response.status == 200,
                        "status_code": response.status,
                        "response_time": end_time - start_time,
                        "error": None,
                        "timeout": False
                    }
            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "status_code": 0,
                    "response_time": time.time() - start_time,
                    "error": "timeout",
                    "timeout": True
                }
            except Exception as e:
                return {
                    "success": False,
                    "status_code": 0,
                    "response_time": time.time() - start_time,
                    "error": str(e),
                    "timeout": False
                }
        
        async def user_session(user_id: int, test_duration: float) -> List[Dict[str, Any]]:
            """Simulate a user session with multiple requests"""
            results = []
            end_time = time.time() + test_duration
            
            async with aiohttp.ClientSession() as session:
                while time.time() < end_time:
                    # Randomly select an endpoint
                    import random
                    endpoint = random.choice(self.endpoints)
                    url = f"{self.base_url}{endpoint}"
                    
                    result = await make_request(session, url)
                    result["user_id"] = user_id
                    result["endpoint"] = endpoint
                    results.append(result)
                    
                    # Small delay between requests (simulate real user behavior)
                    await asyncio.sleep(0.1)
            
            return results
        
        # Run concurrent user sessions
        start_time = time.time()
        
        tasks = [
            user_session(user_id, duration_seconds)
            for user_id in range(concurrent_users)
        ]
        
        all_results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Flatten results
        flat_results = [result for user_results in all_results for result in user_results]
        
        # Calculate metrics
        response_times = [r["response_time"] for r in flat_results if r["success"]]
        success_count = sum(1 for r in flat_results if r["success"])
        error_count = sum(1 for r in flat_results if not r["success"] and not r["timeout"])
        timeout_count = sum(1 for r in flat_results if r["timeout"])
        
        return PerformanceMetrics(
            response_times=response_times,
            success_count=success_count,
            error_count=error_count,
            timeout_count=timeout_count,
            total_requests=len(flat_results),
            start_time=start_time,
            end_time=end_time
        )
    
    def run_sync_load_test(self, concurrent_users: int = 10, requests_per_user: int = 50) -> PerformanceMetrics:
        """Run synchronous load test using threading"""
        logger.info(f"Running sync load test: {concurrent_users} users, {requests_per_user} requests each")
        
        import requests
        
        def user_session(user_id: int) -> List[Dict[str, Any]]:
            """Simulate a user session"""
            results = []
            session = requests.Session()
            
            for i in range(requests_per_user):
                import random
                endpoint = random.choice(self.endpoints)
                url = f"{self.base_url}{endpoint}"
                
                start_time = time.time()
                try:
                    response = session.get(url, timeout=10)
                    end_time = time.time()
                    
                    results.append({
                        "success": response.status_code == 200,
                        "status_code": response.status_code,
                        "response_time": end_time - start_time,
                        "error": None,
                        "timeout": False,
                        "user_id": user_id,
                        "endpoint": endpoint
                    })
                    
                except requests.exceptions.Timeout:
                    results.append({
                        "success": False,
                        "status_code": 0,
                        "response_time": time.time() - start_time,
                        "error": "timeout",
                        "timeout": True,
                        "user_id": user_id,
                        "endpoint": endpoint
                    })
                except Exception as e:
                    results.append({
                        "success": False,
                        "status_code": 0,
                        "response_time": time.time() - start_time,
                        "error": str(e),
                        "timeout": False,
                        "user_id": user_id,
                        "endpoint": endpoint
                    })
            
            return results
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_session, user_id) for user_id in range(concurrent_users)]
            all_results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        
        # Flatten results
        flat_results = [result for user_results in all_results for result in user_results]
        
        # Calculate metrics
        response_times = [r["response_time"] for r in flat_results if r["success"]]
        success_count = sum(1 for r in flat_results if r["success"])
        error_count = sum(1 for r in flat_results if not r["success"] and not r["timeout"])
        timeout_count = sum(1 for r in flat_results if r["timeout"])
        
        return PerformanceMetrics(
            response_times=response_times,
            success_count=success_count,
            error_count=error_count,
            timeout_count=timeout_count,
            total_requests=len(flat_results),
            start_time=start_time,
            end_time=end_time
        )
    
    async def run_stress_test(self) -> Dict[str, Any]:
        """Run stress test to find system limits"""
        logger.info("Running stress test to find system limits...")
        
        user_levels = [1, 5, 10, 20, 30, 50]
        results = {}
        
        for users in user_levels:
            logger.info(f"Testing with {users} concurrent users...")
            
            # Run shorter test for stress testing
            metrics = self.run_sync_load_test(concurrent_users=users, requests_per_user=20)
            
            results[f"{users}_users"] = {
                "concurrent_users": users,
                "success_rate": metrics.success_rate,
                "avg_response_time": metrics.avg_response_time,
                "requests_per_second": metrics.requests_per_second,
                "p95_response_time": metrics.p95_response_time,
                "total_requests": metrics.total_requests,
                "errors": metrics.error_count,
                "timeouts": metrics.timeout_count
            }
            
            # Stop if success rate drops below 80%
            if metrics.success_rate < 80:
                logger.warning(f"Success rate dropped to {metrics.success_rate:.1f}% at {users} users")
                break
            
            # Add delay between stress levels
            time.sleep(2)
        
        return results
    
    async def run_endpoint_performance_analysis(self) -> Dict[str, Any]:
        """Analyze performance of individual endpoints"""
        logger.info("Running endpoint performance analysis...")
        
        import requests
        results = {}
        
        for endpoint in self.endpoints:
            logger.info(f"Analyzing endpoint: {endpoint}")
            
            url = f"{self.base_url}{endpoint}"
            response_times = []
            status_codes = []
            
            # Make 50 requests to each endpoint
            for i in range(50):
                start_time = time.time()
                try:
                    response = requests.get(url, timeout=10)
                    end_time = time.time()
                    
                    response_times.append(end_time - start_time)
                    status_codes.append(response.status_code)
                    
                except Exception as e:
                    logger.warning(f"Request failed for {endpoint}: {e}")
                    status_codes.append(0)
                
                # Small delay between requests
                time.sleep(0.05)
            
            if response_times:
                results[endpoint] = {
                    "avg_response_time": statistics.mean(response_times),
                    "median_response_time": statistics.median(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                    "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
                    "success_rate": (status_codes.count(200) / len(status_codes)) * 100,
                    "total_requests": len(status_codes),
                    "successful_requests": status_codes.count(200)
                }
            else:
                results[endpoint] = {
                    "error": "No successful requests",
                    "total_requests": len(status_codes),
                    "successful_requests": 0,
                    "success_rate": 0
                }
        
        return results
    
    async def run_memory_performance_test(self) -> Dict[str, Any]:
        """Test memory usage during high load"""
        logger.info("Running memory performance test...")
        
        try:
            import psutil
            import gc
            
            # Initial memory reading
            process = psutil.Process()
            initial_memory = process.memory_info()
            
            # Force garbage collection
            gc.collect()
            
            logger.info(f"Initial memory usage: {initial_memory.rss / 1024 / 1024:.1f} MB")
            
            # Run load test while monitoring memory
            memory_readings = []
            
            def monitor_memory():
                for _ in range(60):  # Monitor for 60 seconds
                    memory_info = process.memory_info()
                    memory_readings.append({
                        "timestamp": time.time(),
                        "rss_mb": memory_info.rss / 1024 / 1024,
                        "vms_mb": memory_info.vms / 1024 / 1024
                    })
                    time.sleep(1)
            
            # Start memory monitoring in background
            monitor_thread = threading.Thread(target=monitor_memory)
            monitor_thread.start()
            
            # Run load test
            load_metrics = self.run_sync_load_test(concurrent_users=20, requests_per_user=30)
            
            # Wait for monitoring to complete
            monitor_thread.join()
            
            # Final memory reading
            final_memory = process.memory_info()
            
            # Calculate memory statistics
            if memory_readings:
                rss_values = [r["rss_mb"] for r in memory_readings]
                memory_stats = {
                    "initial_memory_mb": initial_memory.rss / 1024 / 1024,
                    "final_memory_mb": final_memory.rss / 1024 / 1024,
                    "peak_memory_mb": max(rss_values),
                    "avg_memory_mb": statistics.mean(rss_values),
                    "memory_increase_mb": (final_memory.rss - initial_memory.rss) / 1024 / 1024,
                    "memory_readings": len(memory_readings)
                }
            else:
                memory_stats = {
                    "error": "No memory readings collected"
                }
            
            return {
                "memory_stats": memory_stats,
                "load_test_results": {
                    "success_rate": load_metrics.success_rate,
                    "avg_response_time": load_metrics.avg_response_time,
                    "total_requests": load_metrics.total_requests
                }
            }
            
        except ImportError:
            return {
                "error": "psutil not available for memory monitoring"
            }
        except Exception as e:
            return {
                "error": str(e)
            }
    
    async def run_comprehensive_performance_suite(self) -> Dict[str, Any]:
        """Run complete performance test suite"""
        logger.info("Starting comprehensive performance test suite...")
        
        results = {
            "execution_timestamp": datetime.now().isoformat(),
            "test_configuration": {
                "base_url": self.base_url,
                "endpoints_tested": self.endpoints
            }
        }
        
        # 1. Basic load test
        logger.info("1. Running basic load test...")
        basic_load = self.run_sync_load_test(concurrent_users=10, requests_per_user=30)
        results["basic_load_test"] = {
            "concurrent_users": 10,
            "requests_per_user": 30,
            "success_rate": basic_load.success_rate,
            "avg_response_time": basic_load.avg_response_time,
            "median_response_time": basic_load.median_response_time,
            "p95_response_time": basic_load.p95_response_time,
            "p99_response_time": basic_load.p99_response_time,
            "requests_per_second": basic_load.requests_per_second,
            "total_requests": basic_load.total_requests,
            "errors": basic_load.error_count,
            "timeouts": basic_load.timeout_count,
            "duration": basic_load.duration
        }
        
        # 2. Stress test
        logger.info("2. Running stress test...")
        stress_results = await self.run_stress_test()
        results["stress_test"] = stress_results
        
        # 3. Endpoint analysis
        logger.info("3. Running endpoint performance analysis...")
        endpoint_results = await self.run_endpoint_performance_analysis()
        results["endpoint_analysis"] = endpoint_results
        
        # 4. Memory performance test
        logger.info("4. Running memory performance test...")
        memory_results = await self.run_memory_performance_test()
        results["memory_performance"] = memory_results
        
        # 5. High concurrency test
        logger.info("5. Running high concurrency test...")
        high_concurrency = self.run_sync_load_test(concurrent_users=25, requests_per_user=20)
        results["high_concurrency_test"] = {
            "concurrent_users": 25,
            "requests_per_user": 20,
            "success_rate": high_concurrency.success_rate,
            "avg_response_time": high_concurrency.avg_response_time,
            "requests_per_second": high_concurrency.requests_per_second,
            "total_requests": high_concurrency.total_requests,
            "errors": high_concurrency.error_count
        }
        
        # Generate performance summary
        results["performance_summary"] = self._generate_performance_summary(results)
        
        # Save report
        await self._save_performance_report(results)
        
        return results
    
    def _generate_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary and recommendations"""
        
        basic_load = results.get("basic_load_test", {})
        high_concurrency = results.get("high_concurrency_test", {})
        
        # Performance thresholds
        response_time_threshold = 1.0  # 1 second
        success_rate_threshold = 95.0  # 95%
        
        # Evaluate performance
        performance_issues = []
        recommendations = []
        
        if basic_load.get("avg_response_time", 0) > response_time_threshold:
            performance_issues.append("High average response time")
            recommendations.append("Consider optimizing database queries and caching")
        
        if basic_load.get("success_rate", 0) < success_rate_threshold:
            performance_issues.append("Low success rate")
            recommendations.append("Investigate error causes and improve error handling")
        
        if high_concurrency.get("success_rate", 0) < 90:
            performance_issues.append("Poor performance under high concurrency")
            recommendations.append("Consider implementing connection pooling and load balancing")
        
        # Calculate overall performance score
        success_score = min(basic_load.get("success_rate", 0) / 100, 1.0) * 40
        response_time_score = max(0, (2.0 - basic_load.get("avg_response_time", 2.0)) / 2.0) * 30
        concurrency_score = min(high_concurrency.get("success_rate", 0) / 100, 1.0) * 30
        
        overall_score = success_score + response_time_score + concurrency_score
        
        if overall_score >= 90:
            performance_grade = "EXCELLENT"
        elif overall_score >= 80:
            performance_grade = "GOOD"
        elif overall_score >= 70:
            performance_grade = "FAIR"
        elif overall_score >= 60:
            performance_grade = "POOR"
        else:
            performance_grade = "CRITICAL"
        
        return {
            "overall_performance_score": overall_score,
            "performance_grade": performance_grade,
            "performance_issues": performance_issues,
            "recommendations": recommendations,
            "key_metrics": {
                "avg_response_time": basic_load.get("avg_response_time", 0),
                "success_rate": basic_load.get("success_rate", 0),
                "requests_per_second": basic_load.get("requests_per_second", 0),
                "high_concurrency_success_rate": high_concurrency.get("success_rate", 0)
            }
        }
    
    async def _save_performance_report(self, report: Dict[str, Any]) -> None:
        """Save performance test report"""
        try:
            reports_dir = Path("/opt/sutazaiapp/data/workflow_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_file = reports_dir / f"performance_test_report_{timestamp}.json"
            
            with open(json_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Performance test report saved: {json_file}")
            
        except Exception as e:
            logger.error(f"Failed to save performance test report: {e}")

async def main():
    """Run performance tests"""
    suite = PerformanceTestSuite()
    report = await suite.run_comprehensive_performance_suite()
    
    logger.info("\n" + "="*80)
    logger.info("SUTAZAI PERFORMANCE TEST RESULTS")
    logger.info("="*80)
    
    summary = report.get("performance_summary", {})
    logger.info(f"Performance Grade: {summary.get('performance_grade', 'UNKNOWN')}")
    logger.info(f"Overall Score: {summary.get('overall_performance_score', 0):.1f}/100")
    
    key_metrics = summary.get("key_metrics", {})
    logger.info(f"\nKey Metrics:")
    logger.info(f"  Average Response Time: {key_metrics.get('avg_response_time', 0):.3f}s")
    logger.info(f"  Success Rate: {key_metrics.get('success_rate', 0):.1f}%")
    logger.info(f"  Requests per Second: {key_metrics.get('requests_per_second', 0):.1f}")
    logger.info(f"  High Concurrency Success Rate: {key_metrics.get('high_concurrency_success_rate', 0):.1f}%")
    
    issues = summary.get("performance_issues", [])
    if issues:
        logger.info(f"\nPerformance Issues:")
        for issue in issues:
            logger.info(f"  • {issue}")
    
    recommendations = summary.get("recommendations", [])
    if recommendations:
        logger.info(f"\nRecommendations:")
        for rec in recommendations:
            logger.info(f"  • {rec}")
    
    logger.info("="*80)
    
    return report

if __name__ == "__main__":
    asyncio.run(main())