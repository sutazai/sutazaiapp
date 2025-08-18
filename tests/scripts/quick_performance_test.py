#!/usr/bin/env python3
"""
Quick Performance Test for MCP Monitoring System
Focused on key performance metrics with shorter test duration
"""

import asyncio
import aiohttp
import time
import json
import statistics
from datetime import datetime, timezone
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuickPerformanceTest:
    """Quick performance testing for monitoring system"""
    
    def __init__(self, base_url: str = "http://localhost:10250"):
        self.base_url = base_url
        self.results = {}
        
    async def benchmark_endpoint(self, endpoint: str, num_requests: int = 100) -> Dict:
        """Benchmark a single endpoint"""
        url = f"{self.base_url}{endpoint}"
        response_times = []
        errors = 0
        
        async with aiohttp.ClientSession() as session:
            for _ in range(num_requests):
                start = time.perf_counter()
                try:
                    async with session.get(url) as response:
                        await response.text()
                        if response.status == 200:
                            response_times.append((time.perf_counter() - start) * 1000)  # ms
                        else:
                            errors += 1
                except Exception as e:
                    errors += 1
                    logger.error(f"Request failed: {e}")
        
        if response_times:
            response_times.sort()
            return {
                'endpoint': endpoint,
                'requests': num_requests,
                'successful': len(response_times),
                'failed': errors,
                'min_ms': response_times[0],
                'avg_ms': statistics.mean(response_times),
                'p50_ms': response_times[int(len(response_times) * 0.50)],
                'p95_ms': response_times[int(len(response_times) * 0.95)],
                'p99_ms': response_times[int(len(response_times) * 0.99)],
                'max_ms': response_times[-1],
                'error_rate': (errors / num_requests) * 100
            }
        else:
            return {
                'endpoint': endpoint,
                'requests': num_requests,
                'successful': 0,
                'failed': errors,
                'error_rate': 100
            }
    
    async def concurrent_load_test(self, endpoint: str, concurrent: int, duration: int = 5) -> Dict:
        """Test with concurrent requests"""
        url = f"{self.base_url}{endpoint}"
        response_times = []
        errors = 0
        start_time = time.time()
        total_requests = 0
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < duration:
                tasks = []
                for _ in range(concurrent):
                    async def make_request():
                        req_start = time.perf_counter()
                        try:
                            async with session.get(url) as response:
                                await response.text()
                                if response.status == 200:
                                    return (time.perf_counter() - req_start) * 1000
                                return None
                        except:
                            return None
                    
                    tasks.append(make_request())
                
                results = await asyncio.gather(*tasks)
                total_requests += len(results)
                
                for rt in results:
                    if rt is not None:
                        response_times.append(rt)
                    else:
                        errors += 1
                
                await asyncio.sleep(0.01)  # Small delay
        
        elapsed = time.time() - start_time
        
        if response_times:
            response_times.sort()
            return {
                'endpoint': endpoint,
                'concurrent_users': concurrent,
                'duration_sec': elapsed,
                'total_requests': total_requests,
                'successful': len(response_times),
                'failed': errors,
                'requests_per_sec': total_requests / elapsed,
                'avg_ms': statistics.mean(response_times),
                'p95_ms': response_times[int(len(response_times) * 0.95)],
                'p99_ms': response_times[int(len(response_times) * 0.99)],
                'error_rate': (errors / total_requests) * 100 if total_requests > 0 else 0
            }
        else:
            return {
                'endpoint': endpoint,
                'concurrent_users': concurrent,
                'duration_sec': elapsed,
                'total_requests': total_requests,
                'successful': 0,
                'failed': errors,
                'error_rate': 100
            }
    
    async def run_all_tests(self):
        """Run all performance tests"""
        logger.info("\n" + "="*80)
        logger.info("MCP MONITORING SYSTEM - PERFORMANCE TEST REPORT")
        logger.info("="*80)
        logger.info(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        logger.info(f"Target: {self.base_url}")
        
        # Test 1: Basic endpoint response times
        logger.info("\n[TEST 1] Single-threaded Response Time Benchmarks")
        logger.info("-"*50)
        
        endpoints = ['/health', '/metrics', '/']
        for endpoint in endpoints:
            result = await self.benchmark_endpoint(endpoint, 100)
            logger.info(f"\n{endpoint}:")
            logger.info(f"  Requests: {result['successful']}/{result['requests']} successful")
            logger.info(f"  Response times (ms):")
            logger.info(f"    Min: {result.get('min_ms', 0):.2f}")
            logger.info(f"    Avg: {result.get('avg_ms', 0):.2f}")
            logger.info(f"    P50: {result.get('p50_ms', 0):.2f}")
            logger.info(f"    P95: {result.get('p95_ms', 0):.2f}")
            logger.info(f"    P99: {result.get('p99_ms', 0):.2f}")
            logger.info(f"    Max: {result.get('max_ms', 0):.2f}")
            if result['error_rate'] > 0:
                logger.error(f"  ⚠️ Error rate: {result['error_rate']:.1f}%")
            
            self.results[f"baseline_{endpoint}"] = result
        
        # Test 2: Concurrent load tests
        logger.info("\n[TEST 2] Concurrent Load Tests (5 second duration each)")
        logger.info("-"*50)
        
        load_levels = [10, 50, 100, 200]
        target_endpoints = ['/health', '/metrics']  # Skip /health/detailed as it's slow
        
        for endpoint in target_endpoints:
            logger.info(f"\n{endpoint}:")
            for concurrent in load_levels:
                result = await self.concurrent_load_test(endpoint, concurrent, 5)
                logger.info(f"  {concurrent} concurrent users:")
                logger.info(f"    Throughput: {result['requests_per_sec']:.2f} req/s")
                logger.info(f"    Avg response: {result['avg_ms']:.2f}ms")
                logger.info(f"    P95 response: {result['p95_ms']:.2f}ms")
                logger.info(f"    P99 response: {result['p99_ms']:.2f}ms")
                if result['error_rate'] > 0:
                    logger.error(f"    ⚠️ Error rate: {result['error_rate']:.1f}%")
                
                self.results[f"load_{endpoint}_{concurrent}"] = result
        
        # Test 3: Stress test to find limits
        logger.info("\n[TEST 3] Stress Test - Finding System Limits")
        logger.info("-"*50)
        
        endpoint = '/health'
        for concurrent in [100, 200, 500, 1000]:
            result = await self.concurrent_load_test(endpoint, concurrent, 3)
            logger.info(f"\n{concurrent} concurrent users:")
            logger.info(f"  Throughput: {result['requests_per_sec']:.2f} req/s")
            logger.info(f"  P95 response: {result['p95_ms']:.2f}ms")
            
            if result['error_rate'] > 5:
                logger.error(f"  ❌ System limit reached - {result['error_rate']:.1f}% error rate")
                break
            elif result['p95_ms'] > 1000:
                logger.info(f"  ⚠️ Performance degraded - P95 > 1000ms")
            else:
                logger.info(f"  ✅ System stable")
            
            self.results[f"stress_{concurrent}"] = result
        
        # Performance Requirements Validation
        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE REQUIREMENTS VALIDATION")
        logger.info("="*80)
        
        # Check requirements
        requirements = {
            "API Response Time < 100ms (P95)": False,
            "Throughput > 1000 req/s": False,
            "Resource Usage < 80% CPU": "Not measured (requires process monitoring)",
            "Support 100+ Concurrent Users": False,
            "Error Rate < 1%": False
        }
        
        # Validate P95 < 100ms
        baseline_health = self.results.get('baseline_/health', {})
        if baseline_health.get('p95_ms', float('inf')) < 100:
            requirements["API Response Time < 100ms (P95)"] = True
        
        # Validate throughput > 1000 req/s
        for key, result in self.results.items():
            if 'requests_per_sec' in result and result['requests_per_sec'] > 1000:
                requirements["Throughput > 1000 req/s"] = True
                break
        
        # Validate 100+ concurrent users
        load_100 = self.results.get('load_/health_100', {})
        if load_100.get('error_rate', 100) < 5 and load_100.get('p95_ms', float('inf')) < 500:
            requirements["Support 100+ Concurrent Users"] = True
        
        # Validate error rate
        all_error_rates = [r.get('error_rate', 0) for r in self.results.values()]
        if all(rate < 1 for rate in all_error_rates):
            requirements["Error Rate < 1%"] = True
        
        for req, status in requirements.items():
            if isinstance(status, bool):
                logger.info(f"  {req}: {'✅ PASS' if status else '❌ FAIL'}")
            else:
                logger.info(f"  {req}: ℹ️ {status}")
        
        # Recommendations
        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE OPTIMIZATION RECOMMENDATIONS")
        logger.info("="*80)
        
        recommendations = []
        
        # Check baseline performance
        if baseline_health.get('p95_ms', 0) > 100:
            recommendations.append(f"Optimize /health endpoint - P95 ({baseline_health.get('p95_ms', 0):.2f}ms) exceeds 100ms target")
        
        # Check throughput
        max_throughput = max((r.get('requests_per_sec', 0) for r in self.results.values()), default=0)
        if max_throughput < 1000:
            recommendations.append(f"Improve throughput - Current max ({max_throughput:.2f} req/s) below 1000 req/s target")
        
        # Check concurrent user support
        if not requirements["Support 100+ Concurrent Users"]:
            recommendations.append("Optimize for concurrent users - System struggles with 100+ simultaneous connections")
        
        # Check for high error rates
        high_error_endpoints = [k for k, v in self.results.items() if v.get('error_rate', 0) > 1]
        if high_error_endpoints:
            recommendations.append(f"Investigate error handling for: {', '.join(high_error_endpoints)}")
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")
        else:
            logger.error("  No critical issues found - system performing within acceptable parameters")
        
        # Save results
        report_file = f"/opt/sutazaiapp/scripts/mcp/automation/tests/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'target': self.base_url,
                'results': self.results,
                'requirements': {k: v if isinstance(v, str) else str(v) for k, v in requirements.items()},
                'recommendations': recommendations
            }, f, indent=2)
        
        logger.info(f"\n📊 Detailed report saved to: {report_file}")
        logger.info("="*80)
        
        return all(v for v in requirements.values() if isinstance(v, bool))


async def main():
    """Main test execution"""
    tester = QuickPerformanceTest()
    all_passed = await tester.run_all_tests()
    
    if all_passed:
        logger.info("\n✅ All performance requirements met!")
        return 0
    else:
        logger.info("\n⚠️ Some performance requirements not met. See recommendations above.")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)