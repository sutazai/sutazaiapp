#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
ULTRAPERFORMANCE Load Testing Suite
Target: <50ms response, 100+ concurrent users, 80%+ cache hit rate
"""

import asyncio
import aiohttp
import time
import statistics
import json
from typing import List, Dict, Any
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class UltraPerformanceTest:
    """High-performance load testing framework"""
    
    def __init__(self, base_url: str = "http://localhost:10010"):
        self.base_url = base_url
        self.results = {
            "response_times": [],
            "errors": [],
            "status_codes": {},
            "cache_hits": 0,
            "cache_misses": 0,
            "start_time": None,
            "end_time": None
        }
        
    async def test_endpoint(self, session: aiohttp.ClientSession, endpoint: str, method: str = "GET", data: Any = None) -> Dict:
        """Test a single endpoint and measure performance"""
        start_time = time.perf_counter()
        
        try:
            if method == "GET":
                async with session.get(f"{self.base_url}{endpoint}") as response:
                    result = await response.json()
                    elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
                    
                    return {
                        "endpoint": endpoint,
                        "status": response.status,
                        "response_time_ms": elapsed,
                        "cached": result.get("cached", False),
                        "success": response.status == 200
                    }
            else:  # POST
                async with session.post(f"{self.base_url}{endpoint}", json=data) as response:
                    result = await response.json()
                    elapsed = (time.perf_counter() - start_time) * 1000
                    
                    return {
                        "endpoint": endpoint,
                        "status": response.status,
                        "response_time_ms": elapsed,
                        "cached": result.get("cached", False),
                        "success": response.status == 200
                    }
                    
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            return {
                "endpoint": endpoint,
                "status": 0,
                "response_time_ms": elapsed,
                "error": str(e),
                "success": False
            }
    
    async def run_concurrent_load(self, endpoint: str, concurrent_users: int, requests_per_user: int, method: str = "GET", data: Any = None):
        """Run concurrent load test on an endpoint"""
        logger.info(f"\n🚀 Testing {endpoint} with {concurrent_users} concurrent users...")
        
        async with aiohttp.ClientSession() as session:
            # Warm up the endpoint first
            await self.test_endpoint(session, endpoint, method, data)
            
            # Create tasks for all concurrent users
            tasks = []
            for user in range(concurrent_users):
                for req in range(requests_per_user):
                    tasks.append(self.test_endpoint(session, endpoint, method, data))
            
            # Execute all requests concurrently
            results = await asyncio.gather(*tasks)
            
            # Process results
            response_times = []
            errors = 0
            cache_hits = 0
            
            for result in results:
                if result["success"]:
                    response_times.append(result["response_time_ms"])
                    if result.get("cached"):
                        cache_hits += 1
                else:
                    errors += 1
                    self.results["errors"].append(result)
                
                # Track status codes
                status = result["status"]
                self.results["status_codes"][status] = self.results["status_codes"].get(status, 0) + 1
            
            # Calculate statistics
            if response_times:
                stats = {
                    "endpoint": endpoint,
                    "concurrent_users": concurrent_users,
                    "total_requests": len(results),
                    "successful_requests": len(response_times),
                    "failed_requests": errors,
                    "cache_hit_rate": (cache_hits / len(response_times)) * 100 if response_times else 0,
                    "response_times": {
                        "min_ms": min(response_times),
                        "max_ms": max(response_times),
                        "mean_ms": statistics.mean(response_times),
                        "median_ms": statistics.median(response_times),
                        "p95_ms": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
                        "p99_ms": statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
                    }
                }
                
                # Check if meets ULTRAPERFORMANCE targets
                meets_target = stats["response_times"]["median_ms"] < 50
                stats["meets_ultraperformance"] = meets_target
                
                return stats
            else:
                return {
                    "endpoint": endpoint,
                    "error": "All requests failed",
                    "failed_requests": errors
                }
    
    async def run_comprehensive_test(self):
        """Run comprehensive performance test suite"""
        logger.info("=" * 80)
        logger.info("ULTRAPERFORMANCE LOAD TEST SUITE")
        logger.info("=" * 80)
        
        self.results["start_time"] = datetime.now().isoformat()
        
        test_results = []
        
        # Test 1: Health endpoint (should be <5ms)
        result = await self.run_concurrent_load("/health", 100, 10)
        test_results.append(result)
        self.print_results(result)
        
        # Test 2: API status endpoint with caching
        result = await self.run_concurrent_load("/api/v1/status", 50, 10)
        test_results.append(result)
        self.print_results(result)
        
        # Test 3: Agents listing
        result = await self.run_concurrent_load("/api/v1/agents", 50, 5)
        test_results.append(result)
        self.print_results(result)
        
        # Test 4: Chat endpoint (POST)
        chat_data = {
            "message": "Hello, how are you?",
            "model": "tinyllama",
            "use_cache": True
        }
        result = await self.run_concurrent_load("/api/v1/chat", 25, 5, "POST", chat_data)
        test_results.append(result)
        self.print_results(result)
        
        # Test 5: Cache stats endpoint
        result = await self.run_concurrent_load("/api/v1/cache/stats", 100, 5)
        test_results.append(result)
        self.print_results(result)
        
        # Test 6: Metrics endpoint
        result = await self.run_concurrent_load("/api/v1/metrics", 50, 5)
        test_results.append(result)
        self.print_results(result)
        
        self.results["end_time"] = datetime.now().isoformat()
        
        # Generate final report
        self.generate_final_report(test_results)
        
        return test_results
    
    def print_results(self, stats: Dict):
        """Print formatted test results"""
        logger.info(f"\n📊 Results for {stats.get('endpoint', 'Unknown')}:")
        logger.info(f"   Total Requests: {stats.get('total_requests', 0)}")
        logger.info(f"   Successful: {stats.get('successful_requests', 0)}")
        logger.error(f"   Failed: {stats.get('failed_requests', 0)}")
        
        if "response_times" in stats:
            rt = stats["response_times"]
            logger.info(f"   Response Times:")
            logger.info(f"      Min: {rt['min_ms']:.2f}ms")
            logger.info(f"      Median: {rt['median_ms']:.2f}ms")
            logger.info(f"      Mean: {rt['mean_ms']:.2f}ms")
            logger.info(f"      Max: {rt['max_ms']:.2f}ms")
            logger.info(f"      P95: {rt['p95_ms']:.2f}ms")
            logger.info(f"      P99: {rt['p99_ms']:.2f}ms")
            
            if "cache_hit_rate" in stats:
                logger.info(f"   Cache Hit Rate: {stats['cache_hit_rate']:.1f}%")
            
            # Performance verdict
            if stats.get("meets_ultraperformance"):
                logger.info(f"   ✅ MEETS ULTRAPERFORMANCE TARGET (<50ms median)")
            else:
                logger.info(f"   ❌ FAILS ULTRAPERFORMANCE TARGET (>{rt['median_ms']:.2f}ms median)")
    
    def generate_final_report(self, test_results: List[Dict]):
        """Generate comprehensive final report"""
        logger.info("\n" + "=" * 80)
        logger.info("FINAL PERFORMANCE REPORT")
        logger.info("=" * 80)
        
        # Overall statistics
        total_requests = sum(r.get("total_requests", 0) for r in test_results)
        successful_requests = sum(r.get("successful_requests", 0) for r in test_results)
        failed_requests = sum(r.get("failed_requests", 0) for r in test_results)
        
        logger.info(f"\n📈 Overall Statistics:")
        logger.info(f"   Total Requests: {total_requests}")
        logger.info(f"   Successful: {successful_requests} ({(successful_requests/total_requests*100):.1f}%)")
        logger.error(f"   Failed: {failed_requests} ({(failed_requests/total_requests*100):.1f}%)")
        
        # Performance summary
        passing_endpoints = sum(1 for r in test_results if r.get("meets_ultraperformance"))
        total_endpoints = len(test_results)
        
        logger.info(f"\n🎯 ULTRAPERFORMANCE Targets:")
        logger.info(f"   Endpoints Meeting <50ms Target: {passing_endpoints}/{total_endpoints}")
        
        # Endpoint breakdown
        logger.info(f"\n📊 Endpoint Performance Summary:")
        for result in test_results:
            if "response_times" in result:
                endpoint = result["endpoint"]
                median = result["response_times"]["median_ms"]
                status = "✅" if result.get("meets_ultraperformance") else "❌"
                cache_rate = result.get("cache_hit_rate", 0)
                logger.info(f"   {status} {endpoint}: {median:.2f}ms median, {cache_rate:.1f}% cache hits")
        
        # Recommendations
        logger.info(f"\n💡 Recommendations:")
        slow_endpoints = [r for r in test_results if not r.get("meets_ultraperformance")]
        
        if not slow_endpoints:
            logger.info("   ✅ All endpoints meet ULTRAPERFORMANCE targets!")
            logger.info("   ✅ System is ready for production load")
        else:
            logger.info("   ⚠️ The following endpoints need optimization:")
            for result in slow_endpoints:
                endpoint = result.get("endpoint", "Unknown")
                median = result.get("response_times", {}).get("median_ms", 0)
                logger.info(f"      - {endpoint}: {median:.2f}ms (target: <50ms)")
            
            logger.info("\n   Optimization suggestions:")
            logger.info("   1. Implement Redis caching for slow endpoints")
            logger.info("   2. Add database query optimization and indexing")
            logger.info("   3. Use connection pooling for external services")
            logger.info("   4. Implement request coalescing for duplicate requests")
            logger.info("   5. Add CDN caching for static content")
        
        # Save report to file
        report_file = f"performance_report_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump({
                "summary": {
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "failed_requests": failed_requests,
                    "passing_endpoints": passing_endpoints,
                    "total_endpoints": total_endpoints
                },
                "test_results": test_results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        logger.info(f"\n📁 Full report saved to: {report_file}")


async def main():
    """Run the performance test suite"""
    tester = UltraPerformanceTest()
    
    # Check if backend is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:10010/health") as response:
                if response.status != 200:
                    logger.info("❌ Backend is not healthy!")
                    return
    except:
        logger.info("❌ Backend is not running! Start it with: docker-compose up -d")
        return
    
    # Run comprehensive test
    await tester.run_comprehensive_test()


if __name__ == "__main__":
    asyncio.run(main())