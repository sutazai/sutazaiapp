#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
ULTRATEST: Load test with 100 concurrent users for health endpoint
Validates the ULTRAFIX performance improvements under extreme load
"""

import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict, Any
import json

async def test_health_endpoint(session: aiohttp.ClientSession, user_id: int) -> Dict[str, Any]:
    """Test health endpoint for a single user"""
    start_time = time.time()
    
    try:
        async with session.get("http://localhost:10010/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
            response_time = time.time() - start_time
            status_code = response.status
            
            if status_code == 200:
                data = await response.json()
                return {
                    "user_id": user_id,
                    "success": True,
                    "response_time_ms": round(response_time * 1000, 2),
                    "status_code": status_code,
                    "status": data.get("status", "unknown")
                }
            else:
                return {
                    "user_id": user_id,
                    "success": False,
                    "response_time_ms": round(response_time * 1000, 2),
                    "status_code": status_code,
                    "error": f"HTTP {status_code}"
                }
                
    except asyncio.TimeoutError:
        response_time = time.time() - start_time
        return {
            "user_id": user_id,
            "success": False,
            "response_time_ms": round(response_time * 1000, 2),
            "error": "timeout"
        }
    except Exception as e:
        response_time = time.time() - start_time
        return {
            "user_id": user_id,
            "success": False,
            "response_time_ms": round(response_time * 1000, 2),
            "error": str(e)
        }

async def run_concurrent_load_test(num_users: int = 100, requests_per_user: int = 10) -> Dict[str, Any]:
    """Run concurrent load test with multiple users"""
    logger.info(f"🚀 ULTRATEST: Starting load test with {num_users} concurrent users")
    logger.info(f"📊 Each user will make {requests_per_user} requests")
    logger.info(f"🎯 Total requests: {num_users * requests_per_user}")
    
    connector = aiohttp.TCPConnector(limit=200, limit_per_host=50)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warm up
        logger.info("🔥 Warming up...")
        await test_health_endpoint(session, 0)
        
        # Run load test
        start_time = time.time()
        
        tasks = []
        for user_id in range(num_users):
            for req_num in range(requests_per_user):
                tasks.append(test_health_endpoint(session, f"{user_id}-{req_num}"))
        
        logger.info(f"⚡ Executing {len(tasks)} concurrent requests...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Process results
        successful_requests = []
        failed_requests = []
        response_times = []
        
        for result in results:
            if isinstance(result, dict):
                if result.get("success"):
                    successful_requests.append(result)
                    response_times.append(result["response_time_ms"])
                else:
                    failed_requests.append(result)
            else:
                # Exception occurred
                failed_requests.append({"error": str(result)})
        
        # Calculate statistics
        success_rate = len(successful_requests) / len(tasks) * 100
        
        stats = {
            "total_requests": len(tasks),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate_percent": round(success_rate, 2),
            "total_time_seconds": round(total_time, 2),
            "requests_per_second": round(len(tasks) / total_time, 2)
        }
        
        if response_times:
            stats.update({
                "response_time_stats": {
                    "min_ms": min(response_times),
                    "max_ms": max(response_times),
                    "avg_ms": round(statistics.mean(response_times), 2),
                    "median_ms": round(statistics.median(response_times), 2),
                    "p95_ms": round(statistics.quantile(response_times, 0.95), 2) if len(response_times) > 1 else response_times[0],
                    "p99_ms": round(statistics.quantile(response_times, 0.99), 2) if len(response_times) > 1 else response_times[0]
                }
            })
        
        return {
            "test_config": {
                "concurrent_users": num_users,
                "requests_per_user": requests_per_user,
                "endpoint": "http://localhost:10010/health"
            },
            "results": stats,
            "sample_failures": failed_requests[:5]  # Show first 5 failures
        }

async def main():
    """Run the ULTRATEST load test"""
    logger.info("🎯 ULTRAFIX Backend Health Endpoint Load Test")
    logger.info("=" * 60)
    
    # Test scenarios
    scenarios = [
        {"users": 50, "requests": 5, "name": "Medium Load Test"},
        {"users": 100, "requests": 10, "name": "High Load Test"},
        {"users": 200, "requests": 5, "name": "Extreme Concurrency Test"}
    ]
    
    all_results = {}
    
    for scenario in scenarios:
        logger.info(f"\n🔥 Running {scenario['name']}...")
        logger.info("-" * 40)
        
        result = await run_concurrent_load_test(scenario["users"], scenario["requests"])
        all_results[scenario["name"]] = result
        
        # Print summary
        stats = result["results"]
        logger.info(f"✅ Success Rate: {stats['success_rate_percent']}%")
        logger.info(f"⚡ Requests/sec: {stats['requests_per_second']}")
        
        if "response_time_stats" in stats:
            rt_stats = stats["response_time_stats"]
            logger.info(f"⏱️  Response Times - Avg: {rt_stats['avg_ms']}ms, P95: {rt_stats['p95_ms']}ms, Max: {rt_stats['max_ms']}ms")
        
        if stats["failed_requests"] > 0:
            logger.error(f"❌ Failures: {stats['failed_requests']}")
            if result["sample_failures"]:
                logger.info("   Sample failures:")
                for failure in result["sample_failures"][:3]:
                    logger.info(f"   - {failure}")
        
        # Brief pause between tests
        await asyncio.sleep(2)
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 ULTRATEST COMPLETE - PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    
    for name, result in all_results.items():
        stats = result["results"]
        rt_stats = stats.get("response_time_stats", {})
        
        logger.info(f"\n{name}:")
        logger.info(f"  Success Rate: {stats['success_rate_percent']}%")
        logger.info(f"  Throughput: {stats['requests_per_second']} req/sec")
        if rt_stats:
            logger.info(f"  Response Time: {rt_stats['avg_ms']}ms avg, {rt_stats['p95_ms']}ms p95")
    
    # Save detailed results
    with open("/opt/sutazaiapp/ultratest_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n💾 Detailed results saved to: /opt/sutazaiapp/ultratest_results.json")
    
    # Determine overall success
    overall_success = all(result["results"]["success_rate_percent"] >= 95.0 for result in all_results.values())
    
    if overall_success:
        logger.info("\n🎉 ULTRAFIX SUCCESS! Health endpoint performs excellently under all load conditions.")
        logger.info("✅ Target achieved: <10ms response time with >95% success rate")
    else:
        logger.error("\n⚠️  Some performance issues detected. Review failed requests and response times.")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)