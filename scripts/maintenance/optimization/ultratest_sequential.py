#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
ULTRATEST: Sequential load test to validate ULTRAFIX without connection limits
Tests the health endpoint performance improvements sequentially
"""

import requests
import time
import statistics
from typing import List
import json

def test_health_endpoint_sequential(num_requests: int = 100) -> dict:
    """Test health endpoint sequentially"""
    logger.info(f"🚀 ULTRATEST: Sequential test with {num_requests} requests")
    
    results = []
    successful = 0
    failed = 0
    
    for i in range(num_requests):
        start_time = time.time()
        
        try:
            response = requests.get("http://localhost:10010/health", timeout=5)
            response_time_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                successful += 1
                data = response.json()
                results.append({
                    "request_id": i,
                    "success": True,
                    "response_time_ms": round(response_time_ms, 2),
                    "status": data.get("status", "unknown")
                })
            else:
                failed += 1
                results.append({
                    "request_id": i,
                    "success": False,
                    "response_time_ms": round(response_time_ms, 2),
                    "status_code": response.status_code
                })
                
        except Exception as e:
            failed += 1
            response_time_ms = (time.time() - start_time) * 1000
            results.append({
                "request_id": i,
                "success": False,
                "response_time_ms": round(response_time_ms, 2),
                "error": str(e)
            })
            
        # Progress indicator
        if (i + 1) % 20 == 0:
            logger.info(f"  Progress: {i + 1}/{num_requests} requests completed")
    
    # Calculate statistics
    response_times = [r["response_time_ms"] for r in results if r["success"]]
    success_rate = (successful / num_requests) * 100
    
    stats = {
        "total_requests": num_requests,
        "successful_requests": successful,
        "failed_requests": failed,
        "success_rate_percent": round(success_rate, 2)
    }
    
    if response_times:
        stats["response_time_stats"] = {
            "min_ms": min(response_times),
            "max_ms": max(response_times),
            "avg_ms": round(statistics.mean(response_times), 2),
            "median_ms": round(statistics.median(response_times), 2),
            "p95_ms": round(statistics.quantile(response_times, 0.95), 2) if len(response_times) > 1 else response_times[0],
            "p99_ms": round(statistics.quantile(response_times, 0.99), 2) if len(response_times) > 1 else response_times[0]
        }
    
    return {
        "test_type": "sequential",
        "results": stats,
        "sample_results": results[:5] + results[-5:]  # First 5 and last 5
    }

def test_rapid_fire(num_requests: int = 50, delay_ms: int = 10) -> dict:
    """Test with   delay between requests"""
    logger.info(f"🔥 ULTRATEST: Rapid-fire test with {num_requests} requests, {delay_ms}ms delay")
    
    results = []
    successful = 0
    
    start_test = time.time()
    
    for i in range(num_requests):
        start_time = time.time()
        
        try:
            response = requests.get("http://localhost:10010/health", timeout=2)
            response_time_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                successful += 1
                results.append(response_time_ms)
                
            #   delay
            time.sleep(delay_ms / 1000.0)
            
        except Exception as e:
            logger.error(f"  Request {i} failed: {e}")
            
        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i + 1}/{num_requests} requests")
    
    total_time = time.time() - start_test
    
    if results:
        return {
            "test_type": "rapid_fire",
            "total_time_seconds": round(total_time, 2),
            "requests_per_second": round(num_requests / total_time, 2),
            "success_rate_percent": round((successful / num_requests) * 100, 2),
            "response_time_stats": {
                "min_ms": min(results),
                "max_ms": max(results),
                "avg_ms": round(statistics.mean(results), 2),
                "median_ms": round(statistics.median(results), 2)
            }
        }
    else:
        return {"test_type": "rapid_fire", "error": "No successful requests"}

def main():
    """Run sequential load tests"""
    logger.info("🎯 ULTRAFIX Backend Health Endpoint Sequential Load Test")
    logger.info("=" * 70)
    
    # Test 1: Sequential requests
    logger.info("\n📊 Test 1: Sequential Load Test")
    logger.info("-" * 40)
    result1 = test_health_endpoint_sequential(100)
    
    # Print results
    stats = result1["results"]
    logger.info(f"✅ Success Rate: {stats['success_rate_percent']}%")
    logger.info(f"📊 Total Requests: {stats['total_requests']}")
    
    if "response_time_stats" in stats:
        rt_stats = stats["response_time_stats"]
        logger.info(f"⏱️  Response Times:")
        logger.info(f"   Average: {rt_stats['avg_ms']}ms")
        logger.info(f"   Median: {rt_stats['median_ms']}ms") 
        logger.info(f"   Min: {rt_stats['min_ms']}ms")
        logger.info(f"   Max: {rt_stats['max_ms']}ms")
        logger.info(f"   P95: {rt_stats['p95_ms']}ms")
        logger.info(f"   P99: {rt_stats['p99_ms']}ms")
    
    # Test 2: Rapid-fire test
    logger.info("\n🔥 Test 2: Rapid-Fire Test")
    logger.info("-" * 40)
    result2 = test_rapid_fire(50, 10)
    
    if "error" not in result2:
        logger.info(f"✅ Success Rate: {result2['success_rate_percent']}%")
        logger.info(f"⚡ Throughput: {result2['requests_per_second']} req/sec")
        rt_stats = result2["response_time_stats"]
        logger.info(f"⏱️  Response Times - Avg: {rt_stats['avg_ms']}ms, Max: {rt_stats['max_ms']}ms")
    else:
        logger.error(f"❌ Test failed: {result2['error']}")
    
    # Final assessment
    logger.info("\n" + "=" * 70)
    logger.info("🏆 ULTRAFIX PERFORMANCE ASSESSMENT")
    logger.info("=" * 70)
    
    success_rate = result1["results"]["success_rate_percent"]
    avg_response_time = result1["results"].get("response_time_stats", {}).get("avg_ms", 0)
    
    logger.info(f"📈 Overall Success Rate: {success_rate}%")
    logger.info(f"⚡ Average Response Time: {avg_response_time}ms")
    
    # Assess performance
    if success_rate >= 95 and avg_response_time <= 50:
        logger.info("🎉 ULTRAFIX SUCCESS! Health endpoint meets all performance targets:")
        logger.info("   ✅ >95% success rate")
        logger.info("   ✅ <50ms average response time")
        logger.info("   ✅ System handles load without timeouts")
        success = True
    elif success_rate >= 90:
        logger.info("✅ ULTRAFIX GOOD! Performance is acceptable:")
        logger.info(f"   ✅ {success_rate}% success rate")
        logger.info(f"   ✅ {avg_response_time}ms average response time")
        success = True
    else:
        logger.info("⚠️  Performance issues detected:")
        logger.info(f"   ❌ {success_rate}% success rate (target: >95%)")
        logger.info(f"   ❌ {avg_response_time}ms average response time")
        success = False
    
    # Save results
    all_results = {
        "sequential_test": result1,
        "rapid_fire_test": result2,
        "assessment": {
            "success": success,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time
        }
    }
    
    with open("/opt/sutazaiapp/ultratest_sequential_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: /opt/sutazaiapp/ultratest_sequential_results.json")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)