#!/usr/bin/env python3
"""
Performance test for optimized health endpoint
Target: <200ms response time
"""

import asyncio
import time
import statistics
import httpx
from typing import List
import sys

# Configuration
HEALTH_ENDPOINT = "http://localhost:10010/health"
NUM_REQUESTS = 50  # Number of requests to make
CONCURRENT_REQUESTS = 5  # Number of concurrent requests
TARGET_RESPONSE_TIME_MS = 200  # Target response time in milliseconds


async def measure_health_endpoint(client: httpx.AsyncClient) -> float:
    """Measure single health endpoint response time"""
    start_time = time.perf_counter()
    
    try:
        response = await client.get(HEALTH_ENDPOINT, timeout=10.0)
        response.raise_for_status()
        
        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000
        
        return response_time_ms
        
    except Exception as e:
        print(f"Error during health check: {e}")
        return -1


async def run_performance_test():
    """Run performance test with multiple requests"""
    
    print("=" * 60)
    print("HEALTH ENDPOINT PERFORMANCE TEST")
    print("=" * 60)
    print(f"Target Response Time: <{TARGET_RESPONSE_TIME_MS}ms")
    print(f"Number of Requests: {NUM_REQUESTS}")
    print(f"Concurrent Requests: {CONCURRENT_REQUESTS}")
    print("-" * 60)
    
    # Create HTTP client with connection pooling
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        timeout=httpx.Timeout(10.0)
    ) as client:
        
        # Warm-up request (first request might be slower)
        print("\nWarming up...")
        warmup_time = await measure_health_endpoint(client)
        print(f"Warmup request: {warmup_time:.2f}ms")
        
        # Wait for cache to populate
        await asyncio.sleep(1)
        
        # Run performance test
        print(f"\nRunning {NUM_REQUESTS} requests...")
        response_times: List[float] = []
        
        # Create batches of concurrent requests
        for batch_start in range(0, NUM_REQUESTS, CONCURRENT_REQUESTS):
            batch_end = min(batch_start + CONCURRENT_REQUESTS, NUM_REQUESTS)
            batch_size = batch_end - batch_start
            
            # Run batch concurrently
            tasks = [measure_health_endpoint(client) for _ in range(batch_size)]
            batch_results = await asyncio.gather(*tasks)
            
            # Filter out errors (-1 values)
            valid_results = [r for r in batch_results if r > 0]
            response_times.extend(valid_results)
            
            # Show progress
            print(f"  Completed {len(response_times)}/{NUM_REQUESTS} requests", end='\r')
        
        print(f"\n\nTest completed: {len(response_times)} successful requests")
        
    # Calculate statistics
    if response_times:
        print("\n" + "=" * 60)
        print("PERFORMANCE RESULTS")
        print("=" * 60)
        
        avg_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        stdev_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
        
        # Calculate percentiles
        sorted_times = sorted(response_times)
        p95_index = int(len(sorted_times) * 0.95)
        p99_index = int(len(sorted_times) * 0.99)
        p95_time = sorted_times[p95_index] if p95_index < len(sorted_times) else max_time
        p99_time = sorted_times[p99_index] if p99_index < len(sorted_times) else max_time
        
        # Count requests meeting target
        requests_under_target = sum(1 for t in response_times if t < TARGET_RESPONSE_TIME_MS)
        success_rate = (requests_under_target / len(response_times)) * 100
        
        print(f"Average Response Time:   {avg_time:.2f}ms")
        print(f"Median Response Time:    {median_time:.2f}ms")
        print(f"Min Response Time:       {min_time:.2f}ms")
        print(f"Max Response Time:       {max_time:.2f}ms")
        print(f"Standard Deviation:      {stdev_time:.2f}ms")
        print(f"95th Percentile:         {p95_time:.2f}ms")
        print(f"99th Percentile:         {p99_time:.2f}ms")
        print("-" * 60)
        print(f"Requests <{TARGET_RESPONSE_TIME_MS}ms:      {requests_under_target}/{len(response_times)} ({success_rate:.1f}%)")
        
        # Performance verdict
        print("\n" + "=" * 60)
        print("VERDICT")
        print("=" * 60)
        
        if avg_time < TARGET_RESPONSE_TIME_MS and p95_time < TARGET_RESPONSE_TIME_MS * 1.5:
            print(f"✅ SUCCESS: Health endpoint meets performance target!")
            print(f"   Average: {avg_time:.2f}ms < {TARGET_RESPONSE_TIME_MS}ms target")
            print(f"   P95: {p95_time:.2f}ms < {TARGET_RESPONSE_TIME_MS * 1.5:.0f}ms threshold")
            return 0
        else:
            print(f"❌ FAILED: Health endpoint does not meet performance target")
            print(f"   Average: {avg_time:.2f}ms (target: <{TARGET_RESPONSE_TIME_MS}ms)")
            print(f"   P95: {p95_time:.2f}ms (threshold: <{TARGET_RESPONSE_TIME_MS * 1.5:.0f}ms)")
            return 1
    else:
        print("\n❌ ERROR: No successful requests completed")
        return 1


async def test_cache_effectiveness():
    """Test that caching is working effectively"""
    
    print("\n" + "=" * 60)
    print("CACHE EFFECTIVENESS TEST")
    print("=" * 60)
    
    async with httpx.AsyncClient() as client:
        # First request (cache miss)
        print("Testing cache behavior...")
        
        start = time.perf_counter()
        response1 = await client.get(HEALTH_ENDPOINT)
        time1 = (time.perf_counter() - start) * 1000
        
        # Immediate second request (should be cached)
        start = time.perf_counter()
        response2 = await client.get(HEALTH_ENDPOINT)
        time2 = (time.perf_counter() - start) * 1000
        
        # Wait for cache to expire (31 seconds)
        print("Waiting 31 seconds for cache expiry...")
        await asyncio.sleep(31)
        
        # Request after cache expiry
        start = time.perf_counter()
        response3 = await client.get(HEALTH_ENDPOINT)
        time3 = (time.perf_counter() - start) * 1000
        
        print(f"First request (cache miss):     {time1:.2f}ms")
        print(f"Second request (cached):         {time2:.2f}ms")
        print(f"After cache expiry:              {time3:.2f}ms")
        
        # Cache should make second request much faster
        if time2 < time1 * 0.5:  # Second request should be at least 50% faster
            print("✅ Cache is working effectively!")
        else:
            print("⚠️  Cache may not be working optimally")


async def main():
    """Main test runner"""
    
    # Check if backend is running
    print("Checking if backend is accessible...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(HEALTH_ENDPOINT, timeout=5.0)
            print(f"✅ Backend is running (status: {response.status_code})\n")
    except Exception as e:
        print(f"❌ Cannot connect to backend at {HEALTH_ENDPOINT}")
        print(f"   Error: {e}")
        print("\nPlease ensure the backend is running:")
        print("  docker-compose up -d")
        return 1
    
    # Run performance test
    result = await run_performance_test()
    
    # Optionally run cache test (adds 31+ seconds)
    # await test_cache_effectiveness()
    
    return result


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)