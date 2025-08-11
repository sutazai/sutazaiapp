#!/usr/bin/env python3
"""
ULTRADEEP Load Test for Backend Health Endpoint
Tests health endpoint under extreme load to identify timeout issues
"""

import asyncio
import aiohttp
import time
import statistics
from typing import List, Tuple

async def test_health_endpoint(session: aiohttp.ClientSession, test_id: int) -> Tuple[int, float, str]:
    """Test a single health endpoint request"""
    start_time = time.time()
    try:
        timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
        async with session.get('http://localhost:10010/health', timeout=timeout) as response:
            response_text = await response.text()
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            return response.status, response_time_ms, "SUCCESS"
    except asyncio.TimeoutError:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        return 0, response_time_ms, "TIMEOUT"
    except Exception as e:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        return 0, response_time_ms, f"ERROR: {str(e)}"

async def run_load_test(concurrent_requests: int, total_requests: int):
    """Run load test with specified parameters"""
    print(f"🔥 ULTRADEEP Load Test Starting:")
    print(f"   Concurrent requests: {concurrent_requests}")
    print(f"   Total requests: {total_requests}")
    print(f"   Target: <50ms response time with 100% success rate")
    print("-" * 60)
    
    # Create connection pool with proper limits
    connector = aiohttp.TCPConnector(
        limit=concurrent_requests * 2,
        limit_per_host=concurrent_requests * 2,
        ttl_dns_cache=300,
        use_dns_cache=True,
    )
    
    timeout = aiohttp.ClientTimeout(total=10)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        start_time = time.time()
        
        # Create batches of concurrent requests
        results = []
        batch_size = concurrent_requests
        
        for batch_start in range(0, total_requests, batch_size):
            batch_end = min(batch_start + batch_size, total_requests)
            batch_tasks = []
            
            print(f"🚀 Running batch {batch_start//batch_size + 1} ({batch_end - batch_start} requests)...")
            
            # Create tasks for this batch
            for i in range(batch_start, batch_end):
                task = test_health_endpoint(session, i + 1)
                batch_tasks.append(task)
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            success_count = 0
            timeout_count = 0
            error_count = 0
            response_times = []
            
            for result in batch_results:
                if isinstance(result, Exception):
                    error_count += 1
                    print(f"❌ Exception: {result}")
                else:
                    status_code, response_time, status = result
                    response_times.append(response_time)
                    
                    if status == "SUCCESS" and status_code == 200:
                        success_count += 1
                    elif status == "TIMEOUT":
                        timeout_count += 1
                        print(f"⏰ TIMEOUT after {response_time:.2f}ms")
                    else:
                        error_count += 1
                        print(f"❌ ERROR: {status} (status: {status_code}, time: {response_time:.2f}ms)")
            
            results.extend(batch_results)
            
            # Batch summary
            if response_times:
                avg_time = statistics.mean(response_times)
                max_time = max(response_times)
                min_time = min(response_times)
                print(f"   ✅ Success: {success_count}, ⏰ Timeouts: {timeout_count}, ❌ Errors: {error_count}")
                print(f"   📊 Times: avg={avg_time:.2f}ms, min={min_time:.2f}ms, max={max_time:.2f}ms")
            
            # Brief pause between batches
            if batch_end < total_requests:
                await asyncio.sleep(0.1)
    
    end_time = time.time()
    total_test_time = end_time - start_time
    
    # Calculate final statistics
    success_count = 0
    timeout_count = 0
    error_count = 0
    response_times = []
    
    for result in results:
        if isinstance(result, Exception):
            error_count += 1
        else:
            status_code, response_time, status = result
            response_times.append(response_time)
            
            if status == "SUCCESS" and status_code == 200:
                success_count += 1
            elif status == "TIMEOUT":
                timeout_count += 1
            else:
                error_count += 1
    
    # Final report
    print("=" * 60)
    print("🎯 ULTRADEEP LOAD TEST RESULTS")
    print("=" * 60)
    print(f"Total Requests: {total_requests}")
    print(f"✅ Successful: {success_count} ({success_count/total_requests*100:.1f}%)")
    print(f"⏰ Timeouts: {timeout_count} ({timeout_count/total_requests*100:.1f}%)")
    print(f"❌ Errors: {error_count} ({error_count/total_requests*100:.1f}%)")
    print(f"🕒 Total Test Time: {total_test_time:.2f} seconds")
    print(f"🚀 Requests/Second: {total_requests/total_test_time:.2f}")
    
    if response_times:
        avg_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        p95_time = sorted(response_times)[int(0.95 * len(response_times))]
        p99_time = sorted(response_times)[int(0.99 * len(response_times))]
        max_time = max(response_times)
        min_time = min(response_times)
        
        print("-" * 40)
        print("📈 RESPONSE TIME ANALYSIS")
        print(f"Average: {avg_time:.2f}ms")
        print(f"Median:  {median_time:.2f}ms")
        print(f"P95:     {p95_time:.2f}ms")
        print(f"P99:     {p99_time:.2f}ms")
        print(f"Max:     {max_time:.2f}ms")
        print(f"Min:     {min_time:.2f}ms")
        
        # Performance assessment
        print("-" * 40)
        print("🎯 PERFORMANCE ASSESSMENT")
        if success_count == total_requests and avg_time < 50:
            print("🟢 PERFECT: 100% success rate with <50ms average response")
        elif success_count == total_requests and avg_time < 100:
            print("🟡 GOOD: 100% success rate with <100ms average response")
        elif success_count >= total_requests * 0.99 and avg_time < 200:
            print("🟡 ACCEPTABLE: >99% success rate with <200ms average response")
        else:
            print("🔴 NEEDS IMPROVEMENT: Below target performance")
            
        # Identify timeout pattern
        if timeout_count > 0:
            print(f"⚠️  TIMEOUT PATTERN DETECTED: {timeout_count} requests timed out")
            print("🔧 RECOMMENDATION: Optimize health endpoint for better concurrency")
        
        if error_count > 0:
            print(f"⚠️  ERROR PATTERN DETECTED: {error_count} requests failed")
            print("🔧 RECOMMENDATION: Add better error handling and resilience")

async def main():
    """Main load test execution"""
    
    # Test 1: Light load
    print("🧪 Test 1: Light Load (10 concurrent, 50 total)")
    await run_load_test(concurrent_requests=10, total_requests=50)
    await asyncio.sleep(2)
    
    # Test 2: Medium load  
    print("\n🧪 Test 2: Medium Load (25 concurrent, 100 total)")
    await run_load_test(concurrent_requests=25, total_requests=100)
    await asyncio.sleep(2)
    
    # Test 3: Heavy load
    print("\n🧪 Test 3: Heavy Load (50 concurrent, 200 total)")
    await run_load_test(concurrent_requests=50, total_requests=200)
    await asyncio.sleep(2)
    
    # Test 4: Extreme load
    print("\n🧪 Test 4: Extreme Load (100 concurrent, 500 total)")
    await run_load_test(concurrent_requests=100, total_requests=500)

if __name__ == "__main__":
    print("🚀 ULTRADEEP Backend Health Endpoint Load Test")
    print("=" * 60)
    asyncio.run(main())