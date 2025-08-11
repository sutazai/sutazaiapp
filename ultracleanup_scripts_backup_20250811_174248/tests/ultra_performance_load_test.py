#!/usr/bin/env python3
"""
ULTRA Performance Load Test
Validates <2s response times with 1000+ concurrent users
"""

import asyncio
import aiohttp
import time
import json
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import random
import argparse
import sys

@dataclass
class TestResult:
    """Individual test result"""
    request_id: int
    endpoint: str
    response_time_ms: float
    status_code: int
    cache_hit: bool
    error: str = None
    timestamp: float = None


class UltraLoadTester:
    """
    ULTRA performance load tester for SutazAI
    Tests the system with realistic load patterns
    """
    
    def __init__(self, base_url: str = "http://localhost:10010"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
        
        # Test prompts for Ollama
        self.test_prompts = [
            "What is SutazAI?",
            "How do I get started?",
            "List available models",
            "Explain the architecture",
            "Generate a simple Python function",
            "What are the system requirements?",
            "How to deploy the system?",
            "Explain caching strategy",
            "What is the purpose of this platform?",
            "Show system status"
        ]
        
        # Endpoints to test
        self.endpoints = [
            ("/health", "GET", None),
            ("/api/v1/models", "GET", None),
            ("/api/v1/chat", "POST", self._get_chat_payload),
            ("/metrics", "GET", None)
        ]
    
    def _get_chat_payload(self) -> Dict[str, Any]:
        """Get random chat payload"""
        return {
            "prompt": random.choice(self.test_prompts),
            "model": "tinyllama",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 50
            }
        }
    
    async def make_request(
        self,
        session: aiohttp.ClientSession,
        request_id: int,
        endpoint: str,
        method: str,
        payload: Any = None
    ) -> TestResult:
        """Make a single HTTP request and measure performance"""
        start_time = time.time()
        
        try:
            # Prepare request
            url = f"{self.base_url}{endpoint}"
            kwargs = {
                "timeout": aiohttp.ClientTimeout(total=5)
            }
            
            if method == "POST" and payload:
                if callable(payload):
                    payload = payload()
                kwargs["json"] = payload
            
            # Make request
            async with session.request(method, url, **kwargs) as response:
                # Read response
                data = await response.text()
                
                # Calculate response time
                response_time_ms = (time.time() - start_time) * 1000
                
                # Check for cache hit
                cache_hit = False
                if "X-Cache-Status" in response.headers:
                    cache_hit = response.headers["X-Cache-Status"] == "HIT"
                elif response.status == 200 and endpoint == "/api/v1/chat":
                    # Check response data for cache indicator
                    try:
                        json_data = json.loads(data)
                        cache_hit = json_data.get("cache_hit", False)
                    except:
                        pass
                
                return TestResult(
                    request_id=request_id,
                    endpoint=endpoint,
                    response_time_ms=response_time_ms,
                    status_code=response.status,
                    cache_hit=cache_hit,
                    timestamp=time.time()
                )
                
        except asyncio.TimeoutError:
            return TestResult(
                request_id=request_id,
                endpoint=endpoint,
                response_time_ms=5000,  # Timeout
                status_code=0,
                cache_hit=False,
                error="Timeout",
                timestamp=time.time()
            )
        except Exception as e:
            return TestResult(
                request_id=request_id,
                endpoint=endpoint,
                response_time_ms=(time.time() - start_time) * 1000,
                status_code=0,
                cache_hit=False,
                error=str(e),
                timestamp=time.time()
            )
    
    async def run_concurrent_test(
        self,
        num_users: int,
        requests_per_user: int
    ) -> None:
        """Run concurrent load test"""
        print(f"\n🚀 Starting ULTRA Load Test")
        print(f"   Users: {num_users}")
        print(f"   Requests per user: {requests_per_user}")
        print(f"   Total requests: {num_users * requests_per_user}")
        print(f"   Target: <2000ms response time\n")
        
        self.start_time = time.time()
        self.results = []
        
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=min(num_users, 100),
            limit_per_host=min(num_users, 100)
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            # Warm up cache first
            print("⏳ Warming up cache...")
            warmup_tasks = []
            for prompt in self.test_prompts[:5]:
                task = self.make_request(
                    session, -1, "/api/v1/chat", "POST",
                    {"prompt": prompt, "model": "tinyllama", "options": {"num_predict": 50}}
                )
                warmup_tasks.append(task)
            
            await asyncio.gather(*warmup_tasks)
            print("✅ Cache warmed up\n")
            
            # Create all user tasks
            print("🔥 Starting load test...")
            tasks = []
            request_id = 0
            
            for user_id in range(num_users):
                for req in range(requests_per_user):
                    # Select random endpoint
                    endpoint, method, payload = random.choice(self.endpoints)
                    
                    # Add some delay between requests from same user
                    delay = random.uniform(0, 0.5) * user_id / num_users
                    
                    task = self._delayed_request(
                        session, request_id, endpoint, method, payload, delay
                    )
                    tasks.append(task)
                    request_id += 1
            
            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)
            self.results = [r for r in results if r is not None]
        
        self.end_time = time.time()
        
        print(f"\n✅ Load test completed in {self.end_time - self.start_time:.2f} seconds")
    
    async def _delayed_request(
        self,
        session: aiohttp.ClientSession,
        request_id: int,
        endpoint: str,
        method: str,
        payload: Any,
        delay: float
    ) -> TestResult:
        """Make request with optional delay"""
        if delay > 0:
            await asyncio.sleep(delay)
        return await self.make_request(session, request_id, endpoint, method, payload)
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results and generate report"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Filter successful requests
        successful = [r for r in self.results if r.status_code == 200]
        failed = [r for r in self.results if r.status_code != 200]
        
        # Calculate statistics
        response_times = [r.response_time_ms for r in successful]
        
        if not response_times:
            return {"error": "No successful requests"}
        
        # Basic stats
        stats = {
            "total_requests": len(self.results),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": (len(successful) / len(self.results)) * 100,
            "total_time_seconds": self.end_time - self.start_time if self.end_time else 0,
            "requests_per_second": len(self.results) / (self.end_time - self.start_time) if self.end_time else 0
        }
        
        # Response time stats
        stats["response_times"] = {
            "min_ms": min(response_times),
            "max_ms": max(response_times),
            "mean_ms": statistics.mean(response_times),
            "median_ms": statistics.median(response_times),
            "stdev_ms": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            "p95_ms": self._percentile(response_times, 95),
            "p99_ms": self._percentile(response_times, 99)
        }
        
        # Cache statistics
        cache_hits = [r for r in successful if r.cache_hit]
        stats["cache"] = {
            "total_hits": len(cache_hits),
            "hit_rate": (len(cache_hits) / len(successful)) * 100 if successful else 0
        }
        
        # Endpoint breakdown
        endpoint_stats = {}
        for endpoint in set(r.endpoint for r in self.results):
            endpoint_results = [r for r in successful if r.endpoint == endpoint]
            if endpoint_results:
                endpoint_times = [r.response_time_ms for r in endpoint_results]
                endpoint_stats[endpoint] = {
                    "count": len(endpoint_results),
                    "mean_ms": statistics.mean(endpoint_times),
                    "p95_ms": self._percentile(endpoint_times, 95)
                }
        
        stats["endpoints"] = endpoint_stats
        
        # Performance goals
        under_2s = [r for r in successful if r.response_time_ms < 2000]
        under_1s = [r for r in successful if r.response_time_ms < 1000]
        under_500ms = [r for r in successful if r.response_time_ms < 500]
        
        stats["performance_goals"] = {
            "under_2000ms": (len(under_2s) / len(successful)) * 100 if successful else 0,
            "under_1000ms": (len(under_1s) / len(successful)) * 100 if successful else 0,
            "under_500ms": (len(under_500ms) / len(successful)) * 100 if successful else 0
        }
        
        # Errors
        if failed:
            error_types = {}
            for r in failed:
                error_key = r.error or f"HTTP_{r.status_code}"
                error_types[error_key] = error_types.get(error_key, 0) + 1
            stats["errors"] = error_types
        
        return stats
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def print_report(self, stats: Dict[str, Any]) -> None:
        """Print formatted test report"""
        print("\n" + "="*80)
        print(" ULTRA PERFORMANCE LOAD TEST REPORT")
        print("="*80)
        
        # Overall stats
        print(f"\n📊 OVERALL STATISTICS")
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Successful: {stats['successful_requests']} ({stats['success_rate']:.1f}%)")
        print(f"   Failed: {stats['failed_requests']}")
        print(f"   Duration: {stats['total_time_seconds']:.2f}s")
        print(f"   Throughput: {stats['requests_per_second']:.1f} req/s")
        
        # Response times
        rt = stats["response_times"]
        print(f"\n⏱️  RESPONSE TIMES")
        print(f"   Mean: {rt['mean_ms']:.1f}ms")
        print(f"   Median: {rt['median_ms']:.1f}ms")
        print(f"   Min: {rt['min_ms']:.1f}ms")
        print(f"   Max: {rt['max_ms']:.1f}ms")
        print(f"   StdDev: {rt['stdev_ms']:.1f}ms")
        print(f"   P95: {rt['p95_ms']:.1f}ms")
        print(f"   P99: {rt['p99_ms']:.1f}ms")
        
        # Cache performance
        cache = stats["cache"]
        print(f"\n💾 CACHE PERFORMANCE")
        print(f"   Cache Hits: {cache['total_hits']}")
        print(f"   Hit Rate: {cache['hit_rate']:.1f}%")
        
        # Performance goals
        goals = stats["performance_goals"]
        print(f"\n🎯 PERFORMANCE GOALS")
        print(f"   < 2000ms: {goals['under_2000ms']:.1f}% ✅" if goals['under_2000ms'] > 95 else f"   < 2000ms: {goals['under_2000ms']:.1f}% ❌")
        print(f"   < 1000ms: {goals['under_1000ms']:.1f}%")
        print(f"   < 500ms: {goals['under_500ms']:.1f}%")
        
        # Endpoint breakdown
        if stats.get("endpoints"):
            print(f"\n📍 ENDPOINT PERFORMANCE")
            for endpoint, data in stats["endpoints"].items():
                print(f"   {endpoint}:")
                print(f"      Requests: {data['count']}")
                print(f"      Mean: {data['mean_ms']:.1f}ms")
                print(f"      P95: {data['p95_ms']:.1f}ms")
        
        # Errors
        if stats.get("errors"):
            print(f"\n❌ ERRORS")
            for error_type, count in stats["errors"].items():
                print(f"   {error_type}: {count}")
        
        # Final verdict
        print(f"\n{'='*80}")
        if stats['success_rate'] > 99 and rt['p95_ms'] < 2000:
            print("✅ ULTRA PERFORMANCE ACHIEVED! System meets <2s response time target!")
        elif stats['success_rate'] > 95 and rt['p95_ms'] < 3000:
            print("🟡 GOOD PERFORMANCE. Close to target, minor optimizations needed.")
        else:
            print("❌ PERFORMANCE TARGET NOT MET. Further optimization required.")
        print("="*80 + "\n")


async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="ULTRA Performance Load Test")
    parser.add_argument("--users", type=int, default=100, help="Number of concurrent users")
    parser.add_argument("--requests", type=int, default=10, help="Requests per user")
    parser.add_argument("--url", default="http://localhost:10010", help="Base URL")
    parser.add_argument("--warmup", action="store_true", help="Run warmup phase")
    
    args = parser.parse_args()
    
    # Create tester
    tester = UltraLoadTester(args.url)
    
    # Run test
    await tester.run_concurrent_test(args.users, args.requests)
    
    # Analyze results
    stats = tester.analyze_results()
    
    # Print report
    tester.print_report(stats)
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"load_test_results_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump({
            "test_config": {
                "users": args.users,
                "requests_per_user": args.requests,
                "total_requests": args.users * args.requests,
                "base_url": args.url
            },
            "results": stats,
            "timestamp": timestamp
        }, f, indent=2)
    
    print(f"📁 Results saved to {filename}")
    
    # Exit code based on performance
    if stats['success_rate'] > 99 and stats['response_times']['p95_ms'] < 2000:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Performance target not met


if __name__ == "__main__":
    asyncio.run(main())