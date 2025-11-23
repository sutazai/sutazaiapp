"""
Load Testing Script for Sutazai Backend
Tests various concurrency levels and measures performance metrics
Requires: pytest-asyncio, aiohttp
"""

import asyncio
import aiohttp
import time
import statistics
import json
from typing import List, Dict, Any
from datetime import datetime
import sys


class LoadTester:
    """Load testing utility for API endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []
        
    async def test_endpoint(
        self,
        session: aiohttp.ClientSession,
        method: str,
        path: str,
        data: Dict = None,
        headers: Dict = None
    ) -> Dict[str, Any]:
        """Test single API request and measure metrics"""
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{path}"
            
            if method.upper() == "GET":
                async with session.get(url, headers=headers) as response:
                    response_time = time.time() - start_time
                    status = response.status
                    try:
                        content = await response.json()
                    except:
                        content = await response.text()
            else:
                async with session.post(url, json=data, headers=headers) as response:
                    response_time = time.time() - start_time
                    status = response.status
                    try:
                        content = await response.json()
                    except:
                        content = await response.text()
            
            return {
                "success": 200 <= status < 400,
                "status_code": status,
                "response_time": response_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "success": False,
                "status_code": 0,
                "response_time": time.time() - start_time,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def concurrent_requests(
        self,
        num_users: int,
        requests_per_user: int,
        method: str,
        path: str,
        data: Dict = None
    ) -> List[Dict[str, Any]]:
        """Execute concurrent requests from multiple simulated users"""
        results = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for user_id in range(num_users):
                for req_num in range(requests_per_user):
                    task = self.test_endpoint(session, method, path, data)
                    tasks.append(task)
            
            # Execute all requests concurrently
            results = await asyncio.gather(*tasks)
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test results and calculate metrics"""
        response_times = [r["response_time"] for r in results]
        successes = [r for r in results if r["success"]]
        failures = [r for r in results if not r["success"]]
        
        if not response_times:
            return {"error": "No results to analyze"}
        
        # Calculate percentiles
        sorted_times = sorted(response_times)
        p50 = sorted_times[int(len(sorted_times) * 0.5)]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]
        
        # Calculate throughput
        total_time = max([r["timestamp"] for r in results]) - min([r["timestamp"] for r in results])
        throughput = len(results) / total_time if total_time > 0 else 0
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successes),
            "failed_requests": len(failures),
            "success_rate": len(successes) / len(results) * 100,
            "response_times": {
                "min": min(response_times),
                "max": max(response_times),
                "mean": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "p50": p50,
                "p95": p95,
                "p99": p99,
                "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0
            },
            "throughput_rps": throughput,
            "total_duration": total_time,
            "errors": [r.get("error") for r in failures if "error" in r]
        }
    
    async def run_load_test(
        self,
        test_name: str,
        num_users: int,
        requests_per_user: int,
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run complete load test with multiple scenarios"""
        print(f"\n{'='*60}")
        print(f"Load Test: {test_name}")
        print(f"Concurrent Users: {num_users}")
        print(f"Requests per User: {requests_per_user}")
        print(f"Total Requests: {num_users * requests_per_user * len(scenarios)}")
        print(f"{'='*60}\n")
        
        test_results = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "num_users": num_users,
                "requests_per_user": requests_per_user,
                "total_scenarios": len(scenarios)
            },
            "scenarios": []
        }
        
        for scenario in scenarios:
            scenario_name = scenario["name"]
            print(f"Running scenario: {scenario_name}...")
            
            start_time = time.time()
            results = await self.concurrent_requests(
                num_users=num_users,
                requests_per_user=requests_per_user,
                method=scenario["method"],
                path=scenario["path"],
                data=scenario.get("data")
            )
            duration = time.time() - start_time
            
            analysis = self.analyze_results(results)
            analysis["scenario_name"] = scenario_name
            analysis["scenario_duration"] = duration
            
            test_results["scenarios"].append(analysis)
            
            # Print results
            print(f"  ✓ Completed in {duration:.2f}s")
            print(f"  Success Rate: {analysis['success_rate']:.1f}%")
            print(f"  Mean Response Time: {analysis['response_times']['mean']*1000:.1f}ms")
            print(f"  P95 Response Time: {analysis['response_times']['p95']*1000:.1f}ms")
            print(f"  Throughput: {analysis['throughput_rps']:.1f} req/s")
            if analysis['failed_requests'] > 0:
                print(f"  ⚠ Failures: {analysis['failed_requests']}")
            print()
        
        # Overall summary
        total_requests = sum(s["total_requests"] for s in test_results["scenarios"])
        total_successes = sum(s["successful_requests"] for s in test_results["scenarios"])
        overall_success_rate = (total_successes / total_requests * 100) if total_requests > 0 else 0
        
        test_results["summary"] = {
            "total_requests": total_requests,
            "total_successes": total_successes,
            "overall_success_rate": overall_success_rate,
            "passed": overall_success_rate >= 95.0  # 95% success threshold
        }
        
        print(f"{'='*60}")
        print(f"Test Summary:")
        print(f"  Total Requests: {total_requests}")
        print(f"  Success Rate: {overall_success_rate:.1f}%")
        print(f"  Status: {'✓ PASSED' if test_results['summary']['passed'] else '✗ FAILED'}")
        print(f"{'='*60}\n")
        
        return test_results


# Test scenarios
async def test_10_concurrent_users():
    """Load test with 10 concurrent users"""
    tester = LoadTester()
    
    scenarios = [
        {
            "name": "Health Check",
            "method": "GET",
            "path": "/api/v1/health/"
        },
        {
            "name": "Health Services",
            "method": "GET",
            "path": "/api/v1/health/services"
        },
        {
            "name": "Health Metrics",
            "method": "GET",
            "path": "/api/v1/health/metrics"
        },
        {
            "name": "List Agents",
            "method": "GET",
            "path": "/api/v1/agents/"
        },
    ]
    
    results = await tester.run_load_test(
        test_name="10 Concurrent Users Test",
        num_users=10,
        requests_per_user=5,
        scenarios=scenarios
    )
    
    return results


async def test_50_concurrent_users():
    """Load test with 50 concurrent users"""
    tester = LoadTester()
    
    scenarios = [
        {
            "name": "Health Check",
            "method": "GET",
            "path": "/api/v1/health/"
        },
        {
            "name": "Health Metrics",
            "method": "GET",
            "path": "/api/v1/health/metrics"
        },
        {
            "name": "List Agents",
            "method": "GET",
            "path": "/api/v1/agents/"
        },
    ]
    
    results = await tester.run_load_test(
        test_name="50 Concurrent Users Test",
        num_users=50,
        requests_per_user=10,
        scenarios=scenarios
    )
    
    return results


async def test_100_concurrent_users():
    """Load test with 100 concurrent users"""
    tester = LoadTester()
    
    scenarios = [
        {
            "name": "Health Check",
            "method": "GET",
            "path": "/api/v1/health/"
        },
        {
            "name": "List Agents",
            "method": "GET",
            "path": "/api/v1/agents/"
        },
    ]
    
    results = await tester.run_load_test(
        test_name="100 Concurrent Users Test",
        num_users=100,
        requests_per_user=10,
        scenarios=scenarios
    )
    
    return results


async def run_all_tests():
    """Run all load tests sequentially"""
    print("\n" + "="*60)
    print("SUTAZAI BACKEND LOAD TESTING SUITE")
    print("="*60)
    
    all_results = []
    
    # Test 1: 10 users
    print("\n[1/3] Starting 10 concurrent users test...")
    results_10 = await test_10_concurrent_users()
    all_results.append(results_10)
    await asyncio.sleep(5)  # Cool down between tests
    
    # Test 2: 50 users
    print("\n[2/3] Starting 50 concurrent users test...")
    results_50 = await test_50_concurrent_users()
    all_results.append(results_50)
    await asyncio.sleep(5)
    
    # Test 3: 100 users
    print("\n[3/3] Starting 100 concurrent users test...")
    results_100 = await test_100_concurrent_users()
    all_results.append(results_100)
    
    # Save results to file
    output_file = f"load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for result in all_results:
        status = "✓ PASSED" if result["summary"]["passed"] else "✗ FAILED"
        print(f"{result['test_name']}: {status} ({result['summary']['overall_success_rate']:.1f}% success)")
    print("="*60 + "\n")
    
    return all_results


if __name__ == "__main__":
    # Run tests
    results = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    all_passed = all(r["summary"]["passed"] for r in results)
    sys.exit(0 if all_passed else 1)
