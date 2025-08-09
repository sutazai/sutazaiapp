#!/usr/bin/env python3
"""
Comprehensive Load Testing Suite for SutazAI Performance Optimization
Tests system capacity for 1000+ concurrent users with <200ms response time
"""

import asyncio
import time
import statistics
import json
import random
from typing import Dict, List, Any
from datetime import datetime
import httpx
import uvloop
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import pandas as pd


# Use uvloop for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class LoadTester:
    """Advanced load testing framework"""
    
    def __init__(self, base_url: str = "http://localhost:10010"):
        self.base_url = base_url
        self.results = {
            'response_times': [],
            'errors': [],
            'success_count': 0,
            'error_count': 0,
            'start_time': None,
            'end_time': None
        }
        
    async def single_request(self, client: httpx.AsyncClient, endpoint: str, method: str = "GET", data: Any = None) -> Dict[str, Any]:
        """Execute a single request and measure performance"""
        start_time = time.perf_counter()
        
        try:
            if method == "GET":
                response = await client.get(f"{self.base_url}{endpoint}")
            elif method == "POST":
                response = await client.post(f"{self.base_url}{endpoint}", json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")
                
            elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            return {
                'success': response.status_code == 200,
                'status_code': response.status_code,
                'response_time': elapsed,
                'endpoint': endpoint,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            return {
                'success': False,
                'error': str(e),
                'response_time': elapsed,
                'endpoint': endpoint,
                'timestamp': datetime.now().isoformat()
            }
            
    async def concurrent_test(
        self,
        num_users: int,
        requests_per_user: int,
        endpoints: List[Dict[str, Any]]
    ):
        """Run concurrent load test"""
        
        print(f"\n🚀 Starting load test with {num_users} concurrent users")
        print(f"   Each user will make {requests_per_user} requests")
        print(f"   Total requests: {num_users * requests_per_user}")
        
        self.results['start_time'] = datetime.now()
        
        # Create HTTP client pool
        limits = httpx.Limits(max_keepalive_connections=num_users, max_connections=num_users * 2)
        
        async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
            # Create tasks for all users
            tasks = []
            
            for user_id in range(num_users):
                user_task = self.simulate_user(client, user_id, requests_per_user, endpoints)
                tasks.append(user_task)
                
            # Execute all user simulations concurrently
            user_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in user_results:
                if isinstance(result, Exception):
                    self.results['errors'].append(str(result))
                elif isinstance(result, list):
                    for req_result in result:
                        if req_result['success']:
                            self.results['success_count'] += 1
                            self.results['response_times'].append(req_result['response_time'])
                        else:
                            self.results['error_count'] += 1
                            self.results['errors'].append(req_result.get('error', 'Unknown error'))
                            
        self.results['end_time'] = datetime.now()
        
    async def simulate_user(
        self,
        client: httpx.AsyncClient,
        user_id: int,
        num_requests: int,
        endpoints: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simulate a single user making multiple requests"""
        
        results = []
        
        for i in range(num_requests):
            # Random endpoint selection
            endpoint_config = random.choice(endpoints)
            
            # Small random delay between requests (0-100ms)
            await asyncio.sleep(random.uniform(0, 0.1))
            
            result = await self.single_request(
                client,
                endpoint_config['endpoint'],
                endpoint_config.get('method', 'GET'),
                endpoint_config.get('data')
            )
            
            results.append(result)
            
        return results
        
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results and generate report"""
        
        if not self.results['response_times']:
            return {'error': 'No successful requests'}
            
        response_times = self.results['response_times']
        duration = (self.results['end_time'] - self.results['start_time']).total_seconds()
        
        analysis = {
            'summary': {
                'total_requests': self.results['success_count'] + self.results['error_count'],
                'successful_requests': self.results['success_count'],
                'failed_requests': self.results['error_count'],
                'error_rate': self.results['error_count'] / max(1, self.results['success_count'] + self.results['error_count']),
                'duration_seconds': duration,
                'requests_per_second': (self.results['success_count'] + self.results['error_count']) / max(1, duration)
            },
            'response_times': {
                'min_ms': min(response_times),
                'max_ms': max(response_times),
                'mean_ms': statistics.mean(response_times),
                'median_ms': statistics.median(response_times),
                'stdev_ms': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                'p50_ms': statistics.quantiles(response_times, n=100)[49] if len(response_times) > 1 else response_times[0],
                'p95_ms': statistics.quantiles(response_times, n=100)[94] if len(response_times) > 1 else response_times[0],
                'p99_ms': statistics.quantiles(response_times, n=100)[98] if len(response_times) > 1 else response_times[0]
            },
            'performance_goals': {
                'target_response_time_ms': 200,
                'meets_target': statistics.median(response_times) < 200,
                'requests_under_200ms': sum(1 for t in response_times if t < 200),
                'percentage_under_200ms': (sum(1 for t in response_times if t < 200) / len(response_times)) * 100
            }
        }
        
        return analysis
        
    def print_report(self, analysis: Dict[str, Any]):
        """Print formatted test report"""
        
        print("\n" + "="*60)
        print("📊 LOAD TEST RESULTS")
        print("="*60)
        
        print("\n📈 Summary:")
        for key, value in analysis['summary'].items():
            print(f"   {key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"   {key.replace('_', ' ').title()}: {value}")
            
        print("\n⏱️  Response Times:")
        for key, value in analysis['response_times'].items():
            print(f"   {key.replace('_', ' ').upper()}: {value:.2f} ms")
            
        print("\n🎯 Performance Goals:")
        for key, value in analysis['performance_goals'].items():
            if isinstance(value, bool):
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"   {key.replace('_', ' ').title()}: {status}")
            elif isinstance(value, float):
                print(f"   {key.replace('_', ' ').title()}: {value:.2f}%")
            else:
                print(f"   {key.replace('_', ' ').title()}: {value}")
                
        print("\n" + "="*60)
        
        # Overall verdict
        if analysis['performance_goals']['meets_target'] and analysis['summary']['error_rate'] < 0.01:
            print("✅ PERFORMANCE TEST PASSED!")
            print("   System can handle load with <200ms median response time")
        else:
            print("❌ PERFORMANCE TEST FAILED!")
            if not analysis['performance_goals']['meets_target']:
                print("   Median response time exceeds 200ms target")
            if analysis['summary']['error_rate'] >= 0.01:
                print(f"   Error rate too high: {analysis['summary']['error_rate']*100:.2f}%")
                
        print("="*60 + "\n")


async def run_progressive_load_test():
    """Run progressive load test to find system limits"""
    
    tester = LoadTester()
    
    # Test endpoints
    endpoints = [
        {'endpoint': '/health', 'method': 'GET'},
        {'endpoint': '/api/v1/status', 'method': 'GET'},
        {'endpoint': '/api/v1/agents', 'method': 'GET'},
        {'endpoint': '/api/v1/metrics', 'method': 'GET'},
        {'endpoint': '/api/v1/chat', 'method': 'POST', 'data': {'message': 'Hello, how are you?'}},
        {'endpoint': '/api/v1/tasks', 'method': 'POST', 'data': {'task_type': 'automation', 'payload': {'test': True}}},
    ]
    
    # Progressive load levels
    load_levels = [
        {'users': 5, 'requests': 10, 'name': 'Light Load'},
        {'users': 25, 'requests': 10, 'name': 'Moderate Load'},
        {'users': 100, 'requests': 10, 'name': 'Heavy Load'},
        {'users': 500, 'requests': 5, 'name': 'Stress Test'},
        {'users': 1000, 'requests': 3, 'name': 'Maximum Load'}
    ]
    
    all_results = []
    
    for level in load_levels:
        print(f"\n{'='*60}")
        print(f"🔥 Testing: {level['name']}")
        print(f"   Users: {level['users']}, Requests per user: {level['requests']}")
        print(f"{'='*60}")
        
        # Reset results
        tester.results = {
            'response_times': [],
            'errors': [],
            'success_count': 0,
            'error_count': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Run test
        await tester.concurrent_test(
            num_users=level['users'],
            requests_per_user=level['requests'],
            endpoints=endpoints
        )
        
        # Analyze
        analysis = tester.analyze_results()
        analysis['load_level'] = level['name']
        analysis['concurrent_users'] = level['users']
        
        # Print report
        tester.print_report(analysis)
        
        all_results.append(analysis)
        
        # Stop if system is failing
        if analysis['summary'].get('error_rate', 0) > 0.1:
            print("\n⚠️  Stopping test - error rate too high!")
            break
            
        # Brief pause between levels
        await asyncio.sleep(2)
        
    return all_results


async def run_spike_test():
    """Test system behavior under sudden load spike"""
    
    print("\n" + "="*60)
    print("⚡ SPIKE TEST - Sudden load increase")
    print("="*60)
    
    tester = LoadTester()
    
    endpoints = [
        {'endpoint': '/api/v1/chat', 'method': 'POST', 'data': {'message': 'Quick test'}},
        {'endpoint': '/api/v1/agents', 'method': 'GET'},
    ]
    
    # Sudden spike from 10 to 500 users
    print("\n📊 Starting with 10 users...")
    await tester.concurrent_test(10, 5, endpoints)
    baseline = tester.analyze_results()
    print(f"   Baseline median response: {baseline['response_times']['median_ms']:.2f}ms")
    
    # Reset and spike
    tester.results = {
        'response_times': [],
        'errors': [],
        'success_count': 0,
        'error_count': 0,
        'start_time': None,
        'end_time': None
    }
    
    print("\n⚡ SPIKING to 500 users!")
    await tester.concurrent_test(500, 3, endpoints)
    spike_results = tester.analyze_results()
    
    # Compare
    print(f"\n📈 Results comparison:")
    print(f"   Baseline median: {baseline['response_times']['median_ms']:.2f}ms")
    print(f"   Spike median: {spike_results['response_times']['median_ms']:.2f}ms")
    print(f"   Degradation: {(spike_results['response_times']['median_ms'] / baseline['response_times']['median_ms'] - 1) * 100:.1f}%")
    
    if spike_results['response_times']['median_ms'] < 500:
        print("   ✅ System handled spike well!")
    else:
        print("   ⚠️  Significant performance degradation under spike")


async def run_endurance_test(duration_minutes: int = 5):
    """Test system stability over extended period"""
    
    print("\n" + "="*60)
    print(f"⏰ ENDURANCE TEST - {duration_minutes} minutes continuous load")
    print("="*60)
    
    tester = LoadTester()
    
    endpoints = [
        {'endpoint': '/health', 'method': 'GET'},
        {'endpoint': '/api/v1/status', 'method': 'GET'},
        {'endpoint': '/api/v1/chat', 'method': 'POST', 'data': {'message': 'Endurance test'}},
    ]
    
    # Sustained load
    users = 100
    total_iterations = duration_minutes * 2  # Every 30 seconds
    
    results_over_time = []
    
    for i in range(total_iterations):
        print(f"\n🔄 Iteration {i+1}/{total_iterations} (minute {(i+1)*0.5:.1f}/{duration_minutes})")
        
        # Reset results
        tester.results = {
            'response_times': [],
            'errors': [],
            'success_count': 0,
            'error_count': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Run 30 second test
        await tester.concurrent_test(users, 10, endpoints)
        
        analysis = tester.analyze_results()
        analysis['iteration'] = i + 1
        analysis['elapsed_minutes'] = (i + 1) * 0.5
        
        results_over_time.append(analysis)
        
        # Print summary
        print(f"   Median response: {analysis['response_times']['median_ms']:.2f}ms")
        print(f"   Error rate: {analysis['summary']['error_rate']*100:.2f}%")
        
        # Check for degradation
        if i > 0:
            prev_median = results_over_time[i-1]['response_times']['median_ms']
            curr_median = analysis['response_times']['median_ms']
            if curr_median > prev_median * 1.5:
                print("   ⚠️  Performance degrading!")
                
    # Final analysis
    print("\n" + "="*60)
    print("📊 ENDURANCE TEST SUMMARY")
    print("="*60)
    
    medians = [r['response_times']['median_ms'] for r in results_over_time]
    error_rates = [r['summary']['error_rate'] for r in results_over_time]
    
    print(f"   Starting median: {medians[0]:.2f}ms")
    print(f"   Ending median: {medians[-1]:.2f}ms")
    print(f"   Performance change: {((medians[-1] / medians[0]) - 1) * 100:.1f}%")
    print(f"   Average error rate: {statistics.mean(error_rates)*100:.2f}%")
    
    if medians[-1] < medians[0] * 1.2 and statistics.mean(error_rates) < 0.01:
        print("   ✅ System remained stable under sustained load!")
    else:
        print("   ⚠️  Performance degradation or errors detected over time")


async def main():
    """Main test orchestration"""
    
    print("\n" + "="*80)
    print("🚀 SUTAZAI PERFORMANCE TEST SUITE")
    print("   Target: 1000+ concurrent users with <200ms response time")
    print("="*80)
    
    # Check if backend is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:10010/health", timeout=5.0)
            if response.status_code != 200:
                print("❌ Backend not responding properly!")
                return
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("   Please ensure the backend is running on port 10010")
        return
        
    print("✅ Backend is running, starting tests...\n")
    
    # Run test suites
    print("\n📋 Test Plan:")
    print("   1. Progressive Load Test (5 → 1000 users)")
    print("   2. Spike Test (10 → 500 users)")
    print("   3. Endurance Test (5 minutes sustained load)")
    
    # Progressive load test
    print("\n" + "="*80)
    print("TEST 1: PROGRESSIVE LOAD")
    print("="*80)
    progressive_results = await run_progressive_load_test()
    
    # Spike test
    print("\n" + "="*80)
    print("TEST 2: SPIKE TEST")
    print("="*80)
    await run_spike_test()
    
    # Endurance test
    print("\n" + "="*80)
    print("TEST 3: ENDURANCE TEST")
    print("="*80)
    await run_endurance_test(duration_minutes=2)  # Short version for demo
    
    # Final summary
    print("\n" + "="*80)
    print("🏁 FINAL PERFORMANCE REPORT")
    print("="*80)
    
    # Find maximum successful load level
    max_users = 0
    for result in progressive_results:
        if result['performance_goals']['meets_target'] and result['summary']['error_rate'] < 0.01:
            max_users = result['concurrent_users']
            
    print(f"\n✅ Maximum concurrent users with <200ms response: {max_users}")
    
    if max_users >= 1000:
        print("🎉 PERFORMANCE GOAL ACHIEVED!")
        print("   System can handle 1000+ concurrent users with <200ms response time")
    else:
        print(f"⚠️  Performance goal not fully met")
        print(f"   Current capacity: {max_users} concurrent users")
        print(f"   Target: 1000 concurrent users")
        
    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())