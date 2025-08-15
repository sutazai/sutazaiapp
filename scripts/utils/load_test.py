#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
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
        
        logger.info(f"\n🚀 Starting load test with {num_users} concurrent users")
        logger.info(f"   Each user will make {requests_per_user} requests")
        logger.info(f"   Total requests: {num_users * requests_per_user}")
        
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
        
        logger.info("\n" + "="*60)
        logger.info("📊 LOAD TEST RESULTS")
        logger.info("="*60)
        
        logger.info("\n📈 Summary:")
        for key, value in analysis['summary'].items():
            logger.info(f"   {key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"   {key.replace('_', ' ').title()}: {value}")
            
        logger.info("\n⏱️  Response Times:")
        for key, value in analysis['response_times'].items():
            logger.info(f"   {key.replace('_', ' ').upper()}: {value:.2f} ms")
            
        logger.info("\n🎯 Performance Goals:")
        for key, value in analysis['performance_goals'].items():
            if isinstance(value, bool):
                status = "✅ PASS" if value else "❌ FAIL"
                logger.info(f"   {key.replace('_', ' ').title()}: {status}")
            elif isinstance(value, float):
                logger.info(f"   {key.replace('_', ' ').title()}: {value:.2f}%")
            else:
                logger.info(f"   {key.replace('_', ' ').title()}: {value}")
                
        logger.info("\n" + "="*60)
        
        # Overall verdict
        if analysis['performance_goals']['meets_target'] and analysis['summary']['error_rate'] < 0.01:
            logger.info("✅ PERFORMANCE TEST PASSED!")
            logger.info("   System can handle load with <200ms median response time")
        else:
            logger.error("❌ PERFORMANCE TEST FAILED!")
            if not analysis['performance_goals']['meets_target']:
                logger.info("   Median response time exceeds 200ms target")
            if analysis['summary']['error_rate'] >= 0.01:
                logger.error(f"   Error rate too high: {analysis['summary']['error_rate']*100:.2f}%")
                
        logger.info("="*60 + "\n")


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
        logger.info(f"\n{'='*60}")
        logger.info(f"🔥 Testing: {level['name']}")
        logger.info(f"   Users: {level['users']}, Requests per user: {level['requests']}")
        logger.info(f"{'='*60}")
        
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
            logger.error("\n⚠️  Stopping test - error rate too high!")
            break
            
        # Brief pause between levels
        await asyncio.sleep(2)
        
    return all_results


async def run_spike_test():
    """Test system behavior under sudden load spike"""
    
    logger.info("\n" + "="*60)
    logger.info("⚡ SPIKE TEST - Sudden load increase")
    logger.info("="*60)
    
    tester = LoadTester()
    
    endpoints = [
        {'endpoint': '/api/v1/chat', 'method': 'POST', 'data': {'message': 'Quick test'}},
        {'endpoint': '/api/v1/agents', 'method': 'GET'},
    ]
    
    # Sudden spike from 10 to 500 users
    logger.info("\n📊 Starting with 10 users...")
    await tester.concurrent_test(10, 5, endpoints)
    baseline = tester.analyze_results()
    logger.info(f"   Baseline median response: {baseline['response_times']['median_ms']:.2f}ms")
    
    # Reset and spike
    tester.results = {
        'response_times': [],
        'errors': [],
        'success_count': 0,
        'error_count': 0,
        'start_time': None,
        'end_time': None
    }
    
    logger.info("\n⚡ SPIKING to 500 users!")
    await tester.concurrent_test(500, 3, endpoints)
    spike_results = tester.analyze_results()
    
    # Compare
    logger.info(f"\n📈 Results comparison:")
    logger.info(f"   Baseline median: {baseline['response_times']['median_ms']:.2f}ms")
    logger.info(f"   Spike median: {spike_results['response_times']['median_ms']:.2f}ms")
    logger.info(f"   Degradation: {(spike_results['response_times']['median_ms'] / baseline['response_times']['median_ms'] - 1) * 100:.1f}%")
    
    if spike_results['response_times']['median_ms'] < 500:
        logger.info("   ✅ System handled spike well!")
    else:
        logger.info("   ⚠️  Significant performance degradation under spike")


async def run_endurance_test(duration_minutes: int = 5):
    """Test system stability over extended period"""
    
    logger.info("\n" + "="*60)
    logger.info(f"⏰ ENDURANCE TEST - {duration_minutes} minutes continuous load")
    logger.info("="*60)
    
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
        logger.info(f"\n🔄 Iteration {i+1}/{total_iterations} (minute {(i+1)*0.5:.1f}/{duration_minutes})")
        
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
        logger.info(f"   Median response: {analysis['response_times']['median_ms']:.2f}ms")
        logger.error(f"   Error rate: {analysis['summary']['error_rate']*100:.2f}%")
        
        # Check for degradation
        if i > 0:
            prev_median = results_over_time[i-1]['response_times']['median_ms']
            curr_median = analysis['response_times']['median_ms']
            if curr_median > prev_median * 1.5:
                logger.info("   ⚠️  Performance degrading!")
                
    # Final analysis
    logger.info("\n" + "="*60)
    logger.info("📊 ENDURANCE TEST SUMMARY")
    logger.info("="*60)
    
    medians = [r['response_times']['median_ms'] for r in results_over_time]
    error_rates = [r['summary']['error_rate'] for r in results_over_time]
    
    logger.info(f"   Starting median: {medians[0]:.2f}ms")
    logger.info(f"   Ending median: {medians[-1]:.2f}ms")
    logger.info(f"   Performance change: {((medians[-1] / medians[0]) - 1) * 100:.1f}%")
    logger.error(f"   Average error rate: {statistics.mean(error_rates)*100:.2f}%")
    
    if medians[-1] < medians[0] * 1.2 and statistics.mean(error_rates) < 0.01:
        logger.info("   ✅ System remained stable under sustained load!")
    else:
        logger.error("   ⚠️  Performance degradation or errors detected over time")


async def main():
    """Main test orchestration"""
    
    logger.info("\n" + "="*80)
    logger.info("🚀 SUTAZAI PERFORMANCE TEST SUITE")
    logger.info("   Target: 1000+ concurrent users with <200ms response time")
    logger.info("="*80)
    
    # Check if backend is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:10010/health", timeout=5.0)
            if response.status_code != 200:
                logger.info("❌ Backend not responding properly!")
                return
    except Exception as e:
        logger.info(f"❌ Cannot connect to backend: {e}")
        logger.info("   Please ensure the backend is running on port 10010")
        return
        
    logger.info("✅ Backend is running, starting tests...\n")
    
    # Run test suites
    logger.info("\n📋 Test Plan:")
    logger.info("   1. Progressive Load Test (5 → 1000 users)")
    logger.info("   2. Spike Test (10 → 500 users)")
    logger.info("   3. Endurance Test (5 minutes sustained load)")
    
    # Progressive load test
    logger.info("\n" + "="*80)
    logger.info("TEST 1: PROGRESSIVE LOAD")
    logger.info("="*80)
    progressive_results = await run_progressive_load_test()
    
    # Spike test
    logger.info("\n" + "="*80)
    logger.info("TEST 2: SPIKE TEST")
    logger.info("="*80)
    await run_spike_test()
    
    # Endurance test
    logger.info("\n" + "="*80)
    logger.info("TEST 3: ENDURANCE TEST")
    logger.info("="*80)
    await run_endurance_test(duration_minutes=2)  # Short version for demo
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("🏁 FINAL PERFORMANCE REPORT")
    logger.info("="*80)
    
    # Find maximum successful load level
    max_users = 0
    for result in progressive_results:
        if result['performance_goals']['meets_target'] and result['summary']['error_rate'] < 0.01:
            max_users = result['concurrent_users']
            
    logger.info(f"\n✅ Maximum concurrent users with <200ms response: {max_users}")
    
    if max_users >= 1000:
        logger.info("🎉 PERFORMANCE GOAL ACHIEVED!")
        logger.info("   System can handle 1000+ concurrent users with <200ms response time")
    else:
        logger.info(f"⚠️  Performance goal not fully met")
        logger.info(f"   Current capacity: {max_users} concurrent users")
        logger.info(f"   Target: 1000 concurrent users")
        
    logger.info("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())