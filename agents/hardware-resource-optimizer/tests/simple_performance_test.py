#!/usr/bin/env python3
"""
Simple Performance Testing for Hardware Resource Optimizer
==========================================================

Basic performance testing without complex dependencies.
Tests concurrent load and response times.

Author: QA Team Lead
Version: 1.0.0
"""

import json
import time
import requests
import threading
import statistics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SimplePerformanceTest')

class SimplePerformanceTester:
    """Simple performance testing"""
    
    def __init__(self, base_url="http://localhost:8116", timeout=30):
        self.base_url = base_url
        self.timeout = timeout
        self.results = []
    
    def single_request_test(self, endpoint, method="GET", params=None, iterations=20):
        """Test single endpoint multiple times"""
        logger.info(f"Testing {method} {endpoint} - {iterations} iterations")
        
        times = []
        errors = []
        
        for i in range(iterations):
            try:
                start_time = time.time()
                
                session = requests.Session()
                session.timeout = self.timeout
                
                if method.upper() == "GET":
                    response = session.get(f"{self.base_url}{endpoint}", params=params)
                elif method.upper() == "POST":
                    response = session.post(f"{self.base_url}{endpoint}", params=params)
                
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    times.append(duration)
                else:
                    errors.append(f"Status {response.status_code}")
                    
            except Exception as e:
                errors.append(str(e))
        
        if times:
            result = {
                'endpoint': endpoint,
                'method': method,
                'iterations': iterations,
                'successful_requests': len(times),
                'failed_requests': len(errors),
                'success_rate': len(times) / iterations * 100,
                'avg_response_time': statistics.mean(times),
                'min_response_time': min(times),
                'max_response_time': max(times),
                'median_response_time': statistics.median(times),
                'std_dev': statistics.stdev(times) if len(times) > 1 else 0
            }
            
            logger.info(f"  Results: {len(times)}/{iterations} success, avg: {result['avg_response_time']:.3f}s")
            return result
        else:
            logger.error(f"  All requests failed!")
            return {
                'endpoint': endpoint,
                'method': method,
                'iterations': iterations,
                'successful_requests': 0,
                'failed_requests': len(errors),
                'success_rate': 0,
                'errors': errors[:5]
            }
    
    def concurrent_load_test(self, endpoint, method="GET", params=None, 
                           concurrent_users=10, requests_per_user=5):
        """Test with concurrent load"""
        logger.info(f"Concurrent load test: {concurrent_users} users, {requests_per_user} requests each")
        
        results = []
        start_time = time.time()
        
        def user_session():
            """Simulate a user session"""
            session_results = []
            session = requests.Session()
            session.timeout = self.timeout
            
            for _ in range(requests_per_user):
                try:
                    req_start = time.time()
                    
                    if method.upper() == "GET":
                        response = session.get(f"{self.base_url}{endpoint}", params=params)
                    elif method.upper() == "POST":
                        response = session.post(f"{self.base_url}{endpoint}", params=params)
                    
                    duration = time.time() - req_start
                    
                    session_results.append({
                        'success': response.status_code == 200,
                        'duration': duration,
                        'status_code': response.status_code
                    })
                    
                except Exception as e:
                    session_results.append({
                        'success': False,
                        'duration': 0,
                        'error': str(e)
                    })
            
            return session_results
        
        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_session) for _ in range(concurrent_users)]
            
            for future in as_completed(futures):
                try:
                    user_results = future.result()
                    results.extend(user_results)
                except Exception as e:
                    logger.error(f"User session failed: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        response_times = [r['duration'] for r in successful_requests]
        
        result = {
            'test_type': 'concurrent_load',
            'endpoint': endpoint,
            'method': method,
            'concurrent_users': concurrent_users,
            'requests_per_user': requests_per_user,
            'total_requests': len(results),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / len(results) * 100 if results else 0,
            'total_duration': total_time,
            'requests_per_second': len(results) / total_time if total_time > 0 else 0,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'median_response_time': statistics.median(response_times) if response_times else 0
        }
        
        logger.info(f"  Results: {len(successful_requests)}/{len(results)} success, "
                   f"avg: {result['avg_response_time']:.3f}s, "
                   f"rps: {result['requests_per_second']:.1f}")
        
        return result
    
    def run_performance_tests(self):
        """Run comprehensive performance tests"""
        logger.info("Starting performance testing...")
        
        # Single request benchmarks
        single_tests = [
            ("/health", "GET", None),
            ("/status", "GET", None),
            ("/optimize/memory", "POST", None),
            ("/analyze/storage", "GET", {"path": "/tmp"}),
            ("/analyze/storage/report", "GET", None)
        ]
        
        for endpoint, method, params in single_tests:
            result = self.single_request_test(endpoint, method, params, iterations=20)
            self.results.append(result)
        
        # Concurrent load tests
        concurrent_tests = [
            ("/health", "GET", None, 5, 10),  # Light load
            ("/status", "GET", None, 10, 5),  # Medium load
            ("/analyze/storage", "GET", {"path": "/tmp"}, 8, 3),  # Heavy load
        ]
        
        for endpoint, method, params, users, requests in concurrent_tests:
            result = self.concurrent_load_test(endpoint, method, params, users, requests)
            self.results.append(result)
    
    def generate_report(self):
        """Generate performance report"""
        # Single request statistics
        single_tests = [r for r in self.results if 'test_type' not in r]
        concurrent_tests = [r for r in self.results if r.get('test_type') == 'concurrent_load']
        
        single_response_times = []
        single_success_rates = []
        
        for test in single_tests:
            if 'avg_response_time' in test:
                single_response_times.append(test['avg_response_time'])
                single_success_rates.append(test.get('success_rate', 0))
        
        concurrent_response_times = []
        concurrent_success_rates = []
        total_rps = []
        
        for test in concurrent_tests:
            if 'avg_response_time' in test:
                concurrent_response_times.append(test['avg_response_time'])
                concurrent_success_rates.append(test.get('success_rate', 0))
                total_rps.append(test.get('requests_per_second', 0))
        
        report = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(self.results),
                'single_request_tests': len(single_tests),
                'concurrent_load_tests': len(concurrent_tests)
            },
            'single_request_performance': {
                'avg_response_time': statistics.mean(single_response_times) if single_response_times else 0,
                'max_response_time': max(single_response_times) if single_response_times else 0,
                'min_response_time': min(single_response_times) if single_response_times else 0,
                'avg_success_rate': statistics.mean(single_success_rates) if single_success_rates else 0
            },
            'concurrent_load_performance': {
                'avg_response_time': statistics.mean(concurrent_response_times) if concurrent_response_times else 0,
                'max_response_time': max(concurrent_response_times) if concurrent_response_times else 0,
                'avg_success_rate': statistics.mean(concurrent_success_rates) if concurrent_success_rates else 0,
                'total_requests_per_second': sum(total_rps),
                'avg_requests_per_second': statistics.mean(total_rps) if total_rps else 0
            },
            'detailed_results': self.results,
            'agent_url': self.base_url
        }
        
        return report
    
    def print_summary(self, report):
        """Print performance summary"""
        summary = report['test_summary']
        single = report['single_request_performance']
        concurrent = report['concurrent_load_performance']
        
        print("\n" + "="*80)
        print("HARDWARE RESOURCE OPTIMIZER - PERFORMANCE TEST SUMMARY")
        print("="*80)
        
        print(f"Agent URL: {self.base_url}")
        print(f"Test Timestamp: {summary['timestamp']}")
        print(f"Total Tests: {summary['total_tests']}")
        
        print(f"\nSingle Request Performance:")
        print(f"  Tests Run: {summary['single_request_tests']}")
        print(f"  Average Response Time: {single['avg_response_time']:.3f}s")
        print(f"  Max Response Time: {single['max_response_time']:.3f}s")
        print(f"  Min Response Time: {single['min_response_time']:.3f}s")
        print(f"  Average Success Rate: {single['avg_success_rate']:.1f}%")
        
        print(f"\nConcurrent Load Performance:")
        print(f"  Tests Run: {summary['concurrent_load_tests']}")
        print(f"  Average Response Time: {concurrent['avg_response_time']:.3f}s")
        print(f"  Max Response Time: {concurrent['max_response_time']:.3f}s")
        print(f"  Average Success Rate: {concurrent['avg_success_rate']:.1f}%")
        print(f"  Total Requests/Second: {concurrent['total_requests_per_second']:.1f}")
        print(f"  Average Requests/Second: {concurrent['avg_requests_per_second']:.1f}")
        
        # Performance assessment
        if (single['avg_response_time'] < 1.0 and 
            concurrent['avg_response_time'] < 2.0 and 
            single['avg_success_rate'] > 95 and 
            concurrent['avg_success_rate'] > 95):
            print(f"\n🚀 PERFORMANCE ASSESSMENT: EXCELLENT")
        elif (single['avg_response_time'] < 2.0 and 
              concurrent['avg_response_time'] < 5.0 and 
              single['avg_success_rate'] > 90 and 
              concurrent['avg_success_rate'] > 90):
            print(f"\n👍 PERFORMANCE ASSESSMENT: GOOD")
        elif (single['avg_response_time'] < 5.0 and 
              concurrent['avg_response_time'] < 10.0 and 
              single['avg_success_rate'] > 75 and 
              concurrent['avg_success_rate'] > 75):
            print(f"\n⚠️ PERFORMANCE ASSESSMENT: ACCEPTABLE")
        else:
            print(f"\n❌ PERFORMANCE ASSESSMENT: NEEDS IMPROVEMENT")
        
        print("="*80)
    
    def save_report(self, report, filename=None):
        """Save report to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_test_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {filename}")
        return filename

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Hardware Optimizer Performance Testing")
    parser.add_argument("--url", default="http://localhost:8116", help="Agent URL")
    parser.add_argument("--output", help="Output report file")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = SimplePerformanceTester(args.url, args.timeout)
    
    # Check if agent is available
    try:
        response = requests.get(f"{args.url}/health", timeout=10)
        if response.status_code != 200:
            print(f"❌ Agent not available at {args.url}")
            return 1
        print(f"✅ Agent is available at {args.url}")
    except Exception as e:
        print(f"❌ Cannot connect to agent at {args.url}: {e}")
        return 1
    
    # Run tests
    start_time = time.time()
    tester.run_performance_tests()
    total_duration = time.time() - start_time
    
    # Generate and print report
    report = tester.generate_report()
    report['test_summary']['total_duration_seconds'] = total_duration
    
    tester.print_summary(report)
    
    # Save report
    output_file = args.output or f"performance_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    tester.save_report(report, output_file)
    
    # Return appropriate exit code
    single_perf = report['single_request_performance']
    concurrent_perf = report['concurrent_load_performance']
    
    performance_good = (
        single_perf['avg_response_time'] < 5.0 and
        concurrent_perf['avg_response_time'] < 10.0 and
        single_perf['avg_success_rate'] > 80 and
        concurrent_perf['avg_success_rate'] > 80
    )
    
    return 0 if performance_good else 1

if __name__ == "__main__":
    exit(main())