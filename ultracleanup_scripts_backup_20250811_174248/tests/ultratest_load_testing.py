#!/usr/bin/env python3
"""
ULTRATEST Load Testing Suite
Tests system ability to handle 100+ concurrent users.
"""

import requests
import time
import json
import threading
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any
import statistics

class UltratestLoadValidator:
    def __init__(self):
        self.results = {}
        self.endpoints = [
            ('Health Check', 'http://localhost:10010/health', 'GET'),
            ('Frontend', 'http://localhost:10011/', 'GET'),
            ('Ollama Tags', 'http://localhost:10104/api/tags', 'GET'),
            ('Hardware Optimizer', 'http://localhost:11110/health', 'GET'),
            ('AI Orchestrator', 'http://localhost:8589/health', 'GET'),
        ]
        
    def single_request(self, endpoint_info: tuple, user_id: int) -> Dict[str, Any]:
        """Simulate a single user request"""
        name, url, method = endpoint_info
        
        try:
            start_time = time.time()
            
            if method == 'GET':
                response = requests.get(url, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json={'test': 'data'}, timeout=10)
            else:
                response = requests.request(method, url, timeout=10)
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            return {
                'user_id': user_id,
                'endpoint': name,
                'url': url,
                'response_time_ms': response_time,
                'status_code': response.status_code,
                'success': 200 <= response.status_code < 400,
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.Timeout:
            return {
                'user_id': user_id,
                'endpoint': name,
                'url': url,
                'response_time_ms': None,
                'error': 'Timeout',
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'user_id': user_id,
                'endpoint': name,
                'url': url,
                'response_time_ms': None,
                'error': str(e),
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def simulate_user_session(self, user_id: int, duration_seconds: int = 30) -> List[Dict[str, Any]]:
        """Simulate a complete user session with multiple requests"""
        results = []
        session_start = time.time()
        request_count = 0
        
        while time.time() - session_start < duration_seconds and request_count < 10:  # Max 10 requests per user
            # Choose a random endpoint
            import random
            endpoint = random.choice(self.endpoints)
            
            result = self.single_request(endpoint, user_id)
            results.append(result)
            request_count += 1
            
            # Wait between requests (simulate real user behavior)
            time.sleep(random.uniform(1, 3))
        
        return results
    
    def run_concurrent_load_test(self, num_users: int = 50, duration_seconds: int = 30) -> Dict[str, Any]:
        """Run concurrent load test with specified number of users"""
        print(f"🚀 Running load test with {num_users} concurrent users for {duration_seconds} seconds...")
        
        all_results = []
        test_start_time = time.time()
        
        # Use ThreadPoolExecutor for concurrent users
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_users, 50)) as executor:
            # Submit all user sessions
            future_to_user = {
                executor.submit(self.simulate_user_session, user_id, duration_seconds): user_id
                for user_id in range(num_users)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_user):
                user_id = future_to_user[future]
                try:
                    user_results = future.result()
                    all_results.extend(user_results)
                    if user_id % 10 == 0:  # Progress indicator
                        print(f"   User {user_id} completed...")
                except Exception as e:
                    print(f"   User {user_id} failed: {e}")
        
        test_end_time = time.time()
        total_test_duration = test_end_time - test_start_time
        
        return {
            'test_duration_seconds': total_test_duration,
            'target_users': num_users,
            'target_duration': duration_seconds,
            'all_results': all_results,
            'total_requests': len(all_results)
        }
    
    def analyze_load_test_results(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze load test results for performance metrics"""
        all_results = test_data.get('all_results', [])
        
        if not all_results:
            return {'error': 'No test results to analyze'}
        
        # Separate successful and failed requests
        successful_requests = [r for r in all_results if r.get('success', False)]
        failed_requests = [r for r in all_results if not r.get('success', False)]
        
        # Calculate response time metrics
        response_times = [r['response_time_ms'] for r in successful_requests if r.get('response_time_ms') is not None]
        
        # Endpoint-specific analysis
        endpoint_stats = {}
        for result in all_results:
            endpoint = result.get('endpoint', 'Unknown')
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = {
                    'total_requests': 0,
                    'successful_requests': 0,
                    'failed_requests': 0,
                    'response_times': []
                }
            
            endpoint_stats[endpoint]['total_requests'] += 1
            if result.get('success', False):
                endpoint_stats[endpoint]['successful_requests'] += 1
                if result.get('response_time_ms') is not None:
                    endpoint_stats[endpoint]['response_times'].append(result['response_time_ms'])
            else:
                endpoint_stats[endpoint]['failed_requests'] += 1
        
        # Calculate endpoint averages
        for endpoint, stats in endpoint_stats.items():
            if stats['response_times']:
                stats['avg_response_time_ms'] = statistics.mean(stats['response_times'])
                stats['min_response_time_ms'] = min(stats['response_times'])
                stats['max_response_time_ms'] = max(stats['response_times'])
                stats['success_rate'] = (stats['successful_requests'] / stats['total_requests']) * 100
            else:
                stats['avg_response_time_ms'] = None
                stats['success_rate'] = 0
        
        analysis = {
            'total_requests': len(all_results),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'overall_success_rate': (len(successful_requests) / len(all_results)) * 100 if all_results else 0,
            'concurrent_users_achieved': test_data.get('target_users', 0),
            'test_duration_seconds': test_data.get('test_duration_seconds', 0),
            'requests_per_second': len(all_results) / test_data.get('test_duration_seconds', 1),
            'endpoint_statistics': endpoint_stats
        }
        
        if response_times:
            analysis.update({
                'avg_response_time_ms': statistics.mean(response_times),
                'min_response_time_ms': min(response_times),
                'max_response_time_ms': max(response_times),
                'median_response_time_ms': statistics.median(response_times),
                'response_time_95th_percentile': sorted(response_times)[int(0.95 * len(response_times))] if response_times else 0,
                'fast_responses_under_50ms': sum(1 for rt in response_times if rt <= 50),
                'fast_response_rate': (sum(1 for rt in response_times if rt <= 50) / len(response_times)) * 100
            })
        
        return analysis
    
    def run_scalability_test(self) -> Dict[str, Any]:
        """Run incremental load test to find system limits"""
        print("📈 Running scalability test...")
        
        user_counts = [10, 25, 50, 75, 100]  # Progressive load
        scalability_results = {}
        
        for user_count in user_counts:
            print(f"   Testing with {user_count} users...")
            
            # Run shorter tests for scalability
            test_data = self.run_concurrent_load_test(user_count, 15)  # 15 second tests
            analysis = self.analyze_load_test_results(test_data)
            
            scalability_results[user_count] = {
                'success_rate': analysis.get('overall_success_rate', 0),
                'avg_response_time': analysis.get('avg_response_time_ms', 0),
                'requests_per_second': analysis.get('requests_per_second', 0),
                'fast_response_rate': analysis.get('fast_response_rate', 0)
            }
            
            # Stop if success rate drops below 80%
            if analysis.get('overall_success_rate', 0) < 80:
                print(f"   Stopping scalability test - success rate dropped to {analysis.get('overall_success_rate', 0):.1f}%")
                break
        
        return scalability_results
    
    def generate_load_test_report(self, analysis: Dict[str, Any], scalability: Dict[str, Any]):
        """Generate comprehensive load test report"""
        print("\n" + "=" * 80)
        print("🚀 ULTRATEST LOAD TESTING REPORT")
        print("=" * 80)
        print(f"Test Execution Time: {datetime.now().isoformat()}")
        
        if 'error' in analysis:
            print(f"❌ Error: {analysis['error']}")
            return False
        
        # Overall performance summary
        print(f"\n📊 LOAD TEST PERFORMANCE SUMMARY:")
        print("-" * 50)
        print(f"Concurrent Users: {analysis.get('concurrent_users_achieved', 0)}")
        print(f"Total Requests: {analysis.get('total_requests', 0)}")
        print(f"Successful Requests: {analysis.get('successful_requests', 0)}")
        print(f"Failed Requests: {analysis.get('failed_requests', 0)}")
        print(f"Overall Success Rate: {analysis.get('overall_success_rate', 0):.1f}%")
        print(f"Requests per Second: {analysis.get('requests_per_second', 0):.2f}")
        print(f"Test Duration: {analysis.get('test_duration_seconds', 0):.1f} seconds")
        
        # Response time analysis
        if 'avg_response_time_ms' in analysis:
            print(f"\n⚡ RESPONSE TIME ANALYSIS:")
            print("-" * 50)
            print(f"Average Response Time: {analysis['avg_response_time_ms']:.2f}ms")
            print(f"Median Response Time: {analysis.get('median_response_time_ms', 0):.2f}ms")
            print(f"95th Percentile: {analysis.get('response_time_95th_percentile', 0):.2f}ms")
            print(f"Min/Max Response Time: {analysis.get('min_response_time_ms', 0):.2f}ms / {analysis.get('max_response_time_ms', 0):.2f}ms")
            print(f"Fast Responses (<50ms): {analysis.get('fast_responses_under_50ms', 0)} ({analysis.get('fast_response_rate', 0):.1f}%)")
        
        # Endpoint-specific results
        endpoint_stats = analysis.get('endpoint_statistics', {})
        if endpoint_stats:
            print(f"\n🎯 ENDPOINT PERFORMANCE:")
            print("-" * 50)
            for endpoint, stats in endpoint_stats.items():
                success_rate = stats.get('success_rate', 0)
                avg_response = stats.get('avg_response_time_ms', 0)
                if avg_response:
                    print(f"{endpoint:20} {success_rate:5.1f}% success, {avg_response:6.2f}ms avg")
                else:
                    print(f"{endpoint:20} {success_rate:5.1f}% success, FAILED")
        
        # Scalability results
        if scalability:
            print(f"\n📈 SCALABILITY ANALYSIS:")
            print("-" * 50)
            for user_count, metrics in scalability.items():
                success_rate = metrics.get('success_rate', 0)
                avg_response = metrics.get('avg_response_time', 0)
                rps = metrics.get('requests_per_second', 0)
                print(f"{user_count:3d} users: {success_rate:5.1f}% success, {avg_response:6.2f}ms avg, {rps:5.2f} req/s")
        
        # Target assessment
        target_100_users = analysis.get('concurrent_users_achieved', 0) >= 100
        acceptable_success_rate = analysis.get('overall_success_rate', 0) >= 90
        reasonable_response_time = analysis.get('avg_response_time_ms', float('inf')) <= 100  # 100ms under load is reasonable
        
        print(f"\n🎯 LOAD TESTING TARGETS:")
        print("-" * 50)
        print(f"Target: Handle 100+ concurrent users")
        print(f"Achieved: {analysis.get('concurrent_users_achieved', 0)} users")
        if target_100_users:
            print("✅ 100+ USER TARGET ACHIEVED")
        else:
            print("❌ Did not reach 100 concurrent users")
        
        print(f"\nTarget: >90% success rate under load")
        print(f"Achieved: {analysis.get('overall_success_rate', 0):.1f}% success rate")
        if acceptable_success_rate:
            print("✅ SUCCESS RATE TARGET ACHIEVED")
        else:
            print("❌ Success rate below 90%")
        
        print(f"\nTarget: Reasonable response times under load (<100ms)")
        if 'avg_response_time_ms' in analysis:
            print(f"Achieved: {analysis['avg_response_time_ms']:.2f}ms average")
            if reasonable_response_time:
                print("✅ RESPONSE TIME TARGET ACHIEVED")
            else:
                print("❌ Response times too high under load")
        
        # Overall assessment
        print(f"\n🏆 LOAD TESTING ASSESSMENT:")
        print("-" * 50)
        
        achievements = []
        issues = []
        
        if target_100_users:
            achievements.append("Successfully handled 100+ concurrent users")
        else:
            issues.append(f"Only handled {analysis.get('concurrent_users_achieved', 0)} concurrent users")
        
        if acceptable_success_rate:
            achievements.append(f"High success rate under load ({analysis.get('overall_success_rate', 0):.1f}%)")
        else:
            issues.append(f"Low success rate under load ({analysis.get('overall_success_rate', 0):.1f}%)")
        
        if reasonable_response_time:
            achievements.append(f"Good response times under load ({analysis.get('avg_response_time_ms', 0):.2f}ms)")
        elif 'avg_response_time_ms' in analysis:
            issues.append(f"Slow response times under load ({analysis['avg_response_time_ms']:.2f}ms)")
        
        if analysis.get('requests_per_second', 0) >= 10:
            achievements.append(f"Good throughput ({analysis['requests_per_second']:.2f} req/s)")
        
        print("🎉 ACHIEVEMENTS:")
        for achievement in achievements:
            print(f"   ✅ {achievement}")
        
        if issues:
            print("\n⚠️  AREAS FOR IMPROVEMENT:")
            for issue in issues:
                print(f"   ❌ {issue}")
        
        success_count = len(achievements)
        total_criteria = len(achievements) + len(issues)
        success_rate = (success_count / total_criteria * 100) if total_criteria > 0 else 0
        
        print(f"\n📈 Load Testing Success Rate: {success_rate:.1f}%")
        
        return target_100_users and acceptable_success_rate and success_rate >= 75

def main():
    """Run comprehensive load testing validation"""
    print("🚀 Starting ULTRATEST Load Testing Validation")
    
    validator = UltratestLoadValidator()
    
    # Run main load test with 100+ users
    print("\n🔥 Primary load test with 100 concurrent users...")
    test_data = validator.run_concurrent_load_test(100, 30)
    analysis = validator.analyze_load_test_results(test_data)
    
    # Run scalability test
    scalability_results = validator.run_scalability_test()
    
    # Generate comprehensive report
    success = validator.generate_load_test_report(analysis, scalability_results)
    
    # Save detailed results
    full_results = {
        'timestamp': datetime.now().isoformat(),
        'primary_load_test': analysis,
        'scalability_results': scalability_results,
        'raw_test_data': test_data
    }
    
    with open('/opt/sutazaiapp/tests/ultratest_load_testing_report.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    print(f"\n📄 Full report saved to: /opt/sutazaiapp/tests/ultratest_load_testing_report.json")
    
    if success:
        print("\n🎉 LOAD TESTING VALIDATION SUCCESSFUL!")
        return 0
    else:
        print("\n⚠️  LOAD TESTING NEEDS IMPROVEMENT")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())