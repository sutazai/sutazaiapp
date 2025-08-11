#!/usr/bin/env python3
"""
ULTRATEST Response Time Validation
Tests all API endpoints for <50ms response times.
"""

import requests
import time
import statistics
import json
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Tuple, Any

class UltratestResponseTimeValidator:
    def __init__(self):
        self.test_results = {}
        self.endpoint_results = {}
        
        # Define all service endpoints to test
        self.endpoints = [
            ('Backend API Health', 'http://localhost:10010/health', 'GET'),
            ('Backend API Docs', 'http://localhost:10010/docs', 'GET'),
            ('Backend API Models', 'http://localhost:10010/api/v1/models/', 'GET'),
            ('Backend API Chat', 'http://localhost:10010/api/v1/chat/', 'POST'),
            ('Frontend UI', 'http://localhost:10011/', 'GET'),
            ('Ollama API Tags', 'http://localhost:10104/api/tags', 'GET'),
            ('Ollama API Version', 'http://localhost:10104/api/version', 'GET'),
            ('Hardware Optimizer Health', 'http://localhost:11110/health', 'GET'),
            ('Hardware Optimizer Status', 'http://localhost:11110/status', 'GET'),
            ('AI Agent Orchestrator Health', 'http://localhost:8589/health', 'GET'),
            ('Ollama Integration Health', 'http://localhost:8090/health', 'GET'),
            ('FAISS Vector DB Health', 'http://localhost:10103/health', 'GET'),
            ('Resource Arbitration Health', 'http://localhost:8588/health', 'GET'),
            ('Task Assignment Health', 'http://localhost:8551/health', 'GET'),
            ('Prometheus', 'http://localhost:10200/-/ready', 'GET'),
            ('Grafana', 'http://localhost:10201/api/health', 'GET'),
        ]
        
    def test_endpoint_response_time(self, name: str, url: str, method: str, timeout: int = 10) -> Dict[str, Any]:
        """Test response time for a single endpoint"""
        response_times = []
        status_codes = []
        errors = []
        
        # Test each endpoint 5 times to get average
        for i in range(5):
            try:
                start_time = time.time()
                
                if method.upper() == 'GET':
                    response = requests.get(url, timeout=timeout)
                elif method.upper() == 'POST':
                    # For POST endpoints, send minimal valid data
                    if 'chat' in url:
                        data = {"message": "test", "model": "tinyllama"}
                        response = requests.post(url, json=data, timeout=timeout)
                    else:
                        response = requests.post(url, timeout=timeout)
                else:
                    response = requests.request(method, url, timeout=timeout)
                
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                response_times.append(response_time_ms)
                status_codes.append(response.status_code)
                
            except requests.exceptions.Timeout:
                errors.append(f"Timeout after {timeout}s")
            except requests.exceptions.ConnectionError:
                errors.append("Connection error")
            except Exception as e:
                errors.append(str(e))
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            # Consider successful if most responses are under 50ms and status codes are OK
            success_responses = sum(1 for code in status_codes if 200 <= code < 400)
            fast_responses = sum(1 for rt in response_times if rt <= 50.0)
            
            is_successful = (success_responses >= 3 and fast_responses >= 3) or avg_response_time <= 50.0
            
            return {
                'name': name,
                'url': url,
                'method': method,
                'avg_response_time_ms': avg_response_time,
                'min_response_time_ms': min_response_time,
                'max_response_time_ms': max_response_time,
                'successful_responses': success_responses,
                'fast_responses': fast_responses,
                'total_tests': len(response_times),
                'status_codes': status_codes,
                'errors': errors,
                'meets_target': avg_response_time <= 50.0,
                'is_successful': is_successful
            }
        else:
            return {
                'name': name,
                'url': url,
                'method': method,
                'avg_response_time_ms': None,
                'errors': errors,
                'meets_target': False,
                'is_successful': False,
                'total_tests': 0
            }
    
    def test_concurrent_load(self, num_concurrent: int = 10) -> Dict[str, Any]:
        """Test response times under concurrent load"""
        print(f"🔄 Testing concurrent load with {num_concurrent} simultaneous requests...")
        
        # Use the fastest endpoints for concurrent testing
        fast_endpoints = [
            'http://localhost:10010/health',
            'http://localhost:8589/health',
            'http://localhost:8090/health',
            'http://localhost:10103/health'
        ]
        
        def make_request(url):
            try:
                start_time = time.time()
                response = requests.get(url, timeout=5)
                end_time = time.time()
                return {
                    'url': url,
                    'response_time_ms': (end_time - start_time) * 1000,
                    'status_code': response.status_code,
                    'success': 200 <= response.status_code < 400
                }
            except Exception as e:
                return {
                    'url': url,
                    'response_time_ms': None,
                    'error': str(e),
                    'success': False
                }
        
        # Create concurrent requests
        requests_to_make = []
        for _ in range(num_concurrent):
            for url in fast_endpoints:
                requests_to_make.append(url)
        
        # Execute concurrent requests
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            future_to_url = {executor.submit(make_request, url): url for url in requests_to_make}
            for future in concurrent.futures.as_completed(future_to_url):
                result = future.result()
                results.append(result)
        
        # Analyze results
        successful_requests = [r for r in results if r.get('success', False)]
        response_times = [r['response_time_ms'] for r in successful_requests if r['response_time_ms'] is not None]
        
        if response_times:
            return {
                'total_requests': len(results),
                'successful_requests': len(successful_requests),
                'avg_response_time_ms': statistics.mean(response_times),
                'min_response_time_ms': min(response_times),
                'max_response_time_ms': max(response_times),
                'under_50ms_count': sum(1 for rt in response_times if rt <= 50.0),
                'under_100ms_count': sum(1 for rt in response_times if rt <= 100.0),
                'success_rate': len(successful_requests) / len(results) * 100,
                'fast_response_rate': sum(1 for rt in response_times if rt <= 50.0) / len(response_times) * 100
            }
        else:
            return {
                'total_requests': len(results),
                'successful_requests': 0,
                'error': 'No successful requests'
            }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive response time validation"""
        print("\n⚡ ULTRATEST: Response Time Validation")
        print("=" * 70)
        print("Testing all endpoints for <50ms response times...")
        
        # Test each endpoint individually
        endpoint_results = []
        fast_endpoints = 0
        slow_endpoints = 0
        failed_endpoints = 0
        
        for name, url, method in self.endpoints:
            print(f"Testing: {name}")
            result = self.test_endpoint_response_time(name, url, method)
            endpoint_results.append(result)
            
            if result['is_successful']:
                if result.get('avg_response_time_ms', float('inf')) <= 50.0:
                    fast_endpoints += 1
                    print(f"  ✅ {result['avg_response_time_ms']:.2f}ms (FAST)")
                else:
                    slow_endpoints += 1
                    print(f"  ⚠️  {result['avg_response_time_ms']:.2f}ms (SLOW)")
            else:
                failed_endpoints += 1
                print(f"  ❌ FAILED - {result.get('errors', ['Unknown error'])[0]}")
        
        # Test concurrent load
        concurrent_results = self.test_concurrent_load(10)
        
        # Calculate overall performance
        total_endpoints = len(endpoint_results)
        operational_endpoints = fast_endpoints + slow_endpoints
        performance_score = (fast_endpoints / total_endpoints * 100) if total_endpoints > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'endpoint_results': endpoint_results,
            'concurrent_load_results': concurrent_results,
            'summary': {
                'total_endpoints_tested': total_endpoints,
                'fast_endpoints': fast_endpoints,
                'slow_endpoints': slow_endpoints,
                'failed_endpoints': failed_endpoints,
                'operational_endpoints': operational_endpoints,
                'performance_score': performance_score,
                'target_response_time_ms': 50.0,
                'meets_performance_target': performance_score >= 70.0  # 70% of endpoints must be fast
            }
        }
    
    def generate_performance_report(self, results: Dict[str, Any]):
        """Generate detailed performance report"""
        print("\n" + "=" * 80)
        print("⚡ ULTRATEST RESPONSE TIME REPORT")
        print("=" * 80)
        print(f"Test Execution Time: {results.get('timestamp', 'Unknown')}")
        
        summary = results.get('summary', {})
        
        print(f"\n📊 ENDPOINT PERFORMANCE SUMMARY:")
        print("-" * 50)
        print(f"Total Endpoints: {summary.get('total_endpoints_tested', 0)}")
        print(f"Fast Endpoints (<50ms): {summary.get('fast_endpoints', 0)}")
        print(f"Slow Endpoints (>50ms): {summary.get('slow_endpoints', 0)}")
        print(f"Failed Endpoints: {summary.get('failed_endpoints', 0)}")
        print(f"Performance Score: {summary.get('performance_score', 0):.1f}%")
        
        target_met = summary.get('meets_performance_target', False)
        if target_met:
            print("✅ PERFORMANCE TARGET ACHIEVED (70%+ fast endpoints)")
        else:
            print("❌ Performance target not met")
        
        # Detailed endpoint results
        print(f"\n🔍 DETAILED ENDPOINT RESULTS:")
        print("-" * 50)
        
        for result in results.get('endpoint_results', []):
            name = result.get('name', 'Unknown')
            avg_time = result.get('avg_response_time_ms')
            meets_target = result.get('meets_target', False)
            is_successful = result.get('is_successful', False)
            
            if is_successful and avg_time is not None:
                icon = "✅" if meets_target else "⚠️"
                print(f"{icon} {name}: {avg_time:.2f}ms")
            else:
                errors = result.get('errors', ['Unknown error'])
                print(f"❌ {name}: {errors[0] if errors else 'Failed'}")
        
        # Concurrent load results
        concurrent = results.get('concurrent_load_results', {})
        if 'error' not in concurrent:
            print(f"\n🔄 CONCURRENT LOAD RESULTS:")
            print("-" * 50)
            print(f"Total Requests: {concurrent.get('total_requests', 0)}")
            print(f"Successful Requests: {concurrent.get('successful_requests', 0)}")
            print(f"Success Rate: {concurrent.get('success_rate', 0):.1f}%")
            print(f"Average Response Time: {concurrent.get('avg_response_time_ms', 0):.2f}ms")
            print(f"Fast Responses (<50ms): {concurrent.get('under_50ms_count', 0)}")
            print(f"Fast Response Rate: {concurrent.get('fast_response_rate', 0):.1f}%")
        
        # Overall assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        print("-" * 50)
        
        achievements = []
        issues = []
        
        if summary.get('fast_endpoints', 0) > 0:
            achievements.append(f"{summary['fast_endpoints']} endpoints with <50ms response times")
        
        if target_met:
            achievements.append("Overall performance target achieved (70%+ fast)")
        else:
            issues.append(f"Performance below target ({summary.get('performance_score', 0):.1f}% vs 70%)")
        
        if concurrent.get('success_rate', 0) >= 90:
            achievements.append(f"High concurrent load success rate ({concurrent.get('success_rate', 0):.1f}%)")
        elif concurrent.get('success_rate', 0) >= 70:
            achievements.append(f"Good concurrent load success rate ({concurrent.get('success_rate', 0):.1f}%)")
        
        if concurrent.get('fast_response_rate', 0) >= 70:
            achievements.append(f"Fast concurrent responses ({concurrent.get('fast_response_rate', 0):.1f}%)")
        
        if summary.get('failed_endpoints', 0) > 0:
            issues.append(f"{summary['failed_endpoints']} endpoints failed to respond")
        
        print("🎉 ACHIEVEMENTS:")
        for achievement in achievements:
            print(f"   ✅ {achievement}")
        
        if issues:
            print("\n⚠️  AREAS FOR IMPROVEMENT:")
            for issue in issues:
                print(f"   ❌ {issue}")
        
        success_rate = (len(achievements) / (len(achievements) + len(issues))) * 100 if (achievements or issues) else 0
        print(f"\n📈 Response Time Success Rate: {success_rate:.1f}%")
        
        return target_met and success_rate >= 75

def main():
    """Run comprehensive response time validation"""
    print("🚀 Starting ULTRATEST Response Time Validation")
    
    validator = UltratestResponseTimeValidator()
    results = validator.run_comprehensive_test()
    
    # Generate report
    success = validator.generate_performance_report(results)
    
    # Save results
    with open('/opt/sutazaiapp/tests/ultratest_response_times_report.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Full report saved to: /opt/sutazaiapp/tests/ultratest_response_times_report.json")
    
    if success:
        print("\n🎉 RESPONSE TIME VALIDATION SUCCESSFUL!")
        return 0
    else:
        print("\n⚠️  RESPONSE TIME NEEDS IMPROVEMENT")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())