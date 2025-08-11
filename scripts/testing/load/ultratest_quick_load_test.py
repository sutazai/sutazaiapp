#!/usr/bin/env python3
"""
ULTRATEST Quick Load Testing
Fast test to verify system can handle concurrent users.
"""

import requests
import time
import json
import concurrent.futures
from datetime import datetime

def make_request(user_id, endpoint_url):
    """Make a single request as a simulated user"""
    try:
        start_time = time.time()
        response = requests.get(endpoint_url, timeout=5)
        end_time = time.time()
        
        return {
            'user_id': user_id,
            'response_time_ms': (end_time - start_time) * 1000,
            'status_code': response.status_code,
            'success': 200 <= response.status_code < 400
        }
    except Exception as e:
        return {
            'user_id': user_id,
            'error': str(e),
            'success': False
        }

def run_concurrent_test(num_users, endpoint_url):
    """Run concurrent test with specified number of users"""
    print(f"🔥 Testing {num_users} concurrent users against {endpoint_url}")
    
    # Use ThreadPoolExecutor for true concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
        # Submit all requests simultaneously
        futures = [executor.submit(make_request, i, endpoint_url) for i in range(num_users)]
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
    
    return results

def main():
    """Run quick load testing"""
    print("🚀 ULTRATEST: Quick Load Testing")
    print("=" * 50)
    
    # Test endpoints (using fastest ones)
    endpoints = [
        ('Health Check', 'http://localhost:10010/health'),
        ('Hardware Optimizer', 'http://localhost:11110/health'),
        ('AI Orchestrator', 'http://localhost:8589/health'),
    ]
    
    all_results = {}
    
    # Test different user loads
    user_loads = [25, 50, 100]
    
    for users in user_loads:
        print(f"\n📊 Testing with {users} concurrent users:")
        print("-" * 40)
        
        load_results = {}
        
        for name, url in endpoints:
            print(f"Testing {name}...", end=" ")
            
            try:
                results = run_concurrent_test(users, url)
                
                # Analyze results
                successful = [r for r in results if r.get('success', False)]
                failed = len(results) - len(successful)
                
                if successful:
                    response_times = [r['response_time_ms'] for r in successful if 'response_time_ms' in r]
                    avg_response = sum(response_times) / len(response_times) if response_times else 0
                    success_rate = len(successful) / len(results) * 100
                    
                    print(f"✅ {success_rate:.1f}% success, {avg_response:.2f}ms avg")
                    
                    load_results[name] = {
                        'total_requests': len(results),
                        'successful_requests': len(successful),
                        'failed_requests': failed,
                        'success_rate': success_rate,
                        'avg_response_time_ms': avg_response,
                        'min_response_time_ms': min(response_times) if response_times else 0,
                        'max_response_time_ms': max(response_times) if response_times else 0
                    }
                else:
                    print(f"❌ All requests failed")
                    load_results[name] = {
                        'total_requests': len(results),
                        'successful_requests': 0,
                        'failed_requests': len(results),
                        'success_rate': 0
                    }
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                load_results[name] = {'error': str(e)}
        
        all_results[f"{users}_users"] = load_results
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("📋 LOAD TESTING SUMMARY")
    print("=" * 60)
    
    # Check if system handles 100+ users
    if "100_users" in all_results:
        results_100 = all_results["100_users"]
        
        # Calculate overall performance for 100 users
        total_success_rates = []
        total_avg_response_times = []
        
        for endpoint_name, stats in results_100.items():
            if 'success_rate' in stats:
                total_success_rates.append(stats['success_rate'])
            if 'avg_response_time_ms' in stats:
                total_avg_response_times.append(stats['avg_response_time_ms'])
        
        if total_success_rates:
            overall_success_rate = sum(total_success_rates) / len(total_success_rates)
            overall_avg_response = sum(total_avg_response_times) / len(total_avg_response_times) if total_avg_response_times else 0
            
            print(f"100 Concurrent Users Test Results:")
            print(f"  Overall Success Rate: {overall_success_rate:.1f}%")
            print(f"  Overall Avg Response Time: {overall_avg_response:.2f}ms")
            
            # Determine if targets are met
            handles_100_users = overall_success_rate >= 80  # 80% success rate under load
            reasonable_performance = overall_avg_response <= 100  # 100ms under load is reasonable
            
            if handles_100_users:
                print("  ✅ SUCCESSFULLY HANDLES 100+ CONCURRENT USERS")
            else:
                print("  ❌ Struggles with 100 concurrent users")
            
            if reasonable_performance:
                print("  ✅ REASONABLE PERFORMANCE UNDER LOAD")
            else:
                print("  ⚠️  Performance degrades under load")
        
        # Show progression across user loads
        print(f"\nLoad Testing Progression:")
        print(f"{'Users':<8} {'Avg Success Rate':<18} {'Avg Response Time':<20}")
        print("-" * 50)
        
        for load_key in ['25_users', '50_users', '100_users']:
            if load_key in all_results:
                load_data = all_results[load_key]
                success_rates = [stats.get('success_rate', 0) for stats in load_data.values() if 'success_rate' in stats]
                response_times = [stats.get('avg_response_time_ms', 0) for stats in load_data.values() if 'avg_response_time_ms' in stats]
                
                if success_rates:
                    avg_success = sum(success_rates) / len(success_rates)
                    avg_response = sum(response_times) / len(response_times) if response_times else 0
                    
                    users_num = load_key.split('_')[0]
                    print(f"{users_num:<8} {avg_success:<18.1f} {avg_response:<20.2f}")
    
    # Save results
    report = {
        'timestamp': datetime.now().isoformat(),
        'load_test_results': all_results,
        'test_summary': {
            'endpoints_tested': len(endpoints),
            'max_concurrent_users': max(user_loads),
            'load_levels_tested': user_loads
        }
    }
    
    with open('/opt/sutazaiapp/tests/ultratest_quick_load_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Report saved: tests/ultratest_quick_load_report.json")
    
    # Determine overall success
    if "100_users" in all_results:
        results_100 = all_results["100_users"]
        success_rates = [stats.get('success_rate', 0) for stats in results_100.values() if 'success_rate' in stats]
        
        if success_rates:
            overall_success = sum(success_rates) / len(success_rates)
            if overall_success >= 80:
                print("\n🎉 LOAD TESTING SUCCESSFUL - System handles 100+ users!")
                return 0
    
    print("\n⚠️  LOAD TESTING NEEDS IMPROVEMENT")
    return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())