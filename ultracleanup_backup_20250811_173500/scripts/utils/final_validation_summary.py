#!/usr/bin/env python3
"""
Final Validation Summary for Hardware Resource Optimizer
=======================================================

This script performs a final validation of all agent capabilities
and generates a comprehensive summary report.

Author: QA Team Lead
Version: 1.0.0
"""

import json
import time
import requests
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FinalValidation')

def check_agent_health(base_url):
    """Check agent health and extract key information"""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            return {
                'available': True,
                'status': health_data.get('status'),
                'agent_id': health_data.get('agent'),
                'description': health_data.get('description'),
                'docker_available': health_data.get('docker_available'),
                'system_status': health_data.get('system_status', {})
            }
        else:
            return {'available': False, 'error': f'HTTP {response.status_code}'}
    except Exception as e:
        return {'available': False, 'error': str(e)}

def validate_all_endpoints(base_url):
    """Quick validation of all endpoints"""
    endpoints = [
        ('GET', '/health'),
        ('GET', '/status'),
        ('POST', '/optimize/memory'),
        ('POST', '/optimize/cpu'),
        ('POST', '/optimize/disk'),
        ('POST', '/optimize/docker'),
        ('POST', '/optimize/all'),
        ('GET', '/analyze/storage', {'path': '/tmp'}),
        ('GET', '/analyze/storage/duplicates', {'path': '/tmp'}),
        ('GET', '/analyze/storage/large-files', {'path': '/', 'min_size_mb': 100}),
        ('GET', '/analyze/storage/report'),
        ('POST', '/optimize/storage', {'dry_run': True}),
        ('POST', '/optimize/storage/duplicates', {'path': '/tmp', 'dry_run': True}),
        ('POST', '/optimize/storage/cache'),
        ('POST', '/optimize/storage/compress', {'path': '/var/log', 'days_old': 30}),
        ('POST', '/optimize/storage/logs')
    ]
    
    results = {'total': len(endpoints), 'working': 0, 'failed': []}
    
    for method, endpoint, *params in endpoints:
        try:
            params_dict = params[0] if params else None
            
            if method == 'GET':
                response = requests.get(f"{base_url}{endpoint}", params=params_dict, timeout=30)
            else:
                response = requests.post(f"{base_url}{endpoint}", params=params_dict, timeout=30)
            
            if response.status_code == 200:
                results['working'] += 1
            else:
                results['failed'].append(f"{method} {endpoint}: HTTP {response.status_code}")
                
        except Exception as e:
            results['failed'].append(f"{method} {endpoint}: {str(e)}")
    
    results['success_rate'] = (results['working'] / results['total']) * 100
    return results

def test_safety_mechanisms(base_url):
    """Test key safety mechanisms"""
    safety_tests = []
    
    # Test dry run functionality
    try:
        response = requests.post(f"{base_url}/optimize/storage", 
                               params={'dry_run': True}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            dry_run_working = data.get('dry_run') == True
            safety_tests.append({'test': 'dry_run', 'passed': dry_run_working})
        else:
            safety_tests.append({'test': 'dry_run', 'passed': False, 'error': f'HTTP {response.status_code}'})
    except Exception as e:
        safety_tests.append({'test': 'dry_run', 'passed': False, 'error': str(e)})
    
    # Test protected path handling
    try:
        response = requests.get(f"{base_url}/analyze/storage", 
                              params={'path': '/etc'}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            # Should either be denied or return error status
            path_protection_working = data.get('status') == 'error' or 'not accessible' in data.get('error', '').lower()
            safety_tests.append({'test': 'path_protection', 'passed': path_protection_working})
        else:
            safety_tests.append({'test': 'path_protection', 'passed': True, 'note': 'HTTP error as expected'})
    except Exception as e:
        safety_tests.append({'test': 'path_protection', 'passed': False, 'error': str(e)})
    
    return safety_tests

def performance_quick_check(base_url):
    """Quick performance validation"""
    start_time = time.time()
    
    # Test a few key endpoints for response time
    test_endpoints = [
        ('GET', '/health'),
        ('GET', '/status'),
        ('POST', '/optimize/memory'),
        ('GET', '/analyze/storage', {'path': '/tmp'})
    ]
    
    times = []
    for method, endpoint, *params in test_endpoints:
        try:
            req_start = time.time()
            params_dict = params[0] if params else None
            
            if method == 'GET':
                response = requests.get(f"{base_url}{endpoint}", params=params_dict, timeout=30)
            else:
                response = requests.post(f"{base_url}{endpoint}", params=params_dict, timeout=30)
            
            req_time = time.time() - req_start
            if response.status_code == 200:
                times.append(req_time)
                
        except Exception:
            continue
    
    if times:
        return {
            'avg_response_time': sum(times) / len(times),
            'max_response_time': max(times),
            'min_response_time': min(times),
            'requests_tested': len(times)
        }
    else:
        return {'error': 'No successful requests'}

def main():
    """Main validation execution"""
    base_url = "http://localhost:8116"
    
    print("="*80)
    print("HARDWARE RESOURCE OPTIMIZER - FINAL VALIDATION SUMMARY")
    print("="*80)
    print(f"Agent URL: {base_url}")
    print(f"Validation Time: {datetime.now().isoformat()}")
    print()
    
    # 1. Check agent health
    print("1. AGENT HEALTH CHECK")
    print("-" * 40)
    health = check_agent_health(base_url)
    
    if health['available']:
        print(f"✅ Agent Status: {health['status']}")
        print(f"✅ Agent ID: {health['agent_id']}")
        print(f"✅ Description: {health['description']}")
        print(f"✅ Docker Available: {health['docker_available']}")
        
        system = health['system_status']
        print(f"✅ System Status:")
        print(f"   - CPU: {system.get('cpu_percent', 'N/A')}%")
        print(f"   - Memory: {system.get('memory_percent', 'N/A')}%")
        print(f"   - Disk: {system.get('disk_percent', 0):.1f}%")
    else:
        print(f"❌ Agent Health Check Failed: {health['error']}")
        return 1
    
    print()
    
    # 2. Validate all endpoints
    print("2. ENDPOINT VALIDATION")
    print("-" * 40)
    endpoints = validate_all_endpoints(base_url)
    
    print(f"Total Endpoints: {endpoints['total']}")
    print(f"Working Endpoints: {endpoints['working']}")
    print(f"Failed Endpoints: {len(endpoints['failed'])}")
    print(f"Success Rate: {endpoints['success_rate']:.1f}%")
    
    if endpoints['failed']:
        print("\nFailed Endpoints:")
        for failure in endpoints['failed']:
            print(f"  ❌ {failure}")
    else:
        print("✅ All endpoints working correctly!")
    
    print()
    
    # 3. Safety mechanism validation
    print("3. SAFETY MECHANISM VALIDATION")
    print("-" * 40)
    safety = test_safety_mechanisms(base_url)
    
    safety_passed = 0
    for test in safety:
        if test['passed']:
            print(f"✅ {test['test']}: PASSED")
            safety_passed += 1
        else:
            print(f"❌ {test['test']}: FAILED - {test.get('error', 'Unknown error')}")
    
    print(f"\nSafety Tests Passed: {safety_passed}/{len(safety)}")
    
    print()
    
    # 4. Performance check
    print("4. PERFORMANCE VALIDATION")
    print("-" * 40)
    perf = performance_quick_check(base_url)
    
    if 'error' not in perf:
        print(f"✅ Average Response Time: {perf['avg_response_time']:.3f}s")
        print(f"✅ Max Response Time: {perf['max_response_time']:.3f}s")
        print(f"✅ Min Response Time: {perf['min_response_time']:.3f}s")
        print(f"✅ Requests Tested: {perf['requests_tested']}")
        
        # Performance assessment
        if perf['avg_response_time'] < 1.0:
            print("🚀 Performance Rating: EXCELLENT")
        elif perf['avg_response_time'] < 2.0:
            print("👍 Performance Rating: GOOD")
        elif perf['avg_response_time'] < 5.0:
            print("⚠️ Performance Rating: ACCEPTABLE")
        else:
            print("❌ Performance Rating: NEEDS IMPROVEMENT")
    else:
        print(f"❌ Performance Check Failed: {perf['error']}")
    
    print()
    
    # 5. Overall assessment
    print("5. OVERALL ASSESSMENT")
    print("-" * 40)
    
    overall_score = 0
    total_checks = 4
    
    # Health check (25%)
    if health['available']:
        overall_score += 25
        print("✅ Health Check: PASS (25/25 points)")
    else:
        print("❌ Health Check: FAIL (0/25 points)")
    
    # Endpoint validation (35%)
    endpoint_score = int((endpoints['success_rate'] / 100) * 35)
    overall_score += endpoint_score
    print(f"{'✅' if endpoints['success_rate'] >= 90 else '❌'} Endpoint Validation: {endpoint_score}/35 points ({endpoints['success_rate']:.1f}% success)")
    
    # Safety validation (25%)
    safety_score = int((safety_passed / len(safety)) * 25)
    overall_score += safety_score
    print(f"{'✅' if safety_passed == len(safety) else '❌'} Safety Validation: {safety_score}/25 points ({safety_passed}/{len(safety)} passed)")
    
    # Performance validation (15%)
    if 'error' not in perf:
        if perf['avg_response_time'] < 2.0:
            perf_score = 15
        elif perf['avg_response_time'] < 5.0:
            perf_score = 10
        else:
            perf_score = 5
        overall_score += perf_score
        print(f"✅ Performance Validation: {perf_score}/15 points")
    else:
        print("❌ Performance Validation: 0/15 points")
    
    print(f"\nOVERALL SCORE: {overall_score}/100")
    
    if overall_score >= 90:
        print("🎉 FINAL ASSESSMENT: EXCELLENT - READY FOR PRODUCTION")
        final_grade = "A+"
    elif overall_score >= 80:
        print("👍 FINAL ASSESSMENT: GOOD - READY FOR PRODUCTION")
        final_grade = "A"
    elif overall_score >= 70:
        print("⚠️ FINAL ASSESSMENT: ACCEPTABLE - NEEDS MINOR IMPROVEMENTS")
        final_grade = "B"
    else:
        print("❌ FINAL ASSESSMENT: NEEDS SIGNIFICANT IMPROVEMENTS")
        final_grade = "C"
    
    print(f"GRADE: {final_grade}")
    
    # Generate summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'agent_url': base_url,
        'health_check': health,
        'endpoint_validation': endpoints,
        'safety_validation': safety,
        'performance_validation': perf,
        'overall_score': overall_score,
        'final_grade': final_grade,
        'production_ready': overall_score >= 80
    }
    
    # Save summary
    filename = f"final_validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDetailed validation report saved to: {filename}")
    print("="*80)
    
    # Return appropriate exit code
    return 0 if overall_score >= 80 else 1

if __name__ == "__main__":
    exit(main())