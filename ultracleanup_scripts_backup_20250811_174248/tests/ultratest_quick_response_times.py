#!/usr/bin/env python3
"""
ULTRATEST Quick Response Time Validation
Fast test of key endpoints for <50ms response times.
"""

import requests
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any

def test_endpoint(name: str, url: str, timeout: int = 5) -> Dict[str, Any]:
    """Test a single endpoint response time"""
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        return {
            'name': name,
            'url': url,
            'response_time_ms': response_time_ms,
            'status_code': response.status_code,
            'success': 200 <= response.status_code < 400,
            'meets_target': response_time_ms <= 50.0
        }
    except Exception as e:
        return {
            'name': name,
            'url': url,
            'response_time_ms': None,
            'error': str(e),
            'success': False,
            'meets_target': False
        }

def main():
    """Run quick response time validation"""
    print("🚀 ULTRATEST: Quick Response Time Validation")
    print("=" * 60)
    
    # Key endpoints to test
    endpoints = [
        ('Backend Health', 'http://localhost:10010/health'),
        ('Frontend UI', 'http://localhost:10011/'),
        ('Ollama API', 'http://localhost:10104/api/tags'),
        ('Hardware Optimizer', 'http://localhost:11110/health'),
        ('AI Orchestrator', 'http://localhost:8589/health'),
        ('Ollama Integration', 'http://localhost:8090/health'),
        ('FAISS Vector', 'http://localhost:10103/health'),
        ('Resource Arbitration', 'http://localhost:8588/health'),
        ('Task Assignment', 'http://localhost:8551/health'),
        ('Prometheus', 'http://localhost:10200/-/ready'),
        ('Grafana', 'http://localhost:10201/api/health'),
    ]
    
    results = []
    fast_count = 0
    slow_count = 0
    failed_count = 0
    
    print(f"Testing {len(endpoints)} endpoints...")
    
    for name, url in endpoints:
        print(f"Testing {name}...", end=" ")
        result = test_endpoint(name, url)
        results.append(result)
        
        if result['success']:
            if result['meets_target']:
                fast_count += 1
                print(f"✅ {result['response_time_ms']:.2f}ms")
            else:
                slow_count += 1
                print(f"⚠️ {result['response_time_ms']:.2f}ms (SLOW)")
        else:
            failed_count += 1
            print(f"❌ {result.get('error', 'Failed')}")
    
    # Generate summary
    total_endpoints = len(endpoints)
    operational_endpoints = fast_count + slow_count
    performance_score = (fast_count / total_endpoints * 100) if total_endpoints > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 RESPONSE TIME SUMMARY")
    print("=" * 60)
    print(f"Total Endpoints: {total_endpoints}")
    print(f"Fast (<50ms): {fast_count}")
    print(f"Slow (>50ms): {slow_count}")
    print(f"Failed: {failed_count}")
    print(f"Performance Score: {performance_score:.1f}%")
    
    if performance_score >= 70:
        print("✅ PERFORMANCE TARGET ACHIEVED!")
        success = True
    else:
        print("❌ Performance below 70% target")
        success = False
    
    # Save detailed results
    report = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'summary': {
            'total_endpoints': total_endpoints,
            'fast_endpoints': fast_count,
            'slow_endpoints': slow_count,
            'failed_endpoints': failed_count,
            'performance_score': performance_score,
            'meets_target': performance_score >= 70
        }
    }
    
    with open('/opt/sutazaiapp/tests/ultratest_quick_response_times_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Report saved: tests/ultratest_quick_response_times_report.json")
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())