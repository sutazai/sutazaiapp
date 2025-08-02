#!/usr/bin/env python3
"""
Quick Validation Test for SutazAI System
"""
import requests
import json
from datetime import datetime

def test_system():
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    base_url = "http://localhost:8000"
    
    # Test basic endpoints
    tests = [
        ("health", "GET", "/health"),
        ("root", "GET", "/"),
        ("agents", "GET", "/agents"),
        ("models", "GET", "/models"),
        ("simple_chat", "POST", "/simple-chat", {"message": "Hello"}),
        ("system_status", "GET", "/api/v1/system/status"),
    ]
    
    for test_name, method, endpoint, *data in tests:
        try:
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
            else:
                response = requests.post(f"{base_url}{endpoint}", json=data[0] if data else {}, timeout=10)
            
            results['tests'][test_name] = {
                'status': 'PASS' if response.status_code in [200, 201] else 'FAIL',
                'status_code': response.status_code,
                'response_size': len(response.text) if response.text else 0
            }
            print(f"✅ {test_name}: PASS ({response.status_code})")
        except Exception as e:
            results['tests'][test_name] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"❌ {test_name}: ERROR ({str(e)[:50]})")
    
    # Test database
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='localhost', port=5432, database='sutazai_db',
            user='sutazai', password='sutazai123'
        )
        conn.close()
        results['tests']['database'] = {'status': 'PASS'}
        print("✅ database: PASS")
    except Exception as e:
        results['tests']['database'] = {'status': 'ERROR', 'error': str(e)}
        print(f"❌ database: ERROR ({str(e)[:50]})")
    
    # Test Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        results['tests']['redis'] = {'status': 'PASS'}
        print("✅ redis: PASS")
    except Exception as e:
        results['tests']['redis'] = {'status': 'ERROR', 'error': str(e)}
        print(f"❌ redis: ERROR ({str(e)[:50]})")
    
    # Test Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            results['tests']['ollama'] = {'status': 'PASS', 'models_count': len(models)}
            print(f"✅ ollama: PASS ({len(models)} models)")
        else:
            results['tests']['ollama'] = {'status': 'FAIL', 'status_code': response.status_code}
            print(f"❌ ollama: FAIL ({response.status_code})")
    except Exception as e:
        results['tests']['ollama'] = {'status': 'ERROR', 'error': str(e)}
        print(f"❌ ollama: ERROR ({str(e)[:50]})")
    
    # Test Frontend
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        results['tests']['frontend'] = {
            'status': 'PASS' if response.status_code == 200 else 'FAIL',
            'status_code': response.status_code
        }
        status = 'PASS' if response.status_code == 200 else 'FAIL'
        print(f"{'✅' if status == 'PASS' else '❌'} frontend: {status} ({response.status_code})")
    except Exception as e:
        results['tests']['frontend'] = {'status': 'ERROR', 'error': str(e)}
        print(f"❌ frontend: ERROR ({str(e)[:50]})")
    
    # Summary
    total_tests = len(results['tests'])
    passed_tests = sum(1 for t in results['tests'].values() if t.get('status') == 'PASS')
    failed_tests = sum(1 for t in results['tests'].values() if t.get('status') == 'FAIL')
    error_tests = sum(1 for t in results['tests'].values() if t.get('status') == 'ERROR')
    
    results['summary'] = {
        'total': total_tests,
        'passed': passed_tests,
        'failed': failed_tests,
        'errors': error_tests,
        'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
    }
    
    print(f"\n📊 Summary: {passed_tests}/{total_tests} tests passed ({results['summary']['success_rate']:.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'/opt/sutazaiapp/validation_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    test_system()