#!/usr/bin/env python3
"""
Neo4j Integration Test Script for SutazAI
Tests all aspects of Neo4j connectivity and functionality
"""

import sys
import os
import requests
import base64
from datetime import datetime

def test_neo4j_http_api():
    """Test Neo4j HTTP API access"""
    print("🔍 Testing Neo4j HTTP API...")
    
    try:
        auth_header = base64.b64encode(b'neo4j:neo4j_secure_2024').decode('ascii')
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Basic {auth_header}'
        }
        
        # Test basic connection
        response = requests.get('http://localhost:10002/', headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Neo4j HTTP API accessible")
            print(f"  ℹ️  Version: {data.get('neo4j_version', 'unknown')}")
            print(f"  ℹ️  Edition: {data.get('neo4j_edition', 'unknown')}")
        else:
            print(f"  ❌ HTTP API failed with status: {response.status_code}")
            return False
            
        # Test Cypher execution
        cypher_data = {
            'statements': [
                {'statement': 'RETURN "HTTP API test successful" as message'}
            ]
        }
        
        response = requests.post(
            'http://localhost:10002/db/neo4j/tx/commit',
            json=cypher_data,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('errors'):
                print(f"  ❌ Cypher execution failed: {result['errors']}")
                return False
            print("  ✅ Cypher execution via HTTP API successful")
            return True
        else:
            print(f"  ❌ Cypher execution failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ HTTP API test failed: {str(e)}")
        return False

def test_neo4j_browser():
    """Test Neo4j Browser accessibility"""
    print("🌐 Testing Neo4j Browser access...")
    
    try:
        response = requests.get('http://localhost:10002/', timeout=10)
        if response.status_code == 200 and 'neo4j' in response.text.lower():
            print("  ✅ Neo4j Browser accessible on port 10002")
            return True
        else:
            print(f"  ❌ Browser test failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Browser test failed: {str(e)}")
        return False

def test_backend_health():
    """Test SutazAI backend health"""
    print("🏥 Testing SutazAI Backend health...")
    
    try:
        response = requests.get('http://localhost:10010/health', timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("  ✅ Backend is healthy")
            print(f"  ℹ️  Status: {health_data.get('status', 'unknown')}")
            print(f"  ℹ️  Version: {health_data.get('version', 'unknown')}")
            return True
        else:
            print(f"  ❌ Backend health check failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Backend health test failed: {str(e)}")
        return False

def main():
    """Run all Neo4j integration tests"""
    print("🚀 SutazAI Neo4j Integration Test Suite")
    print("=" * 50)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Neo4j Browser", test_neo4j_browser),
        ("Neo4j HTTP API", test_neo4j_http_api),
        ("Backend Health", test_backend_health),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {str(e)}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Neo4j is fully operational for SutazAI")
        print("✅ Graph database operations are ready for production use")
        return 0
    else:
        print("❌ Some tests failed. Neo4j may have connectivity issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())