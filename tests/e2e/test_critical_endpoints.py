#!/usr/bin/env python3
"""
Critical Endpoint E2E Tests - August 20, 2025
Tests the endpoints we just fixed to ensure they're fully operational
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:10010"
HEADERS = {"User-Agent": "e2e-test-client", "Content-Type": "application/json"}

def test_health_endpoint():
    """Test backend health endpoint"""
    print("Testing /health endpoint...")
    try:
        resp = requests.get(f"{BASE_URL}/health", headers=HEADERS)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data["status"] == "healthy", f"Status not healthy: {data}"
        print("✅ Health endpoint working")
        return True
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
        return False

def test_agents_endpoint():
    """Test /api/v1/agents endpoint"""
    print("Testing /api/v1/agents endpoint...")
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/agents", headers=HEADERS)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert isinstance(data, list), "Response should be a list"
        assert len(data) > 0, "Should have at least one agent"
        print(f"✅ Agents endpoint working - {len(data)} agents found")
        return True
    except Exception as e:
        print(f"❌ Agents endpoint failed: {e}")
        return False

def test_models_endpoint():
    """Test /api/v1/models/ endpoint"""
    print("Testing /api/v1/models/ endpoint...")
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/models/", headers=HEADERS)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert "models" in data, "Response should have 'models' key"
        assert len(data["models"]) > 0, "Should have at least one model"
        print(f"✅ Models endpoint working - {len(data['models'])} models found")
        return True
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")
        return False

def test_chat_endpoint():
    """Test /api/v1/simple-chat endpoint"""
    print("Testing /api/v1/simple-chat endpoint...")
    try:
        payload = {
            "message": "Hello, this is an E2E test. Please respond briefly.",
            "model": "tinyllama:latest",
            "temperature": 0.7,
            "max_tokens": 50
        }
        resp = requests.post(f"{BASE_URL}/api/v1/simple-chat", 
                            headers=HEADERS, 
                            json=payload,
                            timeout=30)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert "response" in data, "Response should have 'response' key"
        assert len(data["response"]) > 0, "Should have a non-empty response"
        print(f"✅ Chat endpoint working - Response: {data['response'][:100]}...")
        return True
    except Exception as e:
        print(f"❌ Chat endpoint failed: {e}")
        return False

def test_mesh_endpoints():
    """Test mesh system endpoints"""
    print("Testing mesh system endpoints...")
    results = []
    
    # Test mesh health
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/mesh/v2/health", headers=HEADERS)
        assert resp.status_code == 200, f"Mesh health failed: {resp.status_code}"
        print("✅ Mesh health endpoint working")
        results.append(True)
    except Exception as e:
        print(f"❌ Mesh health failed: {e}")
        results.append(False)
    
    # Test mesh services
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/mesh/v2/services", headers=HEADERS)
        assert resp.status_code == 200, f"Mesh services failed: {resp.status_code}"
        data = resp.json()
        service_count = data.get("count", 0)
        print(f"✅ Mesh services endpoint working - {service_count} services found")
        results.append(True)
    except Exception as e:
        print(f"❌ Mesh services failed: {e}")
        results.append(False)
    
    return all(results)

def test_ollama_direct():
    """Test Ollama service directly"""
    print("Testing Ollama service directly...")
    try:
        resp = requests.get("http://localhost:10104/api/tags")
        assert resp.status_code == 200, f"Ollama tags failed: {resp.status_code}"
        data = resp.json()
        model_count = len(data.get("models", []))
        print(f"✅ Ollama service working - {model_count} models available")
        return True
    except Exception as e:
        print(f"❌ Ollama direct test failed: {e}")
        return False

def main():
    """Run all critical E2E tests"""
    print("=" * 60)
    print("CRITICAL ENDPOINT E2E TESTS")
    print("Testing endpoints fixed on August 20, 2025")
    print("=" * 60)
    
    tests = [
        ("Backend Health", test_health_endpoint),
        ("Agents API", test_agents_endpoint),
        ("Models API", test_models_endpoint),
        ("Chat API", test_chat_endpoint),
        ("Mesh System", test_mesh_endpoints),
        ("Ollama Direct", test_ollama_direct)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        results[test_name] = test_func()
        time.sleep(0.5)  # Small delay between tests
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL CRITICAL ENDPOINTS WORKING!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())