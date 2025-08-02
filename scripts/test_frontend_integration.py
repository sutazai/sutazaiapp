#!/usr/bin/env python3
"""Test frontend-backend integration"""

import requests
import time

def test_frontend_health():
    """Check if frontend is accessible"""
    print("Testing frontend health...")
    try:
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            return True
        else:
            print(f"❌ Frontend returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend error: {e}")
        return False

def test_backend_endpoints():
    """Test key backend endpoints"""
    print("\nTesting backend endpoints...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sutazai-default-token"
    }
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:8000/health", headers=headers)
        if response.status_code == 200:
            print("✅ Backend health check passed")
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Backend health check error: {e}")
    
    # Test chat endpoint
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/models/chat",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "test"}],
                "model": "llama3.2:1b"
            },
            timeout=10
        )
        if response.status_code == 200:
            print("✅ Chat endpoint working")
            result = response.json()
            print(f"   Response: {result.get('response', 'No response')[:50]}...")
        else:
            print(f"❌ Chat endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
    
    # Test coordinator think endpoint
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/coordinator/think",
            headers=headers,
            json={
                "input_data": {"text": "test"},
                "reasoning_type": "strategic"
            },
            timeout=5
        )
        if response.status_code == 200:
            print("✅ Coordinator think endpoint working")
        else:
            print(f"❌ Coordinator think endpoint failed: {response.status_code}")
    except requests.exceptions.Timeout:
        print("⚠️ Coordinator think endpoint timed out (this may be normal for complex queries)")
    except Exception as e:
        print(f"❌ Coordinator think endpoint error: {e}")

def main():
    print("SutazAI Frontend Integration Test")
    print("="*50)
    
    # Wait for services to be ready
    print("Waiting for services to stabilize...")
    time.sleep(5)
    
    # Run tests
    frontend_ok = test_frontend_health()
    test_backend_endpoints()
    
    print("\n" + "="*50)
    if frontend_ok:
        print("✅ Frontend is running. Access it at http://localhost:8501")
        print("\nTo test the chat:")
        print("1. Go to http://localhost:8501")
        print("2. Enter 'test' in the chat input at the bottom")
        print("3. Press Enter or click Send")
        print("\nFor advanced features:")
        print("1. Use the dropdown to select '💬 AI Chat Hub'")
        print("2. This provides more configuration options")
    else:
        print("❌ Frontend is not accessible. Check docker logs.")
    
    print("\nTroubleshooting:")
    print("- Check logs: docker logs sutazai-frontend")
    print("- Check backend: docker logs sutazai-backend")
    print("- Restart: docker-compose restart frontend backend")

if __name__ == "__main__":
    main()