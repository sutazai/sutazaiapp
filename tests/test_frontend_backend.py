#!/usr/bin/env python3
"""Test frontend-backend integration"""

import requests
import json
import time

def test_backend_health():
    """Test backend health endpoint"""
    print("Testing backend health...")
    try:
        response = requests.get("http://localhost:10200/health")
        if response.status_code == 200:
            print("✅ Backend is healthy:", response.json())
            return True
        else:
            print("❌ Backend returned status:", response.status_code)
            return False
    except Exception as e:
        print("❌ Failed to connect to backend:", e)
        return False

def test_chat_endpoint():
    """Test chat endpoint with a simple message"""
    print("\nTesting chat endpoint...")
    try:
        # Use the simple chat endpoint that doesn't require Ollama
        payload = {
            "message": "Hello JARVIS",
            "agent": "default",
            "stream": False
        }
        
        # First try the simple endpoint
        response = requests.post(
            "http://localhost:10200/api/v1/simple_chat/", 
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            print("✅ Chat endpoint working:", response.json())
            return True
        else:
            print("❌ Chat endpoint returned status:", response.status_code)
            print("Response:", response.text)
            return False
    except requests.Timeout:
        print("❌ Chat endpoint timed out - Ollama connection issue")
        return False
    except Exception as e:
        print("❌ Failed to call chat endpoint:", e)
        return False

def test_models_endpoint():
    """Test models endpoint"""
    print("\nTesting models endpoint...")
    try:
        response = requests.get("http://localhost:10200/api/v1/models")
        if response.status_code == 200:
            print("✅ Models endpoint working:", response.json())
            return True
        else:
            print("❌ Models endpoint returned status:", response.status_code)
            return False
    except Exception as e:
        print("❌ Failed to call models endpoint:", e)
        return False

def test_agents_endpoint():
    """Test agents endpoint"""
    print("\nTesting agents endpoint...")
    try:
        response = requests.get("http://localhost:10200/api/v1/agents")
        if response.status_code == 200:
            print("✅ Agents endpoint working:", response.json())
            return True
        else:
            print("❌ Agents endpoint returned status:", response.status_code)
            return False
    except Exception as e:
        print("❌ Failed to call agents endpoint:", e)
        return False

def test_frontend():
    """Test frontend availability"""
    print("\nTesting frontend...")
    try:
        response = requests.get("http://localhost:11000", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            return True
        else:
            print("❌ Frontend returned status:", response.status_code)
            return False
    except Exception as e:
        print("❌ Failed to connect to frontend:", e)
        return False

def main():
    print("=" * 60)
    print("JARVIS Frontend-Backend Integration Test")
    print("=" * 60)
    
    results = {
        "Backend Health": test_backend_health(),
        "Frontend Access": test_frontend(),
        "Models API": test_models_endpoint(),
        "Agents API": test_agents_endpoint(),
        "Chat API": test_chat_endpoint(),
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - System is fully operational!")
    else:
        print("⚠️ SOME TESTS FAILED - System needs attention")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())