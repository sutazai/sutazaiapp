#!/usr/bin/env python3
"""Complete System Integration Test for JARVIS"""

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
        payload = {
            "message": "Hello JARVIS",
            "agent": "default",
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:10200/api/v1/simple_chat/", 
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Chat endpoint working:")
            print("   Response:", data.get("response"))
            print("   Model:", data.get("model"))
            print("   Success:", data.get("success"))
            return True
        else:
            print("❌ Chat endpoint returned status:", response.status_code)
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
            data = response.json()
            print("✅ Models endpoint working:")
            print("   Available models:", data.get("models"))
            print("   Count:", data.get("count"))
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
            agents = response.json()
            print("✅ Agents endpoint working:")
            print(f"   {len(agents)} agents available")
            for agent in agents[:3]:  # Show first 3 agents
                print(f"   - {agent.get('name')} ({agent.get('status')})")
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
            print("✅ Frontend is accessible at http://localhost:11000")
            return True
        else:
            print("❌ Frontend returned status:", response.status_code)
            return False
    except Exception as e:
        print("❌ Failed to connect to frontend:", e)
        return False

def test_voice_endpoint():
    """Test voice health endpoint"""
    print("\nTesting voice system...")
    try:
        response = requests.get("http://localhost:10200/api/v1/voice/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Voice system status:")
            print("   TTS Available:", data.get("tts_available"))
            print("   ASR Available:", data.get("asr_available"))
            print("   Wake Word Available:", data.get("wake_word_available"))
            return True
        else:
            print("❌ Voice endpoint returned status:", response.status_code)
            return False
    except Exception as e:
        print("❌ Failed to call voice endpoint:", e)
        return False

def test_websocket_chat():
    """Test WebSocket chat connectivity"""
    print("\nTesting WebSocket connectivity...")
    try:
        # Test if WebSocket endpoint is accessible
        response = requests.get("http://localhost:10200/api/v1/chat/health")
        if response.status_code == 200:
            print("✅ WebSocket chat endpoint is available")
            return True
        else:
            print("⚠️ WebSocket endpoint returned status:", response.status_code)
            return False
    except Exception as e:
        print("⚠️ WebSocket test skipped:", e)
        return False

def main():
    print("=" * 70)
    print("JARVIS SYSTEM INTEGRATION TEST")
    print("=" * 70)
    
    results = {
        "Backend Health": test_backend_health(),
        "Frontend Access": test_frontend(),
        "Models API": test_models_endpoint(),
        "Agents API": test_agents_endpoint(),
        "Chat API": test_chat_endpoint(),
        "Voice System": test_voice_endpoint(),
        "WebSocket Chat": test_websocket_chat(),
    }
    
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY:")
    print("=" * 70)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
    
    passed = sum(results.values())
    total = len(results)
    percentage = (passed / total) * 100
    
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed ({percentage:.1f}%)")
    
    if percentage == 100:
        print("✅ ALL TESTS PASSED - System is fully operational!")
        print("\nYou can now access:")
        print("  - Frontend UI: http://localhost:11000")
        print("  - Backend API: http://localhost:10200/docs")
        print("  - Chat directly with: curl -X POST http://localhost:10200/api/v1/simple_chat/")
    elif percentage >= 70:
        print("⚠️ SYSTEM MOSTLY OPERATIONAL - Some features may be limited")
    else:
        print("❌ SYSTEM NEEDS ATTENTION - Critical components failing")
    
    print("=" * 70)
    
    return 0 if percentage >= 70 else 1

if __name__ == "__main__":
    exit(main())