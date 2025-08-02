#!/usr/bin/env python3
"""Test all chat endpoints to diagnose the issue"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sutazai-default-token"
}

def test_endpoint(name, endpoint, method, payload):
    """Test a specific endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Endpoint: {endpoint}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print('='*60)
    
    try:
        if method == "POST":
            response = requests.post(
                f"{BASE_URL}{endpoint}",
                headers=HEADERS,
                json=payload,
                timeout=10
            )
        else:
            response = requests.get(
                f"{BASE_URL}{endpoint}",
                headers=HEADERS,
                timeout=10
            )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response Structure:")
            print(f"- Keys: {list(result.keys())}")
            
            # Extract response text based on endpoint
            if endpoint == "/api/v1/coordinator/think":
                if "result" in result and isinstance(result["result"], dict):
                    output = result["result"].get("output", "No output")
                    print(f"- Output: {output[:200]}...")
                else:
                    print(f"- Raw: {str(result)[:200]}...")
            elif endpoint == "/api/v1/chat":
                print(f"- Response: {result.get('response', 'No response')[:200]}...")
            else:
                print(f"- Raw: {str(result)[:200]}...")
                
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {str(e)}")
    
    time.sleep(1)

def main():
    print("SutazAI Chat Endpoints Test")
    print("Testing all chat-related endpoints...")
    
    # Test different endpoints
    tests = [
        # Coordinator think endpoint (used by automation Coordinator)
        ("Coordinator Think - Test", "/api/v1/coordinator/think", "POST", {
            "input_data": {"text": "test"},
            "reasoning_type": "strategic"
        }),
        
        # Chat endpoint
        ("Chat - Test", "/api/v1/chat", "POST", {
            "message": "test",
            "model": "llama3.2:1b"
        }),
        
        # Simple chat endpoint
        ("Simple Chat - Test", "/simple-chat", "POST", {
            "message": "test"
        }),
        
        # Models list
        ("List Models", "/api/v1/models", "GET", None),
        
        # Health check
        ("Health Check", "/health", "GET", None),
        
        # Coordinator status
        ("Coordinator Status", "/api/v1/coordinator/status", "GET", None),
        
        # Coordinator capabilities
        ("Coordinator Capabilities", "/api/v1/coordinator/capabilities", "GET", None)
    ]
    
    for name, endpoint, method, payload in tests:
        test_endpoint(name, endpoint, method, payload)
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)

if __name__ == "__main__":
    main()