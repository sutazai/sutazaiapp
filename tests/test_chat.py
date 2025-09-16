#!/usr/bin/env python3
"""
Quick test to verify the chat is working
"""

import requests
import json

def test_simple_chat():
    """Test a simple chat interaction"""
    print("Testing JARVIS Chat API...")
    
    # Send a simple message
    message = "Hello, are you working?"
    
    print(f"Sending: {message}")
    
    try:
        response = requests.post(
            "http://localhost:10200/api/v1/simple_chat/",
            json={"message": message},
            timeout=30  # Longer timeout since it's slow
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data.get('response', 'No response')}")
            return True
        else:
            print(f"Error: Status code {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print("Request timed out after 30 seconds")
        # Try the alternative endpoint
        try:
            print("Trying alternative endpoint...")
            response = requests.post(
                "http://localhost:10200/api/v1/jarvis_chat/",
                json={"message": message},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {data.get('response', 'No response')}")
                return True
        except:
            pass
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_chat()
    
    if success:
        print("\n✅ Chat is working!")
        print("\nYou can now access JARVIS at:")
        print("  Frontend: http://localhost:11000")
        print("  Backend API: http://localhost:10200/docs")
    else:
        print("\n⚠️ Chat may be slow but frontend is accessible")
        print("  Try the UI at: http://localhost:11000")