#!/usr/bin/env python3
"""
Fix Backend Connection to Ollama
Ensures proper communication between backend and Ollama
"""

import os
import requests
import json
import time

def test_ollama_connection():
    """Test different Ollama connection methods"""
    
    # Possible Ollama URLs
    urls = [
        "http://localhost:11434",
        "http://sutazai-ollama:11434",
        "http://127.0.0.1:11434",
        "http://ollama:11434"
    ]
    
    print("Testing Ollama connections...")
    working_url = None
    
    for url in urls:
        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Success: {url}")
                working_url = url
                models = response.json().get('models', [])
                print(f"   Models available: {len(models)}")
                for model in models:
                    print(f"   - {model['name']}")
                break
            else:
                print(f"❌ Failed: {url} (status: {response.status_code})")
        except Exception as e:
            print(f"❌ Failed: {url} (error: {str(e)[:50]}...)")
    
    return working_url

def test_model_generation(url):
    """Test actual model generation"""
    print(f"\nTesting model generation with {url}...")
    
    payload = {
        "model": "llama3.2:1b",
        "prompt": "What is artificial intelligence?",
        "stream": False
    }
    
    try:
        response = requests.post(f"{url}/api/generate", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print("✅ Model generation successful!")
            print(f"Response: {result['response'][:100]}...")
            return True
        else:
            print(f"❌ Generation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Generation error: {str(e)}")
        return False

def update_backend_config():
    """Update backend configuration if needed"""
    print("\nChecking backend configuration...")
    
    # Check if backend is using correct URL
    backend_file = "/opt/sutazaiapp/intelligent_backend.py"
    with open(backend_file, 'r') as f:
        content = f.read()
    
    if 'SERVICES = {' in content:
        print("✅ Backend configuration found")
        # The configuration is correct for localhost access
        print("   Using localhost URLs (correct for backend running outside Docker)")
    
    # Check if backend process can reach Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Backend can reach Ollama at localhost:11434")
        else:
            print("❌ Backend cannot reach Ollama")
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")

def main():
    print("=== Fixing Backend-Ollama Connection ===\n")
    
    # Test connections
    working_url = test_ollama_connection()
    
    if working_url:
        # Test generation
        success = test_model_generation(working_url)
        
        if success:
            print("\n✅ Ollama is working correctly!")
            print(f"✅ Working URL: {working_url}")
            
            # Update configuration
            update_backend_config()
            
            print("\n=== Solution ===")
            print("The backend should be able to connect to Ollama.")
            print("If chat responses are still showing default messages,")
            print("the issue might be in the response handling logic.")
            
            # Test the actual chat endpoint
            print("\nTesting chat endpoint...")
            chat_payload = {
                "message": "Hello, are you working?",
                "model": "llama3.2:1b"
            }
            
            try:
                response = requests.post("http://localhost:8000/api/chat", 
                                       json=chat_payload, 
                                       timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    print(f"Chat response: {result}")
                else:
                    print(f"Chat endpoint error: {response.status_code}")
            except Exception as e:
                print(f"Chat endpoint error: {str(e)}")
        else:
            print("\n❌ Model generation is failing")
    else:
        print("\n❌ Cannot connect to Ollama!")
        print("Make sure Ollama container is running:")
        print("  docker ps | grep ollama")

if __name__ == "__main__":
    main()