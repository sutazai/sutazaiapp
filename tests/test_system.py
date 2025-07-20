#!/usr/bin/env python3
"""
SutazAI System Test Script
Tests all major components and provides a quick health check
"""

import requests
import json
import time
import sys

def test_backend():
    """Test backend API"""
    print("Testing Backend API...")
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Backend is healthy:", response.json())
        else:
            print("❌ Backend returned status:", response.status_code)
    except Exception as e:
        print("❌ Backend error:", str(e))

def test_models():
    """Test available models"""
    print("\nTesting Model Availability...")
    try:
        response = requests.get("http://localhost:8000/models")
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Found {len(models)} models:")
            for model in models:
                print(f"   - {model['name']} ({model['details']['parameter_size']})")
        else:
            print("❌ Models endpoint returned:", response.status_code)
    except Exception as e:
        print("❌ Models error:", str(e))

def test_chat():
    """Test chat functionality"""
    print("\nTesting Chat Functionality...")
    try:
        payload = {
            "message": "Hello! Can you tell me about SutazAI?",
            "model": "llama3.2:1b"
        }
        response = requests.post(
            "http://localhost:8000/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            result = response.json()
            print("✅ Chat response received:")
            print(f"   Model: {result.get('model')}")
            print(f"   Response: {result.get('response')[:100]}...")
        else:
            print("❌ Chat returned status:", response.status_code)
    except Exception as e:
        print("❌ Chat error:", str(e))

def test_streamlit():
    """Test Streamlit UI"""
    print("\nTesting Streamlit UI...")
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit UI is accessible")
        else:
            print("❌ Streamlit returned status:", response.status_code)
    except Exception as e:
        print("❌ Streamlit error:", str(e))

def test_ollama():
    """Test Ollama directly"""
    print("\nTesting Ollama Service...")
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama is running with {len(models)} models")
        else:
            print("❌ Ollama returned status:", response.status_code)
    except Exception as e:
        print("❌ Ollama error:", str(e))

def main():
    """Run all tests"""
    print("=" * 60)
    print("SutazAI System Test Suite")
    print("=" * 60)
    
    test_backend()
    test_models()
    test_ollama()
    test_streamlit()
    test_chat()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("- Backend API: http://localhost:8000")
    print("- Streamlit UI: http://localhost:8501")
    print("- API Documentation: http://localhost:8000/docs")
    print("=" * 60)

if __name__ == "__main__":
    main()