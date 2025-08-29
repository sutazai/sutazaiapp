#!/usr/bin/env python3
"""
Verify JARVIS fixes are working
"""

import subprocess
import time
import httpx
import sys

def test_streamlit_running():
    """Check if Streamlit is running"""
    try:
        response = httpx.get("http://localhost:11000", timeout=5.0)
        if response.status_code == 200:
            print("✓ Streamlit is running")
            return True
    except:
        pass
    print("✗ Streamlit not accessible")
    return False

def test_backend_connection():
    """Check backend API"""
    try:
        response = httpx.get("http://localhost:10200/health", timeout=5.0)
        if response.status_code == 200:
            print("✓ Backend API is accessible")
            return True
    except:
        pass
    print("✗ Backend API not accessible")
    return False

def test_chat_endpoint():
    """Test chat functionality"""
    try:
        response = httpx.post(
            "http://localhost:10200/api/v1/chat",
            json={"message": "Hello", "model": "GPT-3.5"},
            timeout=10.0
        )
        if response.status_code in [200, 201]:
            print("✓ Chat endpoint working")
            return True
    except:
        pass
    print("✗ Chat endpoint not working")
    return False

def main():
    print("JARVIS Fix Verification")
    print("=" * 40)
    
    tests = [
        test_streamlit_running,
        test_backend_connection,
        test_chat_endpoint
    ]
    
    results = []
    for test in tests:
        results.append(test())
        time.sleep(1)
    
    print("=" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All systems operational!")
    elif passed > 0:
        print("⚠️ Partial functionality restored")
    else:
        print("❌ Critical issues remain")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
