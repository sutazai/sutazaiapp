#!/usr/bin/env python3
"""Test essential AI agents"""

import asyncio
import aiohttp
import json
import sys
import socket

async def test_ollama():
    """Test Ollama connectivity"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:11434/api/tags') as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m['name'] for m in data.get('models', [])]
                    print(f"✅ Ollama is running with models: {models}")
                    return True
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False

async def test_redis():
    """Test Redis connectivity"""
    try:
        # Simple socket test for Redis
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 6379))
        sock.close()
        
        if result == 0:
            print("✅ Redis is running (port 6379 accessible)")
            return True
        else:
            print("❌ Redis port 6379 not accessible")
            return False
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return False

async def test_brain():
    """Test AGI Brain"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8900/health') as resp:
                if resp.status == 200:
                    print("✅ AGI Brain is healthy")
                    return True
    except Exception as e:
        print(f"❌ AGI Brain test failed: {e}")
        return False

async def test_simple_inference():
    """Test simple AI inference"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": "tinyllama:latest",
                "prompt": "Hello, how are you?",
                "stream": False
            }
            async with session.post('http://localhost:11434/api/generate', json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response = data.get('response', '')
                    print(f"✅ AI Response: {response[:100]}...")
                    return True
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

async def main():
    print("\n🧪 Testing Essential AI Services")
    print("=" * 40)
    
    tests = [
        test_ollama(),
        test_redis(),
        test_brain(),
        test_simple_inference()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    success_count = sum(1 for r in results if r is True)
    total_count = len(tests)
    
    print("\n" + "=" * 40)
    print(f"✅ Passed: {success_count}/{total_count} tests")
    
    if success_count == total_count:
        print("\n🎉 All essential AI services are working!")
        return 0
    else:
        print("\n⚠️  Some services need attention")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))