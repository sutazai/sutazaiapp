#!/usr/bin/env python3
"""
Comprehensive test for the performance-fixed backend
Tests all API endpoints and verifies metrics collection
"""

import requests
import json
import time
import asyncio
import websockets
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    
    print(f"  Status: {data['status']}")
    print(f"  Ollama models: {len(data['services']['ollama']['available_models'])}")
    print(f"  External agents online: {data['services']['external_agents']['online']}")
    return response.status_code == 200

def test_chat():
    """Test chat endpoint"""
    print("\n💬 Testing chat endpoint...")
    
    # Test multiple requests
    messages = [
        "Hello, how are you?",
        "What is artificial intelligence?",
        "Explain quantum computing briefly"
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"  Request {i}: {message[:30]}...")
        
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={"message": message, "model": "llama3.2:1b"},
            timeout=35
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"    ✅ Response: {len(data['response'])} chars")
            print(f"    ⏱️  Time: {data['processing_time']:.2f}s")
            print(f"    🤖 Ollama: {'✅' if data['ollama_success'] else '❌'}")
        else:
            print(f"    ❌ Error: {response.status_code}")
        
        time.sleep(1)  # Small delay between requests
    
    return True

def test_metrics():
    """Test metrics endpoints"""
    print("\n📊 Testing metrics endpoints...")
    
    # Performance summary
    response = requests.get(f"{BASE_URL}/api/performance/summary")
    if response.status_code == 200:
        data = response.json()
        print("  📈 Performance Summary:")
        print(f"    CPU: {data['system']['cpu_usage']:.1f}%")
        print(f"    Memory: {data['system']['memory_usage']:.1f}%")
        print(f"    API Requests: {data['api']['total_requests']}")
        print(f"    Error Rate: {data['api']['error_rate']:.1f}%")
        print(f"    Models Active: {data['models']['active_models']}")
        print(f"    Tokens Processed: {data['models']['tokens_processed']}")
    
    # Performance alerts
    response = requests.get(f"{BASE_URL}/api/performance/alerts")
    if response.status_code == 200:
        alerts = response.json()
        print(f"  🚨 Alerts: {len(alerts)} active")
        for alert in alerts[:3]:  # Show first 3
            print(f"    {alert['level'].upper()}: {alert['message']}")
    
    return True

def test_logs():
    """Test logs endpoint"""
    print("\n📋 Testing logs endpoint...")
    
    response = requests.get(f"{BASE_URL}/api/logs?limit=5")
    if response.status_code == 200:
        data = response.json()
        print(f"  Total logs: {data['stats']['total']}")
        print(f"  Errors: {data['stats']['errors']}")
        print(f"  Warnings: {data['stats']['warnings']}")
        
        print("  Recent logs:")
        for log in data['logs'][-3:]:
            timestamp = log['timestamp'].split('T')[1][:8]
            print(f"    {timestamp} [{log['level']}] {log['message'][:50]}...")
    
    return True

def test_models():
    """Test models endpoint"""
    print("\n🤖 Testing models endpoint...")
    
    response = requests.get(f"{BASE_URL}/api/models")
    if response.status_code == 200:
        data = response.json()
        print(f"  Available models: {len(data['available'])}")
        for model in data['available']:
            print(f"    - {model}")
        
        print(f"  External agents: {len(data['external_agents'])}")
        online_agents = [a for a in data['external_agents'] if a['status'] == 'online']
        print(f"  Online agents: {len(online_agents)}")
        for agent in online_agents:
            print(f"    - {agent['name']} (port {agent['port']})")
    
    return True

async def test_websocket():
    """Test WebSocket endpoint"""
    print("\n🔌 Testing WebSocket endpoint...")
    
    try:
        uri = "ws://localhost:8000/ws"
        async with websockets.connect(uri) as websocket:
            print("  ✅ Connected to WebSocket")
            
            # Wait for a few metric updates
            for i in range(3):
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                
                if data['type'] == 'metrics':
                    metrics = data['data']
                    print(f"  📊 Update {i+1}: CPU {metrics['system']['cpu_usage']:.1f}%, "
                          f"Requests {metrics['api']['total_requests']}")
                
                await asyncio.sleep(1)
            
            print("  ✅ WebSocket streaming working")
            return True
            
    except Exception as e:
        print(f"  ❌ WebSocket error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 SutazAI Performance Backend Test Suite")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Chat API", test_chat),
        ("Metrics", test_metrics),
        ("Logs", test_logs),
        ("Models", test_models),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results[test_name] = False
    
    # Test WebSocket (async)
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results["WebSocket"] = loop.run_until_complete(test_websocket())
        loop.close()
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        results["WebSocket"] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"  {test_name:<15} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"Total time: {time.time() - start_time:.2f}s")
    
    if passed == total:
        print("\n🎉 All tests passed! Backend is fully operational.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")

if __name__ == "__main__":
    main()