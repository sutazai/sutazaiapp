#!/usr/bin/env python3
"""
Quick Test Script for Autonomous Agents Demo
============================================

This script validates that the demo components work correctly
before running the full demonstration.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append('/opt/sutazaiapp')

def test_imports():
    """Test that all required packages can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import redis
        print("  ✓ redis")
    except ImportError:
        print("  ✗ redis - Run: pip install redis")
        return False
    
    try:
        import httpx
        print("  ✓ httpx")
    except ImportError:
        print("  ✗ httpx - Run: pip install httpx")
        return False
    
    try:
        from rich.console import Console
        from rich.table import Table
        print("  ✓ rich")
    except ImportError:
        print("  ✗ rich - Run: pip install rich")
        return False
    
    return True

def test_redis_connection():
    """Test Redis connection"""
    print("\n🔴 Testing Redis connection...")
    
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, 
                           password='redis_password', decode_responses=False)
        
        # Test basic operations
        client.ping()
        client.set('test_key', 'test_value')
        value = client.get('test_key')
        client.delete('test_key')
        
        print("  ✓ Redis connection successful")
        return True
        
    except redis.ConnectionError:
        print("  ✗ Redis connection failed - Is Redis running?")
        print("    Run: docker-compose up -d redis")
        return False
    except Exception as e:
        print(f"  ✗ Redis error: {e}")
        return False

def test_ollama_connection():
    """Test Ollama connection"""
    print("\n🤖 Testing Ollama connection...")
    
    try:
        import httpx
        
        # Test if Ollama is accessible
        with httpx.Client(timeout=10.0) as client:
            response = client.get("http://localhost:11434/api/tags")
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"  ✓ Ollama connection successful")
                print(f"  ✓ Available models: {len(models)}")
                
                # Check for required models
                model_names = [model['name'] for model in models]
                if any('deepseek' in name or 'llama' in name for name in model_names):
                    print("  ✓ Compatible models found")
                    return True
                else:
                    print("  ⚠ No compatible models found")
                    print("    Run: docker exec sutazai-ollama ollama pull deepseek-r1:8b")
                    return False
            else:
                print(f"  ✗ Ollama API error: {response.status_code}")
                return False
                
    except httpx.ConnectError:
        print("  ✗ Ollama connection failed - Is Ollama running?")
        print("    Run: docker-compose up -d ollama")
        return False
    except Exception as e:
        print(f"  ✗ Ollama error: {e}")
        return False

def test_demo_components():
    """Test demo components"""
    print("\n📁 Testing demo components...")
    
    demo_dir = Path('/opt/sutazaiapp/demos')
    
    components = [
        'autonomous_agents_demo.py',
        'run_demo.sh',
        'README.md'
    ]
    
    all_good = True
    for component in components:
        path = demo_dir / component
        if path.exists():
            print(f"  ✓ {component}")
        else:
            print(f"  ✗ {component} missing")
            all_good = False
    
    return all_good

def test_agent_creation():
    """Test basic agent creation"""
    print("\n👥 Testing agent creation...")
    
    try:
        # Import demo components
        from demos.autonomous_agents_demo import BaseAgent, CodeAnalyzerAgent
        import redis
        
        redis_client = redis.Redis(host='localhost', port=6379, 
                                 password='redis_password', decode_responses=False)
        
        # Create test agent
        test_agent = CodeAnalyzerAgent("test_analyzer", redis_client)
        
        # Verify agent properties
        assert test_agent.agent_id == "test_analyzer"
        assert "code_analysis" in test_agent.capabilities
        assert test_agent.status == "idle"
        
        # Cleanup
        redis_client.srem("active_agents", "test_analyzer")
        redis_client.delete(f"agent:test_analyzer")
        redis_client.delete(f"agent_queue:test_analyzer")
        
        print("  ✓ Agent creation successful")
        return True
        
    except Exception as e:
        print(f"  ✗ Agent creation failed: {e}")
        return False

def run_quick_demo():
    """Run a quick demo to verify functionality"""
    print("\n🚀 Running quick demo test...")
    
    try:
        from demos.autonomous_agents_demo import DemoManager
        import asyncio
        
        async def quick_test():
            # Create minimal demo
            demo = DemoManager(num_analyzer_agents=1, num_improver_agents=1)
            
            # Start agents briefly
            agent_tasks = await demo.start_agents()
            
            # Wait a moment
            await asyncio.sleep(2)
            
            # Check agent registration
            active_agents = demo.redis_client.smembers("active_agents")
            if len(active_agents) >= 2:
                print("  ✓ Agents started successfully")
                result = True
            else:
                print(f"  ⚠ Only {len(active_agents)} agents registered")
                result = False
            
            # Cleanup
            for task in agent_tasks:
                task.cancel()
            
            # Clean Redis
            for agent_id in active_agents:
                demo.redis_client.srem("active_agents", agent_id)
                demo.redis_client.delete(f"agent:{agent_id.decode()}")
                demo.redis_client.delete(f"agent_queue:{agent_id.decode()}")
            
            return result
        
        return asyncio.run(quick_test())
        
    except Exception as e:
        print(f"  ✗ Quick demo failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔬 SutazAI Autonomous Agents Demo - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Redis Connection", test_redis_connection),
        ("Ollama Connection", test_ollama_connection),
        ("Demo Components", test_demo_components),
        ("Agent Creation", test_agent_creation),
        ("Quick Demo", run_quick_demo)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
    
    print(f"\n{'=' * 50}")
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Demo is ready to run.")
        print("\nTo run the demo:")
        print("  ./demos/run_demo.sh")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues before running the demo.")
        return 1

if __name__ == "__main__":
    sys.exit(main())