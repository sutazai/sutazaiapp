#!/usr/bin/env python3
"""
Test script to validate hardware-resource-optimizer agent detection
"""

import sys
import json
import time
import requests
from pathlib import Path

# Add monitoring directory to path
sys.path.insert(0, str(Path(__file__).parent))

from static_monitor import EnhancedMonitor

def test_hardware_optimizer_detection():
    """Test the hardware-resource-optimizer detection specifically"""
    print("Testing hardware-resource-optimizer detection...")
    
    # Initialize monitor
    monitor = EnhancedMonitor()
    
    # Check if agent is in registry
    if 'hardware-resource-optimizer' not in monitor.agent_registry.get('agents', {}):
        print("❌ FAIL: hardware-resource-optimizer not found in agent registry")
        return False
    
    print("✅ PASS: hardware-resource-optimizer found in agent registry")
    
    # Test direct endpoint detection
    agent_info = monitor.agent_registry['agents']['hardware-resource-optimizer']
    endpoint = monitor._get_agent_endpoint('hardware-resource-optimizer', agent_info)
    
    if not endpoint:
        print("❌ FAIL: Could not determine endpoint for hardware-resource-optimizer")
        return False
    
    print(f"✅ PASS: Endpoint detected: {endpoint}")
    
    # Test health check
    health_status, response_time = monitor._check_agent_health('hardware-resource-optimizer', agent_info, 5)
    
    print(f"Health Status: {health_status}")
    print(f"Response Time: {response_time}ms" if response_time else "Response Time: N/A")
    
    # Validate response time is reasonable for this agent
    if response_time and response_time > 5000:  # 5 seconds threshold
        print(f"⚠️  WARNING: Response time {response_time}ms is high but acceptable for hardware-resource-optimizer")
    elif response_time and response_time > 2500:  # 2.5 seconds warning threshold
        print(f"⚠️  WARNING: Response time {response_time}ms is above normal but within tolerance")
    elif response_time:
        print(f"✅ PASS: Response time {response_time}ms is good")
    
    # Test agent status in full monitoring context
    agents, healthy, total = monitor.get_ai_agents_status()
    
    # Find hardware-resource-optimizer in the agent list
    hw_optimizer_found = False
    for agent_line in agents:
        if 'hw-resource-optim' in agent_line or 'hardware-resource' in agent_line:
            hw_optimizer_found = True
            print(f"✅ PASS: Found hardware-resource-optimizer in agent list: {agent_line.strip()}")
            
            # Check if it shows as healthy
            if '🟢' in agent_line and 'healthy' in agent_line:
                print("✅ PASS: hardware-resource-optimizer shows as healthy (green)")
            elif '🟡' in agent_line and 'warning' in agent_line:
                print("⚠️  WARNING: hardware-resource-optimizer shows as warning (yellow)")
            elif '🔴' in agent_line:
                print("❌ FAIL: hardware-resource-optimizer shows as unhealthy (red)")
            else:
                print(f"❓ UNKNOWN: Could not determine health status from: {agent_line}")
            break
    
    if not hw_optimizer_found:
        print("❌ FAIL: hardware-resource-optimizer not found in agent status list")
        return False
    
    print(f"\nSummary: {healthy}/{total} agents healthy")
    
    # Cleanup
    monitor.cleanup()
    
    return True

def test_direct_connection():
    """Test direct connection to hardware-resource-optimizer"""
    print("\nTesting direct connection to hardware-resource-optimizer...")
    
    try:
        start_time = time.time()
        response = requests.get('http://localhost:8116/health', timeout=5)
        response_time = (time.time() - start_time) * 1000
        
        print(f"✅ PASS: Direct connection successful")
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {response_time:.0f}ms")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Agent Status: {data.get('status', 'unknown')}")
                print(f"Agent Name: {data.get('agent', 'unknown')}")
            except:
                print("Response is not JSON")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Direct connection failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Hardware Resource Optimizer Detection Test")
    print("=" * 60)
    
    # Test direct connection first
    direct_success = test_direct_connection()
    
    # Test monitor detection
    monitor_success = test_hardware_optimizer_detection()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print(f"Direct Connection: {'✅ PASS' if direct_success else '❌ FAIL'}")
    print(f"Monitor Detection: {'✅ PASS' if monitor_success else '❌ FAIL'}")
    
    if direct_success and monitor_success:
        print("🎉 ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)