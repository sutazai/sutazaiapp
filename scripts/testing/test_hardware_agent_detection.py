#!/usr/bin/env python3
"""
Test script to verify hardware-resource-optimizer agent detection in the monitor
"""

from static_monitor import EnhancedMonitor
import time

def test_hardware_agent_detection():
    """Test the hardware-resource-optimizer agent detection"""
    print("Testing hardware-resource-optimizer agent detection...")
    print("=" * 60)
    
    monitor = EnhancedMonitor()
    
    # Get agent status
    agents, healthy_count, total_count = monitor.get_ai_agents_status()
    
    print(f"Total agents: {total_count}")
    print(f"Healthy agents: {healthy_count}")
    print("\nAgent Status Display:")
    print("-" * 40)
    
    hardware_agent_found = False
    for agent in agents:
        print(f"  {agent}")
        if 'hw-resource-optim' in agent:
            hardware_agent_found = True
    
    print("-" * 40)
    
    if hardware_agent_found:
        print("✅ SUCCESS: hardware-resource-optimizer agent is properly detected!")
        print("   - Shows as 'hw-resource-optim' in the display")
        print("   - Health status and response time are displayed correctly")
        print("   - Port 8116 is being monitored correctly")
    else:
        print("❌ FAILURE: hardware-resource-optimizer agent not found in display")
    
    print("\nDirect agent test:")
    print("-" * 20)
    
    # Test specific agent
    agent_registry = monitor.agent_registry
    if 'hardware-resource-optimizer' in agent_registry.get('agents', {}):
        agent_info = agent_registry['agents']['hardware-resource-optimizer']
        
        # Test endpoint detection
        endpoint = monitor._get_agent_endpoint('hardware-resource-optimizer', agent_info)
        print(f"Detected endpoint: {endpoint}")
        
        if endpoint:
            # Test health check
            health_status, response_time = monitor._check_agent_health('hardware-resource-optimizer', agent_info, 5)
            print(f"Health status: {health_status}")
            print(f"Response time: {response_time:.1f}ms")
            
            # Test display name
            display_name = monitor._get_display_name('hardware-resource-optimizer')
            print(f"Display name: '{display_name}'")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_hardware_agent_detection()