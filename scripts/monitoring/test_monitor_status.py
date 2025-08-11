#!/usr/bin/env python3
"""Test monitor agent status display"""

import sys
sys.path.append('/opt/sutazaiapp/scripts/monitoring')

from static_monitor import EnhancedMonitor

def test_monitor():
    """Test monitor agent status"""
    print("=== Testing Monitor Agent Status ===\n")
    
    try:
        monitor = EnhancedMonitor()
        agents, healthy_count, total_count = monitor.get_ai_agents_status()
        
        print(f"Total agents: {total_count}")
        print(f"Healthy agents: {healthy_count}")
        print()
        
        print("Agent Status Display:")
        for i, agent_display in enumerate(agents):
            print(f"  {i+1}. {agent_display}")
        
        print()
        print("=== Testing individual container detection ===")
        test_agents = [
            'document-knowledge-manager',
            'ollama-integration-specialist', 
            'code-generation-improver',
            'semgrep-security-analyzer',
            'senior-ai-engineer',
            'hardware-resource-optimizer'
        ]
        
        for agent_id in test_agents:
            is_deployed = monitor._is_agent_deployed(agent_id)
            container_info = monitor._get_container_info(agent_id)
            
            print(f"{agent_id}:")
            print(f"  Deployed: {is_deployed}")
            if container_info:
                print(f"  Container: {container_info['name']} ({container_info['status']})")
            else:
                print(f"  Container: None")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_monitor()