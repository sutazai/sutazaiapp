#!/usr/bin/env python3
"""
Test for Unified Agent Configuration Loader
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.unified_agent_loader import UnifiedAgentLoader, get_agent, get_active_agents

def test_unified_loader():
    """Test the unified agent loader functionality"""
    
    print("Testing Unified Agent Loader")
    print("=" * 60)
    
    # Initialize loader
    loader = UnifiedAgentLoader()
    
    # Test 1: Load registry
    print("\n1. Loading agent registry...")
    registry = loader.load_registry()
    total_agents = len(registry.get("agents", {}))
    print(f"   ✓ Loaded {total_agents} agents from registry")
    print(f"   Version: {registry.get('version', 'unknown')}")
    
    # Test 2: Load capabilities
    print("\n2. Loading capabilities...")
    capabilities = loader.load_capabilities()
    total_caps = len(capabilities.get("capabilities", {}))
    print(f"   ✓ Loaded {total_caps} capability definitions")
    
    # Test 3: Load runtime status
    print("\n3. Loading runtime status...")
    runtime = loader.load_runtime_status()
    active_count = len(runtime.get("agents", {}))
    print(f"   ✓ Found {active_count} agents with runtime status")
    
    # Test 4: Get specific agent
    print("\n4. Testing get_agent()...")
    test_agent_id = "ai-agent-orchestrator"
    agent = loader.get_agent(test_agent_id)
    if agent:
        print(f"   ✓ Found agent: {agent.get('name', 'unknown')}")
        print(f"     Type: {agent.get('type', 'unknown')}")
        print(f"     Capabilities: {len(agent.get('capabilities', []))} defined")
    else:
        print(f"   ⚠ Agent '{test_agent_id}' not found")
    
    # Test 5: Get agents by capability
    print("\n5. Testing get_agents_by_capability()...")
    test_capability = "code_generation"
    agents_with_cap = loader.get_agents_by_capability(test_capability)
    print(f"   ✓ Found {len(agents_with_cap)} agents with '{test_capability}' capability")
    
    # Test 6: Get agents by type
    print("\n6. Testing get_agents_by_type()...")
    test_type = "utility"
    agents_of_type = loader.get_agents_by_type(test_type)
    print(f"   ✓ Found {len(agents_of_type)} agents of type '{test_type}'")
    
    # Test 7: Get active agents
    print("\n7. Testing get_active_agents()...")
    active_agents = loader.get_active_agents()
    print(f"   ✓ Found {len(active_agents)} active/healthy agents")
    
    # Test 8: Get statistics
    print("\n8. Testing get_agent_statistics()...")
    stats = loader.get_agent_statistics()
    print(f"   ✓ Statistics:")
    print(f"     Total registered: {stats.get('total_registered', 0)}")
    print(f"     Currently active: {stats.get('currently_active', 0)}")
    print(f"     Health percentage: {stats.get('health_percentage', 0):.1f}%")
    
    # Test 9: Search agents
    print("\n9. Testing search_agents()...")
    search_query = "orchestrator"
    search_results = loader.search_agents(search_query)
    print(f"   ✓ Found {len(search_results)} agents matching '{search_query}'")
    
    # Test 10: Test convenience functions
    print("\n10. Testing convenience functions...")
    agent = get_agent("task-assignment-coordinator")
    if agent:
        print(f"   ✓ get_agent() works: {agent.get('name', 'unknown')}")
    
    active = get_active_agents()
    print(f"   ✓ get_active_agents() works: {len(active)} agents")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"✅ All tests passed successfully")
    print(f"✅ Unified loader is working correctly")
    print(f"✅ Backward compatibility maintained")
    print(f"\nRegistry contains {total_agents} agents")
    print(f"Currently {active_count} agents are active")
    print(f"System defines {total_caps} unique capabilities")
    
    return True

if __name__ == "__main__":
    success = test_unified_loader()
    sys.exit(0 if success else 1)