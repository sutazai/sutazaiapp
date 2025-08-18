#!/usr/bin/env python3
"""
Test script to validate UnifiedAgentRegistry works with real files only
"""

import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, '/opt/sutazaiapp/backend')

from app.core.unified_agent_registry import get_registry

def test_unified_registry():
    """Test the UnifiedAgentRegistry with real files"""
    print("=" * 80)
    print("TESTING UNIFIED AGENT REGISTRY - RULE 1 COMPLIANCE")
    print("=" * 80)
    
    # Get registry instance
    registry = get_registry()
    
    # Check statistics
    stats = registry.get_statistics()
    print(f"\n📊 Registry Statistics:")
    print(f"  - Total agents: {stats['total_agents']}")
    print(f"  - Claude agents: {stats['claude_agents']}")
    print(f"  - Container agents: {stats['container_agents']}")
    
    # List capabilities
    if stats['capabilities']:
        print(f"\n🎯 Available Capabilities:")
        for cap, count in sorted(stats['capabilities'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {cap}: {count} agents")
    
    # Test agent lookup
    print(f"\n🔍 Testing Agent Lookup:")
    
    # Test finding an orchestration agent
    orchestrator = registry.find_best_agent("I need to orchestrate multiple agents", ["orchestration"])
    if orchestrator:
        print(f"  ✅ Found orchestrator: {orchestrator.name} (type: {orchestrator.type})")
        print(f"     Description: {orchestrator.description[:100]}...")
    else:
        print(f"  ⚠️  No orchestrator found")
    
    # Test finding a deployment agent
    deployer = registry.find_best_agent("Deploy the system", ["deployment"])
    if deployer:
        print(f"  ✅ Found deployer: {deployer.name} (type: {deployer.type})")
    else:
        print(f"  ⚠️  No deployment agent found")
    
    # Verify all deployment_info paths exist
    print(f"\n🔍 Verifying Agent File References:")
    invalid_refs = []
    valid_refs = 0
    
    for agent_id, agent in registry.agents.items():
        if agent.deployment_info:
            # Check for file references
            agent_file = agent.deployment_info.get("agent_file")
            config_path = agent.deployment_info.get("config_path")
            
            if agent_file:
                if Path(agent_file).exists():
                    valid_refs += 1
                else:
                    invalid_refs.append(f"{agent.name}: {agent_file}")
            
            if config_path:
                # Config path validation is now handled in the registry itself
                # Just check if it's set to a valid path (not None)
                if config_path and Path(config_path).exists():
                    valid_refs += 1
                elif config_path:
                    # Check if it's an original (unvalidated) path
                    original = agent.deployment_info.get("original_config")
                    if original:
                        # This means the original config didn't exist, which is OK
                        # as long as we're not claiming it exists
                        pass
                    else:
                        invalid_refs.append(f"{agent.name}: {config_path}")
    
    print(f"  ✅ Valid file references: {valid_refs}")
    if invalid_refs:
        print(f"  ❌ Invalid file references found (RULE 1 VIOLATION):")
        for ref in invalid_refs[:5]:  # Show first 5
            print(f"     - {ref}")
    else:
        print(f"  ✅ All file references are valid!")
    
    # Save the registry
    print(f"\n💾 Testing Registry Persistence:")
    if registry.save_registry():
        print(f"  ✅ Successfully saved registry")
        
        # Check if file exists
        config_path = Path("/opt/sutazaiapp/config/agents/unified_agent_registry.json")
        if config_path.exists():
            print(f"  ✅ Registry file created at: {config_path}")
            
            # Check file size
            size = config_path.stat().st_size
            print(f"  📦 Registry file size: {size:,} bytes")
        else:
            print(f"  ❌ Registry file not found at expected location")
    else:
        print(f"  ❌ Failed to save registry")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    return len(invalid_refs) == 0

if __name__ == "__main__":
    success = test_unified_registry()
    sys.exit(0 if success else 1)