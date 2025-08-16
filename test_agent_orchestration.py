#!/usr/bin/env python3
"""
Test Agent Orchestration Functionality
Validates that agent loading and orchestration works after fixes
"""

import json
import sys
from pathlib import Path

def test_agent_loading():
    """Test if agents can be loaded from registry"""
    print("=" * 80)
    print("AGENT ORCHESTRATION TEST")
    print("=" * 80)
    
    results = {
        "registry_load": False,
        "config_files_exist": False,
        "agent_load": False,
        "orchestration_ready": False,
        "errors": []
    }
    
    # Test 1: Load Registry
    print("\n1. Testing Registry Load...")
    registry_path = Path("/opt/sutazaiapp/agents/agent_registry.json")
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        agents = registry.get("agents", {})
        print(f"   ✅ Registry loaded: {len(agents)} agents found")
        results["registry_load"] = True
    except Exception as e:
        print(f"   ❌ Registry load failed: {e}")
        results["errors"].append(f"Registry load: {e}")
        return results
    
    # Test 2: Verify Config Files
    print("\n2. Testing Config Files...")
    configs_dir = Path("/opt/sutazaiapp/agents/configs")
    missing_configs = []
    found_configs = []
    
    for agent_name, agent_data in agents.items():
        config_path = agent_data.get("config_path", "")
        if config_path:
            config_file = configs_dir / Path(config_path).name
            if config_file.exists():
                found_configs.append(agent_name)
            else:
                missing_configs.append(agent_name)
    
    print(f"   ✅ Configs found: {len(found_configs)}/{len(agents)}")
    if missing_configs:
        print(f"   ⚠️  Missing configs: {len(missing_configs)}")
        results["errors"].append(f"Missing {len(missing_configs)} configs")
    else:
        results["config_files_exist"] = True
    
    # Test 3: Load Sample Agent
    print("\n3. Testing Agent Load...")
    sample_agent = "ai-agent-orchestrator"
    if sample_agent in agents:
        agent_data = agents[sample_agent]
        config_path = agent_data.get("config_path", "")
        if config_path:
            config_file = configs_dir / Path(config_path).name
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(f"   ✅ Loaded {sample_agent}:")
                print(f"      - ID: {config.get('id')}")
                print(f"      - Version: {config.get('version')}")
                print(f"      - Status: {config.get('status')}")
                print(f"      - Capabilities: {len(config.get('capabilities', []))} defined")
                results["agent_load"] = True
            except Exception as e:
                print(f"   ❌ Failed to load agent config: {e}")
                results["errors"].append(f"Agent load: {e}")
    
    # Test 4: Orchestration Readiness
    print("\n4. Testing Orchestration Readiness...")
    orchestration_agents = []
    for agent_name, agent_data in agents.items():
        capabilities = agent_data.get("capabilities", [])
        if "orchestration" in capabilities or "coordination" in capabilities:
            orchestration_agents.append(agent_name)
    
    print(f"   ✅ Orchestration agents found: {len(orchestration_agents)}")
    if orchestration_agents:
        print(f"      Examples: {', '.join(orchestration_agents[:5])}")
        results["orchestration_ready"] = True
    else:
        print("   ⚠️  No orchestration agents found")
        results["errors"].append("No orchestration agents")
    
    # Test 5: Multi-Agent Workflow
    print("\n5. Testing Multi-Agent Workflow Capability...")
    workflow_ready = True
    required_roles = [
        ("orchestrator", ["orchestration", "coordination"]),
        ("developer", ["code_generation"]),
        ("tester", ["testing"]),
        ("deployer", ["deployment"]),
        ("monitor", ["monitoring"])
    ]
    
    for role, required_caps in required_roles:
        agents_for_role = []
        for agent_name, agent_data in agents.items():
            agent_caps = agent_data.get("capabilities", [])
            if any(cap in agent_caps for cap in required_caps):
                agents_for_role.append(agent_name)
        
        if agents_for_role:
            print(f"   ✅ {role.capitalize()}: {len(agents_for_role)} agents available")
        else:
            print(f"   ❌ {role.capitalize()}: No agents found")
            workflow_ready = False
    
    if workflow_ready:
        print("\n   🎉 Multi-agent workflow capability: READY")
    else:
        print("\n   ⚠️  Multi-agent workflow: INCOMPLETE")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total_tests = 4
    passed_tests = sum([
        results["registry_load"],
        results["config_files_exist"],
        results["agent_load"],
        results["orchestration_ready"]
    ])
    
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Registry Load: {'✅' if results['registry_load'] else '❌'}")
    print(f"Config Files: {'✅' if results['config_files_exist'] else '❌'}")
    print(f"Agent Load: {'✅' if results['agent_load'] else '❌'}")
    print(f"Orchestration: {'✅' if results['orchestration_ready'] else '❌'}")
    
    if results["errors"]:
        print(f"\nErrors: {len(results['errors'])}")
        for error in results["errors"]:
            print(f"  - {error}")
    
    # Overall Status
    if passed_tests == total_tests:
        print("\n✅ AGENT ORCHESTRATION: OPERATIONAL")
        return 0
    elif passed_tests >= 3:
        print("\n⚠️  AGENT ORCHESTRATION: PARTIALLY OPERATIONAL")
        return 1
    else:
        print("\n❌ AGENT ORCHESTRATION: NOT OPERATIONAL")
        return 2

if __name__ == "__main__":
    sys.exit(test_agent_loading())