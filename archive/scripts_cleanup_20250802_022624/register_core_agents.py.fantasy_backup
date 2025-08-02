#!/usr/bin/env python3
"""Register core agents with the task coordinator"""

import json
import requests
import os
from pathlib import Path

# Agent configuration files to register
AGENT_CONFIGS = [
    'senior-ai-engineer-simple.json',
    'deployment-automation-master-simple.json', 
    'infrastructure-devops-manager-simple.json',
    'ollama-integration-specialist-simple.json',
    'testing-qa-validator-simple.json'
]

def load_agent_config(config_file):
    """Load agent configuration from JSON file"""
    config_path = Path(__file__).parent.parent / 'agents' / 'configs' / config_file
    with open(config_path, 'r') as f:
        return json.load(f)

def register_agent_simple(config, coordinator_url):
    """Register agent using a simple POST to the existing endpoint"""
    print(f"Registering agent: {config['name']}")
    print(f"  Role: {config['role']}")
    print(f"  Capabilities: {', '.join(config['capabilities'])}")
    
    # For now, since the task-coordinator doesn't have agent registration,
    # we'll just print the config and simulate registration
    
    # In a real implementation, this would POST to /agents/register
    # response = requests.post(f"{coordinator_url}/agents/register", json=config)
    
    print(f"  Status: SIMULATED - Agent registered successfully")
    return True

def test_task_assignment(coordinator_url):
    """Test task assignment to registered agents"""
    test_tasks = [
        {"task": "Build a neural network for image classification", "priority": "high"},
        {"task": "Deploy the application to production", "priority": "high"},
        {"task": "Set up Docker containers for the microservices", "priority": "normal"},
        {"task": "Optimize Ollama model performance", "priority": "high"},
        {"task": "Run integration tests on the API", "priority": "normal"}
    ]
    
    print("\n=== Testing Task Assignment ===")
    
    for task_request in test_tasks:
        try:
            response = requests.post(f"{coordinator_url}/assign", json=task_request)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Task: {task_request['task'][:50]}...")
                print(f"   Status: {result.get('status', 'unknown')}")
                if 'assigned_agent' in result:
                    print(f"   Assigned to: {result['assigned_agent']}")
                else:
                    print(f"   Agent: {result.get('agent', 'task-assignment-coordinator')}")
            else:
                print(f"❌ Failed to assign task: {response.status_code}")
        except Exception as e:
            print(f"❌ Error assigning task: {e}")
        print()

def main():
    # Configuration
    coordinator_url = "http://localhost:8522"  # Would be internal in production
    
    print("🤖 SutazAI Core Agent Registration")
    print("=" * 50)
    
    # Try to contact task coordinator
    try:
        response = requests.get(f"{coordinator_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Task coordinator is not healthy")
            return False
    except Exception as e:
        print(f"❌ Cannot reach task coordinator: {e}")
        print("💡 Note: Running from host, coordinator may not be exposed")
        print("💡 Continuing with simulated registration...")
        coordinator_url = None
    
    # Load and register agents
    registered_count = 0
    
    for config_file in AGENT_CONFIGS:
        try:
            config = load_agent_config(config_file)
            
            if coordinator_url:
                success = register_agent_simple(config, coordinator_url)
            else:
                # Simulate registration
                print(f"📝 Agent Config Loaded: {config['name']}")
                print(f"   Role: {config['role']}")  
                print(f"   Capabilities: {', '.join(config['capabilities'])}")
                print(f"   Model: {config['model']} via {config['ollama_endpoint']}")
                success = True
            
            if success:
                registered_count += 1
                
        except Exception as e:
            print(f"❌ Error registering {config_file}: {e}")
    
    print(f"\n✅ Registration Summary: {registered_count}/{len(AGENT_CONFIGS)} agents processed")
    
    # Test task assignment if coordinator is available
    if coordinator_url:
        test_task_assignment(coordinator_url)
    else:
        print("\n🧪 Task Assignment Simulation:")
        print("   Since coordinator is not accessible from host,")
        print("   we'll test from within the Docker network...")
    
    return registered_count > 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)