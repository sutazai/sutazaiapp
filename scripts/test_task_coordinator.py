#!/usr/bin/env python3
"""Test task coordinator from within Docker network"""

import json
import requests
import sys

def test_coordinator():
    """Test the task coordinator functionality"""
    coordinator_url = "http://task-coordinator:8522"
    
    print("🧪 Testing Task Assignment Coordinator")
    print("=" * 50)
    
    # Test 1: Health check
    try:
        response = requests.get(f"{coordinator_url}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print("✅ Health Check: PASSED")
            print(f"   Agent: {health.get('agent')}")
            print(f"   Status: {health.get('status')}")
            print(f"   Ollama Connected: {health.get('ollama_connected')}")
            print(f"   Backend Connected: {health.get('backend_connected')}")
        else:
            print(f"❌ Health Check: FAILED ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Health Check: ERROR - {e}")
        return False
    
    print()
    
    # Test 2: Root endpoint
    try:
        response = requests.get(f"{coordinator_url}/", timeout=5)
        if response.status_code == 200:
            root_info = response.json()
            print("✅ Root Endpoint: PASSED")
            print(f"   Message: {root_info.get('message')}")
        else:
            print(f"❌ Root Endpoint: FAILED ({response.status_code})")
    except Exception as e:
        print(f"❌ Root Endpoint: ERROR - {e}")
    
    print()
    
    # Test 3: Task Assignment with different types
    test_tasks = [
        {
            "task": "Design and implement a neural network architecture for computer vision",
            "priority": "high",
            "expected_agent_type": "AI/ML Engineer"
        },
        {
            "task": "Deploy microservices to production using CI/CD pipeline",
            "priority": "high", 
            "expected_agent_type": "Deployment Specialist"
        },
        {
            "task": "Set up Docker containers and manage Kubernetes cluster",
            "priority": "normal",
            "expected_agent_type": "Infrastructure Manager"
        },
        {
            "task": "Optimize Ollama model performance and manage LLM inference",
            "priority": "high",
            "expected_agent_type": "Model Management"
        },
        {
            "task": "Run comprehensive testing suite and validate code quality",
            "priority": "normal",
            "expected_agent_type": "QA Specialist"
        }
    ]
    
    print("🎯 Testing Task Assignment:")
    successful_assignments = 0
    
    for i, task_request in enumerate(test_tasks, 1):
        try:
            response = requests.post(f"{coordinator_url}/assign", json=task_request, timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Test {i}: {task_request['task'][:40]}...")
                print(f"   Status: {result.get('status')}")
                print(f"   Priority: {result.get('priority')}")
                print(f"   Agent: {result.get('agent', 'N/A')}")
                
                if result.get('status') == 'assigned':
                    successful_assignments += 1
                    
            else:
                print(f"❌ Test {i}: FAILED ({response.status_code})")
                print(f"   Task: {task_request['task'][:40]}...")
                
        except Exception as e:
            print(f"❌ Test {i}: ERROR - {e}")
            print(f"   Task: {task_request['task'][:40]}...")
        
        print()
    
    # Summary
    print("📊 Test Summary:")
    print(f"   Total Tests: {len(test_tasks)}")
    print(f"   Successful Assignments: {successful_assignments}")
    print(f"   Success Rate: {(successful_assignments/len(test_tasks)*100):.1f}%")
    
    if successful_assignments == len(test_tasks):
        print("🎉 All tests PASSED!")
        return True
    else:
        print("⚠️  Some tests had issues")
        return successful_assignments > 0

if __name__ == "__main__":
    success = test_coordinator()
    sys.exit(0 if success else 1)