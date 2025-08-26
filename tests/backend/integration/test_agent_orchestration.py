#!/usr/bin/env python3
"""
Test script for the unified agent orchestration system
Verifies that Claude agents can be discovered, selected, and executed
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import json
from app.core.unified_agent_registry import get_registry
from app.core.claude_agent_executor import get_executor, get_pool
from app.core.claude_agent_selector import get_selector

def test_registry():
    """Test the unified agent registry"""
    print("\n=== Testing Unified Agent Registry ===")
    
    registry = get_registry()
    stats = registry.get_statistics()
    
    print(f"Total agents loaded: {stats['total_agents']}")
    print(f"Claude agents: {stats['claude_agents']}")
    print(f"Container agents: {stats['container_agents']}")
    print(f"Capabilities found: {len(stats['capabilities'])}")
    
    # List some Claude agents
    claude_agents = registry.list_agents(agent_type="claude")[:5]
    print(f"\nSample Claude agents:")
    for agent in claude_agents:
        print(f"  - {agent.name}: {agent.capabilities[:3]}")
    
    return stats['total_agents'] > 0

def test_selector():
    """Test the intelligent agent selector"""
    print("\n=== Testing Claude Agent Selector ===")
    
    selector = get_selector()
    registry = get_registry()
    
    test_tasks = [
        "I need to orchestrate multiple AI agents for a complex workflow",
        "Help me write Python code for a REST API",
        "Test my application for security vulnerabilities",
        "Deploy my application to production",
        "Optimize the performance of my database queries"
    ]
    
    for task in test_tasks:
        print(f"\nTask: {task[:60]}...")
        recommendations = selector.get_agent_recommendations(task)
        
        if recommendations['recommended_agent']:
            agent = recommendations['recommended_agent']
            print(f"  Recommended: {agent['name']} ({agent['type']})")
            print(f"  Confidence: {recommendations['task_analysis']['confidence']:.2f}")
            print(f"  Domain: {recommendations['task_analysis']['primary_domain']}")
        else:
            print("  No agent recommended")
    
    return True

async def test_executor():
    """Test the Claude agent executor"""
    print("\n=== Testing Claude Agent Executor ===")
    
    executor = get_executor()
    registry = get_registry()
    
    # Find a test agent
    test_agent = registry.find_best_agent("Help me solve a complex problem")
    
    if test_agent and test_agent.type == "claude":
        print(f"Testing execution with agent: {test_agent.name}")
        
        result = await executor.execute_agent(
            test_agent.name,
            "Analyze this test task and provide recommendations",
            {"test": True, "context": "orchestration test"}
        )
        
        print(f"  Task ID: {result['task_id']}")
        print(f"  Status: {result['status']}")
        print(f"  Agent: {result['agent']}")
        
        return result['status'] == 'success'
    else:
        print("No Claude agent available for testing")
        return False

async def test_pool():
    """Test the agent pool for async execution"""
    print("\n=== Testing Claude Agent Pool ===")
    
    pool = get_pool()
    registry = get_registry()
    
    # Submit multiple tasks
    task_ids = []
    for i in range(3):
        agent = registry.find_best_agent(f"Task {i}: Process some data")
        if agent and agent.type == "claude":
            task_id = await pool.submit_task(
                agent.name,
                f"Process task number {i}",
                {"task_number": i}
            )
            task_ids.append(task_id)
            print(f"Submitted task {i} with ID: {task_id}")
    
    # Wait a bit for processing
    await asyncio.sleep(2)
    
    # Check results
    for task_id in task_ids:
        result = pool.get_result(task_id)
        if result:
            print(f"  Task {task_id}: {result['status']}")
    
    return len(task_ids) > 0

def test_api_integration():
    """Test that API can import our modules"""
    print("\n=== Testing API Integration ===")
    
    try:
        from app.api.v1.agents import agent_manager, _orchestration_available
        
        if _orchestration_available:
            print("✓ Orchestration components available in API")
            
            # Check if manager has registry
            if hasattr(agent_manager, 'registry') and agent_manager.registry:
                stats = agent_manager.registry.get_statistics()
                print(f"✓ API can access {stats['total_agents']} agents")
                return True
            else:
                print("✗ API manager missing registry")
                return False
        else:
            print("✗ Orchestration not available in API")
            return False
            
    except Exception as e:
        print(f"✗ Failed to import API: {e}")
        return False

async def main():
    """Run all tests"""
    print("="*60)
    print("UNIFIED AGENT ORCHESTRATION SYSTEM TEST")
    print("="*60)
    
    results = []
    
    # Test registry
    results.append(("Registry", test_registry()))
    
    # Test selector
    results.append(("Selector", test_selector()))
    
    # Test executor
    results.append(("Executor", await test_executor()))
    
    # Test pool
    results.append(("Pool", await test_pool()))
    
    # Test API integration
    results.append(("API Integration", test_api_integration()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20} {status}")
    
    total_passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 ALL TESTS PASSED - Orchestration system is working!")
    else:
        print("\n⚠️  Some tests failed - Review implementation")

if __name__ == "__main__":
    asyncio.run(main())