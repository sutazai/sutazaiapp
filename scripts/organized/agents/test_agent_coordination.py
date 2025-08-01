#!/usr/bin/env python3
"""
Test script to verify all agents are working together
"""

import asyncio
import aiohttp
import json
from datetime import datetime

async def test_mcp_server():
    """Test MCP server connectivity"""
    try:
        async with aiohttp.ClientSession() as session:
            # List resources
            async with session.get("http://localhost:8100/health") as resp:
                if resp.status == 200:
                    print("✓ MCP Server is healthy")
                    return True
    except Exception as e:
        print(f"✗ MCP Server error: {e}")
    return False

async def test_agent_deployment():
    """Test deploying agents via MCP"""
    agents_to_deploy = [
        {"agent_type": "autogpt", "name": "autogpt-agent"},
        {"agent_type": "crewai", "name": "crewai-agent"},
        {"agent_type": "localagi", "name": "localagi-agent"}
    ]
    
    try:
        async with aiohttp.ClientSession() as session:
            for agent in agents_to_deploy:
                async with session.post(
                    "http://localhost:8100/deploy_agent",
                    json=agent
                ) as resp:
                    if resp.status == 200:
                        print(f"✓ Deployed {agent['name']}")
                    else:
                        print(f"✗ Failed to deploy {agent['name']}: {resp.status}")
    except Exception as e:
        print(f"✗ Agent deployment error: {e}")

async def test_task_execution():
    """Test task execution through agents"""
    test_tasks = [
        {
            "agent_name": "task-assignment-coordinator",
            "task": "Analyze system performance and suggest optimizations",
            "priority": "high"
        },
        {
            "agent_name": "infrastructure-devops-manager",
            "task": "Check health of all running containers",
            "priority": "normal"
        }
    ]
    
    try:
        async with aiohttp.ClientSession() as session:
            for task in test_tasks:
                async with session.post(
                    "http://localhost:8100/execute_agent_task",
                    json=task
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"✓ Task executed: {task['task'][:50]}...")
                        print(f"  Result: {result}")
                    else:
                        print(f"✗ Task failed: {task['task'][:50]}...")
    except Exception as e:
        print(f"✗ Task execution error: {e}")

async def test_agent_coordination():
    """Test agent coordination capabilities"""
    try:
        async with aiohttp.ClientSession() as session:
            # Orchestrate a multi-agent task
            orchestration_request = {
                "task_description": "Deploy a new AI model with full testing",
                "agents": [
                    {"agent_name": "senior-ai-engineer", "role": "model preparation"},
                    {"agent_name": "testing-qa-validator", "role": "testing"},
                    {"agent_name": "deployment-automation-master", "role": "deployment"}
                ],
                "coordination_strategy": "sequential"
            }
            
            async with session.post(
                "http://localhost:8100/orchestrate_multi_agent",
                json=orchestration_request
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print("✓ Multi-agent orchestration successful")
                    print(f"  Result: {result}")
                else:
                    print(f"✗ Orchestration failed: {resp.status}")
    except Exception as e:
        print(f"✗ Coordination error: {e}")

async def test_system_monitoring():
    """Test system monitoring capabilities"""
    metric_types = ["cpu", "memory", "containers", "agents"]
    
    try:
        async with aiohttp.ClientSession() as session:
            for metric in metric_types:
                async with session.post(
                    "http://localhost:8100/monitor_system",
                    json={"metric_type": metric}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        print(f"✓ {metric.upper()} metrics retrieved")
                        if metric == "agents":
                            print(f"  Active agents: {data.get('active_count', 0)}")
                    else:
                        print(f"✗ Failed to get {metric} metrics")
    except Exception as e:
        print(f"✗ Monitoring error: {e}")

async def test_knowledge_base():
    """Test knowledge base queries"""
    queries = [
        "How to deploy agents?",
        "What are the available AI models?",
        "System architecture overview"
    ]
    
    try:
        async with aiohttp.ClientSession() as session:
            for query in queries:
                async with session.post(
                    "http://localhost:8100/query_knowledge_base",
                    json={"query": query}
                ) as resp:
                    if resp.status == 200:
                        results = await resp.json()
                        print(f"✓ Query: '{query}' - Found {len(results.get('results', []))} results")
                    else:
                        print(f"✗ Query failed: '{query}'")
    except Exception as e:
        print(f"✗ Knowledge base error: {e}")

async def main():
    """Run all tests"""
    print("=" * 60)
    print("SutazAI Agent Coordination Test Suite")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run tests
    print("1. Testing MCP Server...")
    await test_mcp_server()
    print()
    
    print("2. Testing Agent Deployment...")
    await test_agent_deployment()
    print()
    
    print("3. Testing Task Execution...")
    await test_task_execution()
    print()
    
    print("4. Testing Agent Coordination...")
    await test_agent_coordination()
    print()
    
    print("5. Testing System Monitoring...")
    await test_system_monitoring()
    print()
    
    print("6. Testing Knowledge Base...")
    await test_knowledge_base()
    print()
    
    print("=" * 60)
    print("Test suite completed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())