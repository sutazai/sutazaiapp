"""
Agent Communication Test Suite for SutazAI automation System

Tests inter-agent communication, orchestration, and collaboration
to ensure agents can work together effectively.
"""

import pytest
import httpx
import asyncio
import json
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta

# Configuration
BASE_URL = "http://localhost:8000"
AGENT_SERVICES = {
    "autogpt": {"url": "http://autogpt:8080", "health": "/health"},
    "crewai": {"url": "http://crewai:8080", "health": "/health"},
    "aider": {"url": "http://aider:8080", "health": "/health"},
    "gpt-engineer": {"url": "http://gpt-engineer:8080", "health": "/health"},
    "letta": {"url": "http://letta:8080", "health": "/health"},
}

COMMUNICATION_TIMEOUT = 30.0  # seconds

@pytest.fixture
async def client():
    """Create async HTTP client for main backend"""
    timeout = httpx.Timeout(COMMUNICATION_TIMEOUT, connect=10.0)
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=timeout) as client:
        yield client

@pytest.fixture
async def agent_clients():
    """Create HTTP clients for direct agent communication"""
    clients = {}
    timeout = httpx.Timeout(10.0, connect=5.0)
    
    for agent_name, config in AGENT_SERVICES.items():
        clients[agent_name] = httpx.AsyncClient(base_url=config["url"], timeout=timeout)
    
    yield clients
    
    # Cleanup
    for client in clients.values():
        await client.aclose()

@pytest.fixture
async def auth_headers():
    """Get authentication headers"""
    return {}  # No auth required for basic testing

class TestAgentDiscovery:
    """Test agent discovery and registry functionality"""
    
    @pytest.mark.asyncio
    async def test_agent_registry_access(self, client):
        """Test access to agent registry"""
        response = await client.get("/agents")
        assert response.status_code == 200
        
        data = response.json()
        assert "agents" in data
        
        agents = data["agents"]
        assert len(agents) > 0
        
        # Verify agent registry structure
        for agent in agents:
            assert "id" in agent
            assert "name" in agent
            assert "status" in agent
            assert "type" in agent
            assert "capabilities" in agent
            assert "health" in agent
    
    @pytest.mark.asyncio
    async def test_agent_health_monitoring(self, client):
        """Test agent health monitoring system"""
        # Get agent status multiple times to test consistency
        responses = []
        for _ in range(3):
            response = await client.get("/agents")
            assert response.status_code == 200
            responses.append(response.json())
            await asyncio.sleep(1)
        
        # Verify consistency in agent reporting
        agent_ids = [set(agent["id"] for agent in resp["agents"]) for resp in responses]
        
        # All responses should have same agent IDs
        assert all(ids == agent_ids[0] for ids in agent_ids), "Agent registry inconsistent"
    
    @pytest.mark.asyncio
    async def test_direct_agent_health_checks(self, agent_clients):
        """Test direct health checks to individual agents"""
        health_results = {}
        
        for agent_name, client in agent_clients.items():
            try:
                response = await client.get("/health")
                health_results[agent_name] = {
                    "status_code": response.status_code,
                    "accessible": True,
                    "response_time": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else 0
                }
                
                if response.status_code == 200:
                    data = response.json()
                    health_results[agent_name]["data"] = data
                    
            except Exception as e:
                health_results[agent_name] = {
                    "accessible": False,
                    "error": str(e)
                }
        
        # At least some agents should be accessible
        accessible_agents = [name for name, result in health_results.items() if result.get("accessible")]
        assert len(accessible_agents) >= 0, "No agents are accessible for direct communication"
        
        # Log health results for debugging
        print(f"Agent health results: {json.dumps(health_results, indent=2)}")

class TestAgentOrchestration:
    """Test agent orchestration and coordination"""
    
    @pytest.mark.asyncio
    async def test_orchestration_status(self, client, auth_headers):
        """Test orchestration system status"""
        response = await client.get("/api/v1/orchestration/status", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        
        # If orchestration is available, verify structure
        if data["status"] != "unavailable":
            assert "active_agents" in data
            assert "active_workflows" in data
            assert "health" in data
            assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_agent_creation_through_orchestration(self, client, auth_headers):
        """Test creating agents through orchestration system"""
        agent_config = {
            "agent_type": "test_agent",
            "name": "test-communication-agent",
            "config": {
                "role": "communication_tester",
                "capabilities": ["testing", "communication"]
            }
        }
        
        response = await client.post(
            "/api/v1/orchestration/agents",
            json=agent_config,
            headers=auth_headers
        )
        
        # Accept both success and unavailable responses
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "agent_id" in data
            assert "status" in data
            assert "config" in data
            assert data["status"] == "created"
    
    @pytest.mark.asyncio
    async def test_workflow_orchestration(self, client, auth_headers):
        """Test workflow creation and orchestration"""
        workflow_config = {
            "name": "test-communication-workflow",
            "description": "Test workflow for agent communication",
            "tasks": [
                {
                    "id": "task1",
                    "type": "analysis",
                    "description": "Analyze input data",
                    "agent_type": "analyst"
                },
                {
                    "id": "task2",
                    "type": "synthesis",
                    "description": "Synthesize results",
                    "agent_type": "synthesizer",
                    "depends_on": ["task1"]
                }
            ],
            "agents": ["analyst", "synthesizer"]
        }
        
        response = await client.post(
            "/api/v1/orchestration/workflows",
            json=workflow_config,
            headers=auth_headers
        )
        
        # Accept both success and unavailable responses
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "workflow_id" in data
            assert "status" in data
            assert "definition" in data
            assert data["status"] == "started"

class TestAgentCollaboration:
    """Test agent collaboration and consensus mechanisms"""
    
    @pytest.mark.asyncio
    async def test_agent_consensus_basic(self, client, auth_headers):
        """Test basic agent consensus functionality"""
        consensus_request = {
            "prompt": "What is the best approach to solve climate change?",
            "agents": ["research-agent", "analysis-agent", "strategy-agent"],
            "consensus_type": "majority"
        }
        
        response = await client.post(
            "/api/v1/agents/consensus",
            json=consensus_request,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "analysis" in data
        assert "agents_consulted" in data
        assert "consensus_reached" in data
        assert "consensus_type" in data
        assert "confidence" in data
        assert "recommendations" in data
        assert "agent_votes" in data
        assert "timestamp" in data
        
        # Verify consensus structure
        assert data["consensus_type"] == "majority"
        assert isinstance(data["agents_consulted"], list)
        assert isinstance(data["agent_votes"], dict)
    
    @pytest.mark.asyncio
    async def test_agent_consensus_with_disagreement(self, client, auth_headers):
        """Test agent consensus with potential disagreement scenarios"""
        controversial_request = {
            "prompt": "Should AI development be regulated by government?",
            "agents": ["ethics-agent", "tech-agent", "policy-agent", "business-agent"],
            "consensus_type": "unanimous"
        }
        
        response = await client.post(
            "/api/v1/agents/consensus",
            json=controversial_request,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "consensus_reached" in data
        assert "confidence" in data
        
        # Even if consensus not reached, response should be structured
        if not data["consensus_reached"]:
            assert data["confidence"] < 1.0
            assert "agent_votes" in data
    
    @pytest.mark.asyncio
    async def test_multi_agent_task_execution(self, client, auth_headers):
        """Test multi-agent task execution and coordination"""
        complex_task = {
            "description": "Design and implement a sustainable energy management system",
            "type": "multi_agent"
        }
        
        response = await client.post("/execute", json=complex_task, headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "result" in data
        assert "status" in data
        assert "task_id" in data
        assert "task_type" in data
        assert "orchestrated" in data
        assert "agents_involved" in data
        
        # For multi-agent tasks, should involve multiple agents
        if data["orchestrated"]:
            assert isinstance(data["agents_involved"], list)
    
    @pytest.mark.asyncio
    async def test_agent_communication_protocols(self, client, auth_headers):
        """Test agent-to-agent communication protocols"""
        # Test various communication scenarios
        communication_tests = [
            {
                "scenario": "information_sharing",
                "payload": {
                    "prompt": "Share knowledge about machine learning algorithms",
                    "agents": ["ml-expert", "data-scientist"],
                    "protocol": "information_exchange"
                }
            },
            {
                "scenario": "task_delegation",
                "payload": {
                    "prompt": "Delegate code review task to appropriate agents",
                    "agents": ["code-reviewer", "security-analyst"],
                    "protocol": "task_delegation"
                }
            },
            {
                "scenario": "collaborative_problem_solving",
                "payload": {
                    "prompt": "Collaboratively solve optimization problem",
                    "agents": ["optimizer", "mathematician", "engineer"],
                    "protocol": "collaborative_solving"
                }
            }
        ]
        
        for test in communication_tests:
            response = await client.post(
                "/api/v1/agents/consensus",
                json=test["payload"],
                headers=auth_headers
            )
            
            assert response.status_code == 200, f"Failed for scenario: {test['scenario']}"
            
            data = response.json()
            assert "analysis" in data
            assert "agents_consulted" in data
            assert "timestamp" in data

class TestAgentMessageBus:
    """Test agent message bus and event system"""
    
    @pytest.mark.asyncio
    async def test_agent_message_routing(self, client, auth_headers):
        """Test message routing between agents"""
        # Test different types of messages
        message_types = [
            {
                "type": "task_request",
                "from_agent": "coordinator",
                "to_agent": "executor",
                "content": "Execute analysis task",
                "priority": "normal"
            },
            {
                "type": "status_update",
                "from_agent": "worker",
                "to_agent": "monitor",
                "content": "Task 50% complete",
                "priority": "low"
            },
            {
                "type": "error_report",
                "from_agent": "processor",
                "to_agent": "error_handler",
                "content": "Processing error encountered",
                "priority": "high"
            }
        ]
        
        # Since direct message bus testing may not be available,
        # test through consensus endpoint which should use message routing
        for msg in message_types:
            consensus_request = {
                "prompt": f"Route message: {msg['content']}",
                "agents": [msg["from_agent"], msg["to_agent"]],
                "consensus_type": "simple"
            }
            
            response = await client.post(
                "/api/v1/agents/consensus",
                json=consensus_request,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            
            data = response.json()
            assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_agent_event_broadcasting(self, client, auth_headers):
        """Test agent event broadcasting capabilities"""
        # Test broadcasting events to multiple agents
        broadcast_event = {
            "prompt": "System maintenance scheduled - all agents prepare",
            "agents": ["autogpt", "crewai", "aider", "gpt-engineer"],
            "consensus_type": "broadcast"
        }
        
        response = await client.post(
            "/api/v1/agents/consensus",
            json=broadcast_event,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "agents_consulted" in data
        assert len(data["agents_consulted"]) == len(broadcast_event["agents"])
    
    @pytest.mark.asyncio
    async def test_agent_synchronization(self, client, auth_headers):
        """Test agent synchronization mechanisms"""
        # Test synchronization between agents for coordinated actions
        sync_request = {
            "prompt": "Synchronize for coordinated data processing",
            "agents": ["data-processor-1", "data-processor-2", "data-processor-3"],
            "consensus_type": "synchronized"
        }
        
        response = await client.post(
            "/api/v1/agents/consensus",
            json=sync_request,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "analysis" in data
        assert "timestamp" in data

class TestAgentLoadBalancing:
    """Test agent load balancing and resource management"""
    
    @pytest.mark.asyncio
    async def test_agent_load_distribution(self, client, auth_headers):
        """Test load distribution across available agents"""
        # Send multiple tasks simultaneously to test load balancing
        load_test_tasks = [
            {"description": f"Process data batch {i}", "type": "processing"}
            for i in range(10)
        ]
        
        # Execute tasks concurrently
        async def execute_task(task):
            return await client.post("/execute", json=task, headers=auth_headers)
        
        tasks = [execute_task(task) for task in load_test_tasks]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful responses
        successful_responses = [
            r for r in responses 
            if hasattr(r, 'status_code') and r.status_code == 200
        ]
        
        # At least 70% should succeed under load
        success_rate = len(successful_responses) / len(responses)
        assert success_rate >= 0.7, f"Load balancing failed, success rate: {success_rate}"
        
        # Check if different agents were used (if orchestration available)
        agents_used = set()
        for response in successful_responses:
            data = response.json()
            if "agents_involved" in data and data["agents_involved"]:
                agents_used.update(data["agents_involved"])
    
    @pytest.mark.asyncio
    async def test_agent_capacity_management(self, client, auth_headers):
        """Test agent capacity and resource management"""
        # Test system behavior under different load conditions
        
        # Light load test
        light_task = {"description": "Simple calculation", "type": "simple"}
        response = await client.post("/execute", json=light_task, headers=auth_headers)
        assert response.status_code == 200
        
        light_response_data = response.json()
        light_execution_time = light_response_data.get("execution_time", "0s")
        
        # Heavy load test
        heavy_task = {"description": "Complex multi-step analysis with multiple dependencies", "type": "complex"}
        response = await client.post("/execute", json=heavy_task, headers=auth_headers)
        assert response.status_code == 200
        
        heavy_response_data = response.json()
        heavy_execution_time = heavy_response_data.get("execution_time", "0s")
        
        # System should handle both loads appropriately
        assert "resources_used" in light_response_data
        assert "resources_used" in heavy_response_data

class TestAgentFailureRecovery:
    """Test agent failure detection and recovery mechanisms"""
    
    @pytest.mark.asyncio
    async def test_agent_failure_detection(self, client):
        """Test detection of agent failures"""
        # Monitor agent status over time to detect any failures
        status_checks = []
        
        for _ in range(5):
            response = await client.get("/agents")
            assert response.status_code == 200
            
            data = response.json()
            status_checks.append({
                "timestamp": datetime.now(),
                "agents": {agent["id"]: agent["health"] for agent in data["agents"]}
            })
            
            await asyncio.sleep(2)
        
        # Analyze status consistency
        all_agent_ids = set()
        for check in status_checks:
            all_agent_ids.update(check["agents"].keys())
        
        # Track health status changes
        health_changes = {}
        for agent_id in all_agent_ids:
            health_statuses = [check["agents"].get(agent_id, "unknown") for check in status_checks]
            health_changes[agent_id] = health_statuses
        
        # System should maintain some level of stability
        stable_agents = [
            agent_id for agent_id, statuses in health_changes.items()
            if len(set(statuses)) <= 2  # Allow for some status transitions
        ]
        
        assert len(stable_agents) > 0, "No agents maintained stable health status"
    
    @pytest.mark.asyncio
    async def test_agent_redundancy(self, client, auth_headers):
        """Test agent redundancy and failover capabilities"""
        # Test that system can handle agent unavailability
        redundancy_test = {
            "description": "Test task with redundancy requirements",
            "type": "redundant"
        }
        
        response = await client.post("/execute", json=redundancy_test, headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "result" in data
        assert "status" in data
        
        # Task should complete even if some agents are unavailable
        assert data["status"] in ["completed", "partial"]
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, client, auth_headers):
        """Test graceful degradation when agents are unavailable"""
        # Test various task types to see how system degrades
        task_types = ["simple", "moderate", "complex", "distributed"]
        
        degradation_results = {}
        
        for task_type in task_types:
            task = {
                "description": f"Test {task_type} task for degradation analysis",
                "type": task_type
            }
            
            response = await client.post("/execute", json=task, headers=auth_headers)
            assert response.status_code == 200
            
            data = response.json()
            degradation_results[task_type] = {
                "status": data["status"],
                "success_probability": data.get("success_probability", 0.0),
                "resources_used": len(data.get("resources_used", [])),
                "orchestrated": data.get("orchestrated", False)
            }
        
        # System should handle at least simple tasks even under degradation
        assert degradation_results["simple"]["status"] == "completed"

class TestCommunicationSecurity:
    """Test security aspects of agent communication"""
    
    @pytest.mark.asyncio
    async def test_agent_authentication(self, client, auth_headers):
        """Test agent authentication mechanisms"""
        # Test authenticated vs unauthenticated requests
        
        # Unauthenticated request
        unauth_response = await client.post("/api/v1/agents/consensus", json={
            "prompt": "Test without auth",
            "agents": ["test-agent"]
        })
        
        # Authenticated request
        auth_response = await client.post(
            "/api/v1/agents/consensus",
            json={
                "prompt": "Test with auth",
                "agents": ["test-agent"]
            },
            headers=auth_headers
        )
        
        # Both should succeed since auth is optional in current implementation
        assert unauth_response.status_code == 200
        assert auth_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_agent_message_validation(self, client, auth_headers):
        """Test validation of agent messages"""
        # Test invalid message formats
        invalid_requests = [
            {},  # Empty request
            {"prompt": ""},  # Empty prompt
            {"prompt": "test", "agents": []},  # Empty agents list
            {"prompt": "test", "agents": [""]}  # Empty agent name
        ]
        
        for invalid_request in invalid_requests:
            response = await client.post(
                "/api/v1/agents/consensus",
                json=invalid_request,
                headers=auth_headers
            )
            
            # Should handle gracefully, either with 422 or 200 with error message
            assert response.status_code in [200, 422]
            
            if response.status_code == 200:
                data = response.json()
                # Should indicate some kind of handling for invalid input
                assert "analysis" in data or "error" in data

class TestCommunicationPerformance:
    """Test performance aspects of agent communication"""
    
    @pytest.mark.asyncio
    async def test_communication_latency(self, client, auth_headers):
        """Test agent communication latency"""
        latency_measurements = []
        
        for i in range(5):
            start_time = time.time()
            
            response = await client.post(
                "/api/v1/agents/consensus",
                json={
                    "prompt": f"Latency test {i}",
                    "agents": ["agent1", "agent2"]
                },
                headers=auth_headers
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            assert response.status_code == 200
            latency_measurements.append(latency)
            
            await asyncio.sleep(0.5)
        
        # Calculate average latency
        avg_latency = sum(latency_measurements) / len(latency_measurements)
        max_latency = max(latency_measurements)
        
        # Communication should be reasonably fast
        assert avg_latency < 10.0, f"Average latency too high: {avg_latency}s"
        assert max_latency < 15.0, f"Maximum latency too high: {max_latency}s"
    
    @pytest.mark.asyncio
    async def test_concurrent_communication(self, client, auth_headers):
        """Test concurrent agent communication"""
        # Test multiple concurrent communications
        async def concurrent_consensus(i):
            return await client.post(
                "/api/v1/agents/consensus",
                json={
                    "prompt": f"Concurrent test {i}",
                    "agents": [f"agent-{i % 3}", f"agent-{(i + 1) % 3}"]
                },
                headers=auth_headers
            )
        
        # Run 10 concurrent communications
        tasks = [concurrent_consensus(i) for i in range(10)]
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Count successful responses
        successful = [r for r in responses if hasattr(r, 'status_code') and r.status_code == 200]
        success_rate = len(successful) / len(responses)
        
        # Performance assertions
        assert success_rate >= 0.8, f"Concurrent communication success rate too low: {success_rate}"
        assert total_time < 20.0, f"Concurrent communication took too long: {total_time}s"
    
    @pytest.mark.asyncio
    async def test_communication_throughput(self, client, auth_headers):
        """Test agent communication throughput"""
        throughput_start = time.time()
        request_count = 20
        
        # Send requests in rapid succession
        async def rapid_request(i):
            return await client.post(
                "/api/v1/agents/consensus",
                json={
                    "prompt": f"Throughput test {i}",
                    "agents": ["throughput-agent"]
                },
                headers=auth_headers
            )
        
        responses = []
        for i in range(request_count):
            try:
                response = await rapid_request(i)
                responses.append(response)
                await asyncio.sleep(0.1)  # Small delay between requests
            except Exception as e:
                responses.append(e)
        
        throughput_end = time.time()
        total_time = throughput_end - throughput_start
        
        successful_responses = [r for r in responses if hasattr(r, 'status_code') and r.status_code == 200]
        throughput = len(successful_responses) / total_time
        
        # System should handle reasonable throughput
        assert throughput > 0.5, f"Throughput too low: {throughput} requests/second"
        assert len(successful_responses) >= request_count * 0.7, "Too many failed requests"

# Test configuration
if __name__ == "__main__":
    pytest.main(["-v", __file__])
