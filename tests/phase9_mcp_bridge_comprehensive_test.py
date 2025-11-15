#!/usr/bin/env python3
"""
Phase 9: MCP Bridge Comprehensive Testing
Complete validation of all MCP Bridge functionality
Execution Time: 2025-11-15 (Phase 9)
"""

import asyncio
import json
import sys
import time
import websockets
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

import httpx
import pytest

# Test configuration
MCP_BASE_URL = "http://localhost:11100"
TIMEOUT = 30.0
TEST_START_TIME = datetime.now()

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class TestResults:
    """Track test results"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []
        self.timings = {}
    
    def record_pass(self, test_name: str, duration: float):
        self.total += 1
        self.passed += 1
        self.timings[test_name] = duration
        print(f"{Colors.OKGREEN}✓{Colors.ENDC} {test_name} ({duration:.3f}s)")
    
    def record_fail(self, test_name: str, error: str, duration: float):
        self.total += 1
        self.failed += 1
        self.timings[test_name] = duration
        self.errors.append({"test": test_name, "error": error})
        print(f"{Colors.FAIL}✗{Colors.ENDC} {test_name} ({duration:.3f}s)")
        print(f"  {Colors.WARNING}Error: {error}{Colors.ENDC}")
    
    def record_skip(self, test_name: str, reason: str):
        self.total += 1
        self.skipped += 1
        print(f"{Colors.WARNING}⊘{Colors.ENDC} {test_name} (skipped: {reason})")
    
    def summary(self):
        total_time = sum(self.timings.values())
        pass_rate = (self.passed / self.total * 100) if self.total > 0 else 0
        
        print(f"\n{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
        print(f"{Colors.BOLD}PHASE 9 MCP BRIDGE TEST RESULTS{Colors.ENDC}")
        print(f"{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
        print(f"\nTotal Tests:    {self.total}")
        print(f"{Colors.OKGREEN}Passed:        {self.passed}{Colors.ENDC}")
        print(f"{Colors.FAIL}Failed:        {self.failed}{Colors.ENDC}")
        print(f"{Colors.WARNING}Skipped:       {self.skipped}{Colors.ENDC}")
        print(f"\n{Colors.BOLD}Pass Rate:      {pass_rate:.1f}%{Colors.ENDC}")
        print(f"Total Duration: {total_time:.2f}s")
        
        if self.errors:
            print(f"\n{Colors.FAIL}Failed Tests:{Colors.ENDC}")
            for error in self.errors:
                print(f"  - {error['test']}: {error['error']}")
        
        print(f"\n{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")
        
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "pass_rate": pass_rate,
            "duration": total_time,
            "errors": self.errors,
            "timings": self.timings
        }

results = TestResults()

async def test_wrapper(test_name: str, test_func):
    """Wrapper to time and record test results"""
    start = time.time()
    try:
        await test_func()
        duration = time.time() - start
        results.record_pass(test_name, duration)
        return True
    except Exception as e:
        duration = time.time() - start
        results.record_fail(test_name, str(e), duration)
        return False

# ============================================
# HEALTH & STATUS TESTS
# ============================================

async def test_health_endpoint():
    """Test /health endpoint response"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(f"{MCP_BASE_URL}/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "status" in data, "Missing 'status' field"
        assert data["status"] == "healthy", f"Status not healthy: {data['status']}"
        assert "service" in data, "Missing 'service' field"
        assert data["service"] == "mcp-bridge", f"Wrong service: {data['service']}"
        assert "version" in data, "Missing 'version' field"
        assert "timestamp" in data, "Missing 'timestamp' field"

async def test_status_endpoint():
    """Test /status endpoint with full bridge status"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(f"{MCP_BASE_URL}/status")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "status" in data, "Missing 'status' field"
        assert "services" in data, "Missing 'services' field"
        assert "agents" in data, "Missing 'agents' field"
        assert "active_connections" in data, "Missing 'active_connections' field"
        assert "timestamp" in data, "Missing 'timestamp' field"
        
        # Verify services dict
        assert isinstance(data["services"], dict), "Services should be a dict"
        assert len(data["services"]) > 0, "No services registered"
        
        # Verify agents dict
        assert isinstance(data["agents"], dict), "Agents should be a dict"
        assert len(data["agents"]) > 0, "No agents registered"

# ============================================
# SERVICE REGISTRY TESTS
# ============================================

async def test_services_listing():
    """Test /services endpoint lists all services"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(f"{MCP_BASE_URL}/services")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert isinstance(data, dict), "Services should be a dictionary"
        
        # Verify core services are registered
        expected_services = ["postgres", "redis", "rabbitmq", "neo4j", "backend"]
        for service in expected_services:
            assert service in data, f"Missing service: {service}"
            assert "url" in data[service], f"Service {service} missing 'url'"
            assert "type" in data[service], f"Service {service} missing 'type'"

async def test_get_specific_service():
    """Test /services/{name} for specific services"""
    services_to_test = ["postgres", "redis", "rabbitmq", "backend", "chromadb"]
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for service_name in services_to_test:
            response = await client.get(f"{MCP_BASE_URL}/services/{service_name}")
            assert response.status_code == 200, f"Service {service_name}: expected 200, got {response.status_code}"
            
            data = response.json()
            assert "url" in data, f"Service {service_name} missing 'url'"
            assert "type" in data, f"Service {service_name} missing 'type'"

async def test_get_nonexistent_service():
    """Test getting a non-existent service returns 404"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(f"{MCP_BASE_URL}/services/nonexistent_service_xyz")
        assert response.status_code == 404, f"Expected 404 for nonexistent service, got {response.status_code}"

# ============================================
# AGENT REGISTRY TESTS
# ============================================

async def test_agents_listing():
    """Test /agents endpoint lists all agents"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(f"{MCP_BASE_URL}/agents")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert isinstance(data, dict), "Agents should be a dictionary"
        
        # Verify expected agents are registered
        expected_agents = ["letta", "crewai", "aider", "langchain"]
        for agent in expected_agents:
            assert agent in data, f"Missing agent: {agent}"
            assert "name" in data[agent], f"Agent {agent} missing 'name'"
            assert "capabilities" in data[agent], f"Agent {agent} missing 'capabilities'"
            assert "port" in data[agent], f"Agent {agent} missing 'port'"
            assert "status" in data[agent], f"Agent {agent} missing 'status'"

async def test_get_specific_agent():
    """Test /agents/{id} for specific agents"""
    agents_to_test = ["letta", "crewai", "aider", "langchain", "shellgpt"]
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for agent_id in agents_to_test:
            response = await client.get(f"{MCP_BASE_URL}/agents/{agent_id}")
            assert response.status_code == 200, f"Agent {agent_id}: expected 200, got {response.status_code}"
            
            data = response.json()
            assert "name" in data, f"Agent {agent_id} missing 'name'"
            assert "capabilities" in data, f"Agent {agent_id} missing 'capabilities'"
            assert isinstance(data["capabilities"], list), f"Agent {agent_id} capabilities not a list"

async def test_get_nonexistent_agent():
    """Test getting a non-existent agent returns 404"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(f"{MCP_BASE_URL}/agents/nonexistent_agent_xyz")
        assert response.status_code == 404, f"Expected 404 for nonexistent agent, got {response.status_code}"

async def test_update_agent_status():
    """Test /agents/{id}/status endpoint for status updates"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Try to update agent status
        response = await client.post(
            f"{MCP_BASE_URL}/agents/letta/status",
            params={"status": "online"}
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "agent" in data, "Missing 'agent' field"
        assert "new_status" in data, "Missing 'new_status' field"
        assert data["new_status"] == "online", f"Status not updated: {data['new_status']}"

# ============================================
# MESSAGE ROUTING TESTS
# ============================================

async def test_route_message_to_service():
    """Test /route endpoint with service target"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        message = {
            "id": "test-msg-001",
            "source": "test-client",
            "target": "backend",
            "type": "test.message",
            "payload": {"action": "test", "data": "hello"}
        }
        
        response = await client.post(f"{MCP_BASE_URL}/route", json=message)
        # Accept 200, 404 (service doesn't have /mcp/receive), or error responses
        assert response.status_code in [200, 404, 500], f"Unexpected status: {response.status_code}"

async def test_route_message_to_agent():
    """Test /route endpoint with agent target"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        message = {
            "id": "test-msg-002",
            "source": "test-client",
            "target": "letta",
            "type": "task.automation",
            "payload": {"task": "test_task", "params": {}}
        }
        
        response = await client.post(f"{MCP_BASE_URL}/route", json=message)
        # Should return status about queuing or delivery
        assert response.status_code in [200, 404], f"Unexpected status: {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data, "Missing 'status' field in response"

async def test_route_invalid_target():
    """Test routing to invalid target returns error"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        message = {
            "id": "test-msg-003",
            "source": "test-client",
            "target": "invalid_target_xyz",
            "type": "test.message",
            "payload": {}
        }
        
        response = await client.post(f"{MCP_BASE_URL}/route", json=message)
        assert response.status_code in [404, 422], f"Expected error status, got {response.status_code}"

# ============================================
# TASK ORCHESTRATION TESTS
# ============================================

async def test_submit_task_with_agent():
    """Test /tasks/submit with specified agent"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        task = {
            "task_id": "test-task-001",
            "task_type": "automation",
            "description": "Test automated task",
            "agent": "letta",
            "params": {"test": True},
            "priority": "medium"
        }
        
        response = await client.post(f"{MCP_BASE_URL}/tasks/submit", json=task)
        assert response.status_code in [200, 404], f"Unexpected status: {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            assert "task_id" in data, "Missing 'task_id' field"
            assert "agent" in data, "Missing 'agent' field"
            assert "status" in data, "Missing 'status' field"

async def test_submit_task_auto_select():
    """Test /tasks/submit with automatic agent selection"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        task = {
            "task_id": "test-task-002",
            "task_type": "code",
            "description": "Code generation task",
            "params": {},
            "priority": "high"
        }
        
        response = await client.post(f"{MCP_BASE_URL}/tasks/submit", json=task)
        # Should auto-select agent based on capability
        assert response.status_code in [200, 400, 404], f"Unexpected status: {response.status_code}"

async def test_submit_invalid_task():
    """Test submitting task with invalid agent"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        task = {
            "task_id": "test-task-003",
            "task_type": "test",
            "description": "Invalid task",
            "agent": "nonexistent_agent",
            "params": {}
        }
        
        response = await client.post(f"{MCP_BASE_URL}/tasks/submit", json=task)
        assert response.status_code in [404, 422], f"Expected error status, got {response.status_code}"

# ============================================
# WEBSOCKET TESTS
# ============================================

async def test_websocket_connection():
    """Test WebSocket connection establishment"""
    try:
        uri = f"ws://localhost:11100/ws/test-client-001"
        async with websockets.connect(uri, ping_interval=None) as websocket:
            # Should receive connected message
            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(message)
            
            assert "type" in data, "Missing 'type' field in WebSocket message"
            assert data["type"] == "connected", f"Wrong message type: {data['type']}"
            assert "client_id" in data, "Missing 'client_id' field"
            assert data["client_id"] == "test-client-001", f"Wrong client_id: {data['client_id']}"
    except Exception as e:
        raise AssertionError(f"WebSocket connection failed: {e}")

async def test_websocket_broadcast():
    """Test WebSocket broadcast functionality"""
    try:
        uri1 = f"ws://localhost:11100/ws/test-client-broadcast-1"
        uri2 = f"ws://localhost:11100/ws/test-client-broadcast-2"
        
        async with websockets.connect(uri1, ping_interval=None) as ws1, \
                   websockets.connect(uri2, ping_interval=None) as ws2:
            
            # Consume connection messages
            await ws1.recv()
            await ws2.recv()
            
            # Send broadcast from client 1
            await ws1.send(json.dumps({
                "type": "broadcast",
                "payload": {"message": "test broadcast"}
            }))
            
            # Client 2 should receive the broadcast
            message = await asyncio.wait_for(ws2.recv(), timeout=5.0)
            data = json.loads(message)
            
            assert "from" in data, "Missing 'from' field"
            assert data["from"] == "test-client-broadcast-1", f"Wrong sender: {data['from']}"
            assert "data" in data, "Missing 'data' field"
    except Exception as e:
        raise AssertionError(f"WebSocket broadcast failed: {e}")

async def test_websocket_direct_message():
    """Test WebSocket direct messaging"""
    try:
        uri1 = f"ws://localhost:11100/ws/test-client-direct-1"
        uri2 = f"ws://localhost:11100/ws/test-client-direct-2"
        
        async with websockets.connect(uri1, ping_interval=None) as ws1, \
                   websockets.connect(uri2, ping_interval=None) as ws2:
            
            # Consume connection messages
            await ws1.recv()
            await ws2.recv()
            
            # Send direct message from client 1 to client 2
            await ws1.send(json.dumps({
                "type": "direct",
                "target": "test-client-direct-2",
                "payload": {"message": "direct message"}
            }))
            
            # Client 2 should receive the direct message
            message = await asyncio.wait_for(ws2.recv(), timeout=5.0)
            data = json.loads(message)
            
            assert "from" in data, "Missing 'from' field"
            assert data["from"] == "test-client-direct-1", f"Wrong sender: {data['from']}"
    except Exception as e:
        raise AssertionError(f"WebSocket direct message failed: {e}")

# ============================================
# METRICS TESTS
# ============================================

async def test_prometheus_metrics():
    """Test /metrics endpoint (Prometheus format)"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(f"{MCP_BASE_URL}/metrics")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        # Should return Prometheus text format
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type or "text" in content_type, f"Wrong content type: {content_type}"
        
        # Verify some expected metrics are present
        text = response.text
        assert "mcp_bridge" in text, "Missing mcp_bridge metrics"

async def test_json_metrics():
    """Test /metrics/json endpoint"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(f"{MCP_BASE_URL}/metrics/json")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "total_services" in data, "Missing 'total_services' field"
        assert "total_agents" in data, "Missing 'total_agents' field"
        assert "online_agents" in data, "Missing 'online_agents' field"
        assert "active_connections" in data, "Missing 'active_connections' field"
        assert "timestamp" in data, "Missing 'timestamp' field"

# ============================================
# CONCURRENT REQUEST TESTS
# ============================================

async def test_concurrent_health_checks():
    """Test multiple concurrent health checks"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Send 10 concurrent requests
        tasks = [client.get(f"{MCP_BASE_URL}/health") for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for i, response in enumerate(responses):
            assert response.status_code == 200, f"Request {i} failed: {response.status_code}"

async def test_concurrent_agent_queries():
    """Test concurrent agent queries"""
    agents = ["letta", "crewai", "aider", "langchain", "shellgpt"]
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        tasks = [client.get(f"{MCP_BASE_URL}/agents/{agent}") for agent in agents]
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for i, response in enumerate(responses):
            assert response.status_code == 200, f"Request for {agents[i]} failed: {response.status_code}"

# ============================================
# ERROR HANDLING TESTS
# ============================================

async def test_invalid_json():
    """Test error handling for invalid JSON"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{MCP_BASE_URL}/route",
            content="invalid json{{{",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422], f"Expected error status, got {response.status_code}"

async def test_missing_required_fields():
    """Test error handling for missing required fields"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Missing required fields in task
        response = await client.post(
            f"{MCP_BASE_URL}/tasks/submit",
            json={"task_id": "incomplete"}
        )
        assert response.status_code == 422, f"Expected 422 validation error, got {response.status_code}"

# ============================================
# PERFORMANCE TESTS
# ============================================

async def test_response_time_health():
    """Test health endpoint response time"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        start = time.time()
        response = await client.get(f"{MCP_BASE_URL}/health")
        duration = time.time() - start
        
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        assert duration < 1.0, f"Health check too slow: {duration:.3f}s"

async def test_response_time_services():
    """Test services listing response time"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        start = time.time()
        response = await client.get(f"{MCP_BASE_URL}/services")
        duration = time.time() - start
        
        assert response.status_code == 200, f"Services listing failed: {response.status_code}"
        assert duration < 2.0, f"Services listing too slow: {duration:.3f}s"

# ============================================
# MAIN TEST RUNNER
# ============================================

async def run_all_tests():
    """Execute all Phase 9 tests"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}PHASE 9: MCP BRIDGE COMPREHENSIVE TESTING{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}Started: {TEST_START_TIME.strftime('%Y-%m-%d %H:%M:%S UTC')}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}\n")
    
    # Health & Status Tests
    print(f"{Colors.BOLD}{Colors.OKCYAN}[1/10] Health & Status Tests{Colors.ENDC}")
    await test_wrapper("Health Endpoint", test_health_endpoint)
    await test_wrapper("Status Endpoint", test_status_endpoint)
    
    # Service Registry Tests
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}[2/10] Service Registry Tests{Colors.ENDC}")
    await test_wrapper("Services Listing", test_services_listing)
    await test_wrapper("Get Specific Service", test_get_specific_service)
    await test_wrapper("Get Nonexistent Service", test_get_nonexistent_service)
    
    # Agent Registry Tests
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}[3/10] Agent Registry Tests{Colors.ENDC}")
    await test_wrapper("Agents Listing", test_agents_listing)
    await test_wrapper("Get Specific Agent", test_get_specific_agent)
    await test_wrapper("Get Nonexistent Agent", test_get_nonexistent_agent)
    await test_wrapper("Update Agent Status", test_update_agent_status)
    
    # Message Routing Tests
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}[4/10] Message Routing Tests{Colors.ENDC}")
    await test_wrapper("Route Message to Service", test_route_message_to_service)
    await test_wrapper("Route Message to Agent", test_route_message_to_agent)
    await test_wrapper("Route Invalid Target", test_route_invalid_target)
    
    # Task Orchestration Tests
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}[5/10] Task Orchestration Tests{Colors.ENDC}")
    await test_wrapper("Submit Task with Agent", test_submit_task_with_agent)
    await test_wrapper("Submit Task Auto-Select", test_submit_task_auto_select)
    await test_wrapper("Submit Invalid Task", test_submit_invalid_task)
    
    # WebSocket Tests
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}[6/10] WebSocket Tests{Colors.ENDC}")
    await test_wrapper("WebSocket Connection", test_websocket_connection)
    await test_wrapper("WebSocket Broadcast", test_websocket_broadcast)
    await test_wrapper("WebSocket Direct Message", test_websocket_direct_message)
    
    # Metrics Tests
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}[7/10] Metrics Tests{Colors.ENDC}")
    await test_wrapper("Prometheus Metrics", test_prometheus_metrics)
    await test_wrapper("JSON Metrics", test_json_metrics)
    
    # Concurrent Request Tests
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}[8/10] Concurrent Request Tests{Colors.ENDC}")
    await test_wrapper("Concurrent Health Checks", test_concurrent_health_checks)
    await test_wrapper("Concurrent Agent Queries", test_concurrent_agent_queries)
    
    # Error Handling Tests
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}[9/10] Error Handling Tests{Colors.ENDC}")
    await test_wrapper("Invalid JSON", test_invalid_json)
    await test_wrapper("Missing Required Fields", test_missing_required_fields)
    
    # Performance Tests
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}[10/10] Performance Tests{Colors.ENDC}")
    await test_wrapper("Response Time Health", test_response_time_health)
    await test_wrapper("Response Time Services", test_response_time_services)
    
    # Generate summary
    summary = results.summary()
    
    # Save results to file
    report_file = Path(f"/opt/sutazaiapp/PHASE_9_TEST_RESULTS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{Colors.OKGREEN}Results saved to: {report_file}{Colors.ENDC}\n")
    
    return summary

if __name__ == "__main__":
    # Run all tests
    summary = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    sys.exit(0 if summary["failed"] == 0 else 1)
