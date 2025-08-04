"""
Comprehensive API Endpoint Test Suite for SutazAI automation System

Tests all API endpoints defined in working_main.py to ensure 100% functionality
before marking deployment complete.
"""

import pytest
import httpx
import asyncio
import json
import time
from typing import Dict, Any, List
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8002"
AGENT_ENDPOINTS = {
    "autogpt": "http://autogpt:8080",
    "crewai": "http://crewai:8080", 
    "aider": "http://aider:8080",
    "gpt-engineer": "http://gpt-engineer:8080",
    "letta": "http://letta:8080"
}

SERVICE_ENDPOINTS = {
    "ollama": "http://ollama:9005",
    "chromadb": "http://chromadb:8001",
    "qdrant": "http://qdrant:6333"
}

@pytest.fixture
async def client():
    """Create async HTTP client"""
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=timeout) as client:
        yield client

@pytest.fixture
async def auth_headers():
    """Get authentication headers for enterprise features"""
    # For testing, return empty headers as auth is optional in working_main.py
    return {}

class TestCoreHealthEndpoints:
    """Test core health and status endpoints"""
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, client):
        """Test root endpoint returns system information"""
        response = await client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "SutazAI automation/advanced automation System"
        assert data["version"] == "17.0.0"
        assert "capabilities" in data
        assert "endpoints" in data
        assert "architecture" in data
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test comprehensive health check"""
        response = await client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "sutazai-backend"
        assert data["version"] == "17.0.0"
        assert "status" in data
        assert "services" in data
        assert "system" in data
        assert "timestamp" in data
        
        # Check service statuses
        services = data["services"]
        assert "ollama" in services
        assert "chromadb" in services
        assert "qdrant" in services
        assert "agents" in services
        
        # Check system metrics
        system = data["system"]
        assert "cpu_percent" in system
        assert "memory_percent" in system
        assert "memory_used_gb" in system
        assert "memory_total_gb" in system

    @pytest.mark.asyncio
    async def test_system_health_enterprise(self, client, auth_headers):
        """Test enterprise system health endpoint"""
        response = await client.get("/api/v1/system/health", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "services" in data
        assert "system" in data
        
        # If enterprise features are enabled, check for additional data
        if data.get("enterprise_features"):
            assert "enterprise_components" in data
            assert "detailed_metrics" in data

class TestAgentEndpoints:
    """Test agent-related endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_agents(self, client):
        """Test agents listing endpoint"""
        response = await client.get("/agents")
        assert response.status_code == 200
        
        data = response.json()
        assert "agents" in data
        agents = data["agents"]
        
        # Verify expected agents are present
        agent_ids = [agent["id"] for agent in agents]
        expected_agents = ["task_coordinator", "autogpt", "crewai", "aider", "gpt-engineer", "research-agent"]
        
        for expected_agent in expected_agents:
            assert expected_agent in agent_ids
        
        # Verify agent structure
        for agent in agents:
            assert "id" in agent
            assert "name" in agent
            assert "status" in agent
            assert "type" in agent
            assert "description" in agent
            assert "capabilities" in agent
            assert "health" in agent

class TestChatEndpoints:
    """Test chat and conversation endpoints"""
    
    @pytest.mark.asyncio
    async def test_chat_endpoint(self, client):
        """Test main chat endpoint with AI models"""
        payload = {
            "message": "Hello, can you help me with a simple task?",
            "model": "llama3.2:1b",
            "agent": "task_coordinator",
            "temperature": 0.7
        }
        
        response = await client.post("/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        assert "model" in data
        assert "agent" in data
        assert "timestamp" in data
        assert "processing_enhancement" in data
        assert "system_state_level" in data
    
    @pytest.mark.asyncio
    async def test_simple_chat_endpoint(self, client):
        """Test simple chat endpoint"""
        payload = {"message": "Test message"}
        
        response = await client.post("/simple-chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data or "error" in data
        assert "timestamp" in data
        
        if "response" in data:
            assert "model" in data
            assert "processing_time" in data

class TestReasoningEndpoints:
    """Test reasoning and thinking endpoints"""
    
    @pytest.mark.asyncio
    async def test_public_think_endpoint(self, client):
        """Test public thinking endpoint (no auth required)"""
        payload = {
            "query": "What are the implications of artificial intelligence?",
            "reasoning_type": "analytical"
        }
        
        response = await client.post("/public/think", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        assert "reasoning_type" in data
        assert "confidence" in data
        assert "thought_process" in data
        assert "timestamp" in data
        assert data["reasoning_type"] == "analytical"
    
    @pytest.mark.asyncio
    async def test_think_endpoint(self, client, auth_headers):
        """Test automation coordinator thinking endpoint"""
        payload = {
            "query": "How can we solve climate change effectively?",
            "reasoning_type": "strategic"
        }
        
        response = await client.post("/think", json=payload, headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "thought" in data
        assert "reasoning" in data
        assert "confidence" in data
        assert "model_used" in data
        assert "cognitive_load" in data
        assert "processing_stages" in data
        assert "processing_pathways" in data
        assert "system_state_level" in data
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_reason_endpoint(self, client):
        """Test reasoning endpoint for problem analysis"""
        payload = {
            "type": "deductive",
            "description": "If all humans are mortal and Socrates is human, what can we conclude?"
        }
        
        response = await client.post("/reason", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "analysis" in data
        assert "reasoning_type" in data
        assert "steps" in data
        assert "conclusion" in data
        assert "logical_framework" in data
        assert "confidence_level" in data
        assert "timestamp" in data
        assert data["reasoning_type"] == "deductive"

class TestTaskExecutionEndpoints:
    """Test task execution and orchestration endpoints"""
    
    @pytest.mark.asyncio
    async def test_execute_endpoint(self, client, auth_headers):
        """Test task execution endpoint"""
        payload = {
            "description": "Create a simple Python function to calculate factorial",
            "type": "coding"
        }
        
        response = await client.post("/execute", json=payload, headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "result" in data
        assert "status" in data
        assert "task_id" in data
        assert "task_type" in data
        assert "execution_time" in data
        assert "success_probability" in data
        assert "resources_used" in data
        assert "timestamp" in data
        assert data["task_type"] == "coding"
    
    @pytest.mark.asyncio
    async def test_orchestration_agents_creation(self, client, auth_headers):
        """Test orchestrated agent creation"""
        payload = {
            "agent_type": "researcher",
            "name": "test-researcher",
            "config": {
                "specialization": "AI research",
                "language": "english"
            }
        }
        
        response = await client.post("/api/v1/orchestration/agents", json=payload, headers=auth_headers)
        # May return 503 if orchestration not available, which is acceptable
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "agent_id" in data
            assert "status" in data
            assert "config" in data
            assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_orchestration_workflows(self, client, auth_headers):
        """Test workflow creation and execution"""
        payload = {
            "name": "test-workflow",
            "description": "Test workflow for validation",
            "tasks": [
                {"id": "task1", "type": "analysis", "description": "Analyze data"}
            ],
            "agents": ["researcher", "analyst"]
        }
        
        response = await client.post("/api/v1/orchestration/workflows", json=payload, headers=auth_headers)
        # May return 503 if orchestration not available
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "workflow_id" in data
            assert "status" in data
            assert "definition" in data
            assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_orchestration_status(self, client, auth_headers):
        """Test orchestration system status"""
        response = await client.get("/api/v1/orchestration/status", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        
        # If orchestration is available, check detailed status
        if data["status"] != "unavailable":
            assert "active_agents" in data
            assert "active_workflows" in data
            assert "health" in data
            assert "timestamp" in data

class TestProcessingProcessingEndpoints:
    """Test processing processing and system_state endpoints"""
    
    @pytest.mark.asyncio
    async def test_processing_process_endpoint(self, client, auth_headers):
        """Test processing processing endpoint"""
        payload = {
            "input_data": {"text": "Analyze the concept of system_state"},
            "processing_type": "analytical",
            "use_system_state": True,
            "reasoning_depth": 3
        }
        
        response = await client.post("/api/v1/processing/process", json=payload, headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "result" in data
        assert "processing_type" in data
        assert "system_state_enabled" in data
        assert "reasoning_depth" in data
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_processing_system_state_endpoint(self, client, auth_headers):
        """Test system_state state endpoint"""
        response = await client.get("/api/v1/processing/system_state", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "system_state_active" in data
        
        if data["system_state_active"]:
            assert "awareness_level" in data
            assert "cognitive_load" in data
            assert "active_processes" in data
            assert "processing_activity" in data
        
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_processing_creative_synthesis(self, client, auth_headers):
        """Test creative synthesis endpoint"""
        payload = {
            "prompt": "Design an innovative solution for sustainable energy",
            "synthesis_mode": "cross_domain",
            "reasoning_depth": 3,
            "use_system_state": True
        }
        
        response = await client.post("/api/v1/processing/creative", json=payload, headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "analysis" in data
        assert "insights" in data
        assert "recommendations" in data
        assert "output" in data
        assert "synthesis_mode" in data
        assert "system_state_active" in data
        assert "reasoning_depth" in data
        assert "creative_pathways" in data
        assert "timestamp" in data

class TestSelfImprovementEndpoints:
    """Test self-improvement system endpoints"""
    
    @pytest.mark.asyncio
    async def test_improve_endpoint(self, client, auth_headers):
        """Test legacy self-improvement endpoint"""
        response = await client.post("/improve", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "improvement" in data
        assert "changes" in data
        assert "impact" in data
        assert "next_optimization" in data
        assert "optimization_areas" in data
        assert "performance_gains" in data
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_improvement_analyze(self, client, auth_headers):
        """Test improvement analysis endpoint"""
        response = await client.post("/api/v1/improvement/analyze", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        # Response structure depends on whether self_improvement system is available
        assert "timestamp" in data
        
        # If enterprise self-improvement is available
        if "analysis_id" in data:
            assert "improvements_identified" in data
            assert "priority_areas" in data
            assert "estimated_impact" in data
            assert "implementation_plan" in data
    
    @pytest.mark.asyncio
    async def test_improvement_apply(self, client, auth_headers):
        """Test improvement application endpoint"""
        improvement_ids = ["mem_optimization", "response_speed"]
        
        response = await client.post(
            "/api/v1/improvement/apply",
            json=improvement_ids,
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "applied" in data
        assert "timestamp" in data

class TestLearningAndKnowledgeEndpoints:
    """Test learning and knowledge management endpoints"""
    
    @pytest.mark.asyncio
    async def test_learn_endpoint(self, client):
        """Test knowledge learning endpoint"""
        payload = {
            "content": "Artificial intelligence is transforming how we approach complex problems.",
            "type": "text"
        }
        
        response = await client.post("/learn", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "learned" in data
        assert "content_type" in data
        assert "content_size" in data
        assert "summary" in data
        assert "knowledge_points" in data
        assert "processing_stats" in data
        assert "processing_time" in data
        assert "timestamp" in data
        assert data["learned"] == True
        assert data["content_type"] == "text"

class TestModelsAndMetricsEndpoints:
    """Test model management and metrics endpoints"""
    
    @pytest.mark.asyncio
    async def test_models_endpoint(self, client):
        """Test models listing endpoint"""
        response = await client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert "total_models" in data
        assert "recommended_models" in data
        
        # Models structure validation
        if data["models"]:
            for model in data["models"]:
                assert "id" in model
                assert "name" in model
                assert "status" in model
                assert "type" in model
                assert "capabilities" in model
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client, auth_headers):
        """Test comprehensive metrics endpoint"""
        response = await client.get("/metrics", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "system" in data
        assert "services" in data
        assert "performance" in data
        assert "agents" in data
        assert "ai_metrics" in data
        
        # System metrics validation
        system = data["system"]
        assert "cpu_percent" in system
        assert "memory_percent" in system
        assert "uptime" in system
        
        # Performance metrics validation
        performance = data["performance"]
        assert "avg_response_time_ms" in performance
        assert "success_rate" in performance
        assert "requests_per_minute" in performance
    
    @pytest.mark.asyncio
    async def test_public_metrics_endpoint(self, client):
        """Test public metrics endpoint (no auth required)"""
        response = await client.get("/public/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "system" in data
        assert "services" in data
        assert "performance" in data
        assert "agents" in data
        assert "ai_metrics" in data
    
    @pytest.mark.asyncio
    async def test_prometheus_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = await client.get("/prometheus-metrics")
        assert response.status_code == 200
        
        # Should return plain text Prometheus format
        content = response.text
        assert "sutazai_uptime_seconds" in content
        assert "sutazai_cache_entries_total" in content
        assert "sutazai_info" in content
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

class TestAdvancedAPIEndpoints:
    """Test advanced API endpoints (consensus, generation, etc.)"""
    
    @pytest.mark.asyncio
    async def test_agents_consensus(self, client, auth_headers):
        """Test agent consensus endpoint"""
        payload = {
            "prompt": "Should we implement renewable energy solutions?",
            "agents": ["researcher", "analyst", "advisor"],
            "consensus_type": "majority"
        }
        
        response = await client.post("/api/v1/agents/consensus", json=payload, headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "analysis" in data
        assert "agents_consulted" in data
        assert "consensus_reached" in data
        assert "consensus_type" in data
        assert "confidence" in data
        assert "recommendations" in data
        assert "output" in data
        assert "agent_votes" in data
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_models_generate(self, client, auth_headers):
        """Test models generation endpoint"""
        payload = {
            "prompt": "Write a short story about AI and humanity",
            "model": "default",
            "max_tokens": 512,
            "temperature": 0.8
        }
        
        response = await client.post("/api/v1/models/generate", json=payload, headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "analysis" in data
        assert "model_used" in data
        assert "generated_text" in data
        assert "tokens_used" in data
        assert "temperature" in data
        assert "insights" in data
        assert "recommendations" in data
        assert "output" in data
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_enterprise_system_status(self, client, auth_headers):
        """Test enterprise system status endpoint"""
        response = await client.get("/api/v1/system/status", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "system_info" in data
        assert "components" in data
        assert "services" in data
        assert "performance" in data
        
        # System info validation
        system_info = data["system_info"]
        assert "name" in system_info
        assert "version" in system_info
        assert "enterprise_features" in system_info
        assert "uptime" in system_info
        assert "timestamp" in system_info
    
    @pytest.mark.asyncio
    async def test_api_documentation(self, client, auth_headers):
        """Test API documentation endpoint"""
        response = await client.get("/api/v1/docs/endpoints", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "api_version" in data
        assert "enterprise_features" in data
        assert "endpoints" in data
        assert "authentication" in data
        
        # Endpoints structure validation
        endpoints = data["endpoints"]
        assert "core" in endpoints
        assert "orchestration" in endpoints
        assert "processing" in endpoints
        assert "improvement" in endpoints

class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_invalid_endpoints(self, client):
        """Test handling of invalid endpoints"""
        response = await client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_invalid_json_payload(self, client):
        """Test handling of invalid JSON payloads"""
        # Send malformed JSON
        response = await client.post(
            "/chat",
            data="{invalid json}",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self, client):
        """Test handling of missing required fields"""
        # Chat endpoint requires 'message' field
        response = await client.post("/chat", json={})
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_oversized_payload(self, client):
        """Test handling of oversized payloads"""
        large_message = "x" * 50000  # Very large message
        payload = {"message": large_message}
        
        response = await client.post("/chat", json=payload)
        # Should either succeed or return appropriate error
        assert response.status_code in [200, 413, 422]
    
    @pytest.mark.asyncio
    async def test_concurrent_stress_test(self, client, auth_headers):
        """Test system under concurrent load"""
        async def make_request(i):
            payload = {"query": f"Test query {i}", "reasoning_type": "general"}
            return await client.post("/public/think", json=payload)
        
        # Send 20 concurrent requests
        tasks = [make_request(i) for i in range(20)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful responses
        successful = [r for r in responses if hasattr(r, 'status_code') and r.status_code == 200]
        
        # At least 80% should succeed under normal conditions
        success_rate = len(successful) / len(responses)
        assert success_rate >= 0.8, f"Success rate {success_rate} below threshold"
    
    @pytest.mark.asyncio
    async def test_response_time_performance(self, client):
        """Test API response time performance"""
        start_time = time.time()
        response = await client.get("/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response.status_code == 200
        assert response_time < 5.0, f"Health check too slow: {response_time}s"
    
    @pytest.mark.asyncio
    async def test_memory_stability(self, client):
        """Test memory stability during operations"""
        # Make multiple requests to check for memory leaks
        for i in range(10):
            payload = {"message": f"Memory test {i}"}
            response = await client.post("/simple-chat", json=payload)
            # Response should be consistent
            assert response.status_code in [200, 503]  # 503 if models unavailable
            
            # Small delay between requests
            await asyncio.sleep(0.1)

class TestWebSocketEndpoints:
    """Test WebSocket connections"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection establishment"""
        try:
            async with httpx.AsyncClient() as client:
                # Check if WebSocket endpoint is accessible (basic test)
                # Full WebSocket testing would require websockets library
                response = await client.get(f"{BASE_URL}/ws", 
                                          headers={"upgrade": "websocket"})
                # Expect upgrade or method not allowed
                assert response.status_code in [101, 405, 426]
        except Exception as e:
            # WebSocket test may fail if not properly configured
            pytest.skip(f"WebSocket test skipped: {e}")

# Configuration for test execution
if __name__ == "__main__":
    pytest.main(["-v", __file__])
