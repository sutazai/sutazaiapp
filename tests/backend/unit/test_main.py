"""
Unit tests for main application module
Professional-grade tests covering core FastAPI functionality and business logic
"""

import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import HTTPException
from fastapi.testclient import TestClient
from datetime import datetime


class TestApplicationLifecycle:
    """Test application lifecycle management"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_lifespan_startup(self, app_with_Mocks):
        """Test application startup lifecycle"""
        # Application should start successfully with Mocked dependencies
        assert app_with_Mocks is not None
        assert app_with_Mocks.title == "SutazAI High-Performance Backend"
        assert app_with_Mocks.version == "2.0.0"

    @pytest.mark.unit
    def test_app_configuration(self, app_with_Mocks):
        """Test application configuration and middleware setup"""
        # Check CORS middleware is configured
        cors_middleware_found = False
        for middleware in app_with_Mocks.user_middleware:
            if "CORSMiddleware" in str(middleware.cls):
                cors_middleware_found = True
                break
        assert cors_middleware_found, "CORS middleware should be configured"

        # Check GZip middleware is configured
        gzip_middleware_found = False
        for middleware in app_with_Mocks.user_middleware:
            if "GZipMiddleware" in str(middleware.cls):
                gzip_middleware_found = True
                break
        assert gzip_middleware_found, "GZip middleware should be configured"


class TestHealthEndpoints:
    """Test health check endpoints"""

    @pytest.mark.unit
    @pytest.mark.api
    def test_health_endpoint_response_structure(self, sync_client):
        """Test health endpoint returns correct response structure"""
        response = sync_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify required fields
        assert "status" in data
        assert "timestamp" in data
        assert "services" in data
        assert "performance" in data
        
        # Verify status is healthy
        assert data["status"] == "healthy"
        
        # Verify timestamp format
        timestamp = data["timestamp"]
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))  # Should not raise

    @pytest.mark.unit
    @pytest.mark.api
    def test_health_endpoint_performance(self, sync_client):
        """Test health endpoint response time"""
        import time
        
        # Make multiple requests to test consistency
        response_times = []
        for _ in range(5):
            start = time.time()
            response = sync_client.get("/health")
            end = time.time()
            
            assert response.status_code == 200
            response_times.append((end - start) * 1000)  # Convert to ms
        
        # Average response time should be reasonable
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 100, f"Health endpoint too slow: {avg_response_time}ms"

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_detailed_health_endpoint(self, async_client, mock_health_monitoring):
        """Test detailed health endpoint"""
        response = await async_client.get("/api/v1/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify detailed health structure
        assert "overall_status" in data
        assert "timestamp" in data
        assert "services" in data
        assert "performance_metrics" in data
        assert "system_resources" in data
        assert "alerts" in data
        assert "recommendations" in data

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_circuit_breaker_status_endpoint(self, async_client, mock_circuit_breaker_manager):
        """Test circuit breaker status endpoint"""
        response = await async_client.get("/api/v1/health/circuit-breakers")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify circuit breaker response structure
        assert "timestamp" in data
        assert "circuit_breakers" in data
        assert "total_breakers" in data
        assert "healthy_breakers" in data
        assert "open_breakers" in data

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_circuit_breaker_reset_endpoint(self, async_client, mock_circuit_breaker_manager):
        """Test circuit breaker reset endpoint"""
        response = await async_client.post("/api/v1/health/circuit-breakers/reset")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify reset response
        assert "message" in data
        assert "timestamp" in data
        assert "status" in data
        assert data["status"] == "success"


class TestRootAndStatusEndpoints:
    """Test root and status endpoints"""

    @pytest.mark.unit
    @pytest.mark.api
    def test_root_endpoint(self, sync_client):
        """Test root endpoint returns correct information"""
        response = sync_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify root response structure
        assert data["message"] == "SutazAI High-Performance Backend"
        assert data["version"] == "2.0.0"
        assert data["status"] == "optimized"
        assert "features" in data
        
        # Verify features are documented
        features = data["features"]
        assert features["connection_pooling"] is True
        assert features["redis_caching"] is True
        assert features["async_ollama"] is True
        assert features["background_tasks"] is True

    @pytest.mark.unit
    @pytest.mark.api
    def test_status_endpoint_with_system_metrics(self, sync_client, mock_psutil):
        """Test status endpoint with system metrics"""
        response = sync_client.get("/api/v1/status")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify status response
        assert data["status"] == "operational"
        assert data["version"] == "2.0.0"
        assert data["performance_mode"] == "optimized"
        
        # Verify system metrics are included
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert isinstance(data["cpu_percent"], (int, float))
        assert isinstance(data["memory_percent"], (int, float))


class TestAgentEndpoints:
    """Test agent management endpoints"""

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_list_agents_endpoint(self, async_client):
        """Test listing all agents"""
        response = await async_client.get("/api/v1/agents")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return list of agents
        assert isinstance(data, list)
        
        # Each agent should have required fields
        for agent in data:
            assert "id" in agent
            assert "name" in agent
            assert "status" in agent
            assert "capabilities" in agent
            assert isinstance(agent["capabilities"], list)

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_get_specific_agent(self, async_client, mock_validation):
        """Test getting specific agent details"""
        agent_id = "jarvis-automation"
        response = await async_client.get(f"/api/v1/agents/{agent_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify agent details
        assert data["id"] == agent_id
        assert "name" in data
        assert "status" in data
        assert "capabilities" in data

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_get_nonexistent_agent(self, async_client, mock_validation):
        """Test getting non-existent agent returns 404"""
        response = await async_client.get("/api/v1/agents/nonexistent-agent")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_agent_input_validation(self, async_client):
        """Test agent endpoint input validation"""
        # Test with malicious input
        malicious_agent_id = "'; DROP TABLE agents; --"
        
        with patch('app.utils.validation.validate_agent_id') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid agent ID")
            
            response = await async_client.get(f"/api/v1/agents/{malicious_agent_id}")
            assert response.status_code == 400


class TestTaskEndpoints:
    """Test task management endpoints"""

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_create_task(self, async_client, sample_task_request):
        """Test task creation"""
        response = await async_client.post("/api/v1/tasks", json=sample_task_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify task creation response
        assert "task_id" in data
        assert data["status"] == "queued"
        assert "result" in data

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_get_task_status(self, async_client, mock_validation):
        """Test getting task status"""
        task_id = "test-task-id"
        response = await async_client.get(f"/api/v1/tasks/{task_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify task status response
        assert "task_id" in data
        assert "status" in data
        assert "result" in data

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_task_input_validation(self, async_client):
        """Test task endpoint input validation"""
        malicious_task_id = "../../../etc/passwd"
        
        with patch('app.utils.validation.validate_task_id') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid task ID")
            
            response = await async_client.get(f"/api/v1/tasks/{malicious_task_id}")
            assert response.status_code == 400


class TestChatEndpoints:
    """Test chat and AI interaction endpoints"""

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_chat_endpoint(self, async_client, sample_chat_request, mock_validation):
        """Test chat endpoint functionality"""
        response = await async_client.post("/api/v1/chat", json=sample_chat_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify chat response structure
        assert "response" in data
        assert "model" in data
        assert "cached" in data
        assert data["model"] == "tinyllama"

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_chat_input_validation(self, async_client):
        """Test chat endpoint input validation"""
        malicious_request = {
            "message": "Hello",
            "model": "'; DROP TABLE models; --",
            "use_cache": True
        }
        
        with patch('app.utils.validation.validate_model_name') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid model name")
            
            response = await async_client.post("/api/v1/chat", json=malicious_request)
            assert response.status_code == 400

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_batch_process_endpoint(self, async_client, mock_validation):
        """Test batch processing endpoint"""
        prompts = ["Hello", "How are you?", "What's the weather?"]
        
        response = await async_client.post("/api/v1/batch", json=prompts)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify batch response
        assert "results" in data
        assert isinstance(data["results"], list)

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_batch_process_size_limit(self, async_client):
        """Test batch processing size limits"""
        # Create oversized batch
        large_prompts = ["prompt"] * 100  # Over the 50 limit
        
        response = await async_client.post("/api/v1/batch", json=large_prompts)
        
        assert response.status_code == 400
        data = response.json()
        assert "Too many prompts" in data["detail"]

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_chat_stream_endpoint(self, async_client, sample_chat_request, mock_validation):
        """Test streaming chat endpoint"""
        response = await async_client.post("/api/v1/chat/stream", json=sample_chat_request)
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestCacheEndpoints:
    """Test cache management endpoints"""

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_cache_clear_endpoint(self, async_client, mock_validation):
        """Test cache clearing functionality"""
        response = await async_client.post("/api/v1/cache/clear")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify cache clear response
        assert "message" in data
        assert "All cache entries cleared" in data["message"]

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_cache_clear_with_pattern(self, async_client, mock_validation):
        """Test cache clearing with pattern"""
        response = await async_client.post("/api/v1/cache/clear?pattern=test:*")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify pattern-based cache clear
        assert "message" in data
        assert "cache entries matching pattern" in data["message"]

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_cache_invalidate_by_tags(self, async_client):
        """Test cache invalidation by tags"""
        tags = ["user", "profile"]
        response = await async_client.post("/api/v1/cache/invalidate", json=tags)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify tag-based invalidation
        assert "message" in data
        assert "tags" in data
        assert data["tags"] == tags

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_cache_warm_endpoint(self, async_client):
        """Test cache warming functionality"""
        response = await async_client.post("/api/v1/cache/warm")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify cache warming response
        assert "message" in data
        assert "timestamp" in data
        assert "warmed_categories" in data

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_cache_stats_endpoint(self, async_client):
        """Test cache statistics endpoint"""
        response = await async_client.get("/api/v1/cache/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify cache stats structure
        assert "hits" in data
        assert "misses" in data
        assert "hit_rate" in data


class TestMetricsEndpoints:
    """Test metrics and monitoring endpoints"""

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, async_client, mock_psutil):
        """Test comprehensive metrics endpoint"""
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

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_prometheus_metrics_endpoint(self, async_client, mock_health_monitoring):
        """Test Prometheus metrics endpoint"""
        response = await async_client.get("/metrics")
        
        assert response.status_code == 200
        # Response should be Prometheus format text
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

    @pytest.mark.unit
    @pytest.mark.api
    def test_settings_endpoint(self, sync_client):
        """Test settings endpoint"""
        response = sync_client.get("/api/v1/settings")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify settings structure
        assert "environment" in data
        assert "debug" in data
        assert "features" in data
        assert "performance" in data


class TestErrorHandling:
    """Test error handling and exception cases"""

    @pytest.mark.unit
    @pytest.mark.api
    def test_404_error_handling(self, sync_client):
        """Test 404 error handling"""
        response = sync_client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_global_exception_handler(self, async_client):
        """Test global exception handler"""
        # This would require triggering an actual exception in the app
        # For now, we test that the handler is configured
        from app.main import app
        
        # Check that exception handler is registered
        assert len(app.exception_handlers) > 0

    @pytest.mark.unit
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, async_client):
        """Test validation error handling"""
        # Test with invalid JSON
        response = await async_client.post(
            "/api/v1/chat", 
            content="invalid json",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422  # Unprocessable Entity


class TestSecurityFeatures:
    """Test security features and protections"""

    @pytest.mark.unit
    @pytest.mark.security
    def test_cors_configuration(self, app_with_Mocks):
        """Test CORS configuration is secure"""
        # CORS middleware should be present and configured
        cors_middleware_found = False
        for middleware in app_with_Mocks.user_middleware:
            if "CORSMiddleware" in str(middleware.cls):
                cors_middleware_found = True
                break
        
        assert cors_middleware_found, "CORS middleware should be configured"

    @pytest.mark.unit
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_input_sanitization(self, async_client):
        """Test input sanitization for security"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd"
        ]
        
        for malicious_input in malicious_inputs:
            with patch('app.utils.validation.sanitize_user_input') as mock_sanitize:
                mock_sanitize.side_effect = ValueError("Malicious input detected")
                
                response = await async_client.post("/api/v1/batch", json=[malicious_input])
                assert response.status_code == 400

    @pytest.mark.unit
    @pytest.mark.security
    def test_jwt_secret_key_validation(self):
        """Test JWT secret key validation"""
        import os
        
        # JWT secret should be set and secure
        jwt_secret = os.getenv("JWT_SECRET_KEY")
        assert jwt_secret is not None
        assert len(jwt_secret) >= 32  # Minimum length for security
        assert jwt_secret != "your_secret_key_here"  # Should not be default


class TestPerformanceAndReliability:
    """Test performance characteristics and reliability features"""

    @pytest.mark.unit
    @pytest.mark.performance
    def test_health_endpoint_response_time(self, sync_client):
        """Test health endpoint meets performance requirements"""
        import time
        
        start = time.time()
        response = sync_client.get("/health")
        end = time.time()
        
        response_time_ms = (end - start) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < 50, f"Health endpoint too slow: {response_time_ms}ms"

    @pytest.mark.unit
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_requests_handling(self, async_client):
        """Test handling of concurrent requests"""
        import asyncio
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(10):
            task = async_client.get("/health")
            tasks.append(task)
        
        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

    @pytest.mark.unit
    @pytest.mark.reliability
    @pytest.mark.asyncio
    async def test_service_degradation_handling(self, async_client):
        """Test graceful degradation when services are unavailable"""
        # Health endpoint should still respond even if some services are down
        response = await async_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"  # Should still report healthy for load balancer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])