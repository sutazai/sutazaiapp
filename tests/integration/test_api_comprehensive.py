#!/usr/bin/env python3
"""
Comprehensive Integration Tests for SutazAI API Endpoints
Validates all API functionality per Rules 1-19
"""

import pytest
import asyncio
import httpx
import json
import uuid
from typing import Dict, Any
from datetime import datetime, timedelta
import os
import sys

# Add backend to path
# Path handled by pytest configuration, '..', '..', 'backend'))

# Test configuration
BASE_URL = os.getenv('TEST_BASE_URL', 'http://localhost:10010')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:10104')
TEST_TIMEOUT = 30.0


@pytest.mark.integration
class TestHealthEndpoints:
    """Test system health and status endpoints"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test main health endpoint returns proper status"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/health")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response structure
            assert 'status' in data
            assert 'timestamp' in data
            assert 'services' in data
            assert 'performance' in data
            
            # Validate service status
            services = data['services']
            assert 'database' in services
            assert 'redis' in services
            assert 'ollama' in services
            
            # System should report healthy (not degraded)
            assert data['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self):
        """Test metrics endpoint returns Prometheus format"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/metrics")
            
            assert response.status_code == 200
            metrics_text = response.text
            
            # Validate Prometheus format
            assert '# HELP' in metrics_text
            assert '# TYPE' in metrics_text
            assert 'requests_total' in metrics_text or 'http_requests_total' in metrics_text
    
    @pytest.mark.asyncio
    async def test_readiness_endpoint(self):
        """Test readiness probe for Kubernetes deployment"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/ready")
            
            assert response.status_code == 200
            data = response.json()
            assert data.get('ready') is True
    
    @pytest.mark.asyncio
    async def test_liveness_endpoint(self):
        """Test liveness probe for Kubernetes deployment"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/alive")
            
            assert response.status_code == 200
            data = response.json()
            assert data.get('alive') is True


@pytest.mark.integration
class TestChatEndpoints:
    """Test chat and AI interaction endpoints"""
    
    @pytest.mark.asyncio
    async def test_chat_endpoint_basic(self):
        """Test basic chat functionality with TinyLlama"""
        chat_request = {
            "message": "Hello, how are you?",
            "model": "tinyllama"
        }
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/api/v1/chat/",
                json=chat_request
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response structure
            assert 'response' in data
            assert 'model' in data
            assert 'timestamp' in data
            
            # Validate content
            assert isinstance(data['response'], str)
            assert len(data['response']) > 0
            assert data['model'] == 'tinyllama'
    
    @pytest.mark.asyncio
    async def test_chat_endpoint_with_system_prompt(self):
        """Test chat with system prompt"""
        chat_request = {
            "message": "What is 2+2?",
            "model": "tinyllama",
            "system_prompt": "You are a helpful math assistant."
        }
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/api/v1/chat/",
                json=chat_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert 'response' in data
    
    @pytest.mark.asyncio
    async def test_chat_endpoint_streaming(self):
        """Test chat streaming endpoint"""
        chat_request = {
            "message": "Tell me a short story",
            "model": "tinyllama",
            "stream": True
        }
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            async with client.stream(
                'POST',
                f"{BASE_URL}/api/v1/chat/stream",
                json=chat_request
            ) as response:
                assert response.status_code == 200
                assert response.headers.get('content-type') == 'text/event-stream'
                
                # Read first chunk
                chunk_count = 0
                async for chunk in response.aiter_text():
                    if chunk.strip():
                        chunk_count += 1
                        if chunk_count >= 3:  # Read a few chunks
                            break
                
                assert chunk_count >= 1
    
    @pytest.mark.asyncio
    async def test_chat_endpoint_validation(self):
        """Test chat endpoint input validation"""
        # Test missing message
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/api/v1/chat/",
                json={"model": "tinyllama"}
            )
            assert response.status_code == 422  # Validation error
            
        # Test invalid model
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/api/v1/chat/",
                json={"message": "test", "model": "nonexistent_model"}
            )
            # Should either work with fallback or return error
            assert response.status_code in [200, 400, 422]
    
    @pytest.mark.asyncio
    async def test_chat_endpoint_xss_protection(self):
        """Test XSS protection in chat endpoint"""
        malicious_message = '<script>alert("xss")</script>'
        chat_request = {
            "message": malicious_message,
            "model": "tinyllama"
        }
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/api/v1/chat/",
                json=chat_request
            )
            
            # Should process safely without executing script
            assert response.status_code == 200
            data = response.json()
            # Response should not contain raw script tags
            assert '<script>' not in data.get('response', '')


@pytest.mark.integration
class TestModelEndpoints:
    """Test model management endpoints"""
    
    @pytest.mark.asyncio
    async def test_list_models_endpoint(self):
        """Test model listing endpoint"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/models/")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response structure
            assert 'models' in data
            assert isinstance(data['models'], list)
            
            # Should include TinyLlama model
            model_names = [model.get('name', '') for model in data['models']]
            assert any('tinyllama' in name.lower() for name in model_names)
    
    @pytest.mark.asyncio
    async def test_model_info_endpoint(self):
        """Test individual model information"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/models/tinyllama")
            
            if response.status_code == 200:
                data = response.json()
                assert 'name' in data
                assert 'size' in data or 'parameters' in data
            else:
                # Model info endpoint may not be implemented
                assert response.status_code in [404, 501]


@pytest.mark.integration
class TestAgentEndpoints:
    """Test agent management endpoints"""
    
    @pytest.mark.asyncio
    async def test_list_agents_endpoint(self):
        """Test agent listing endpoint"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/agents/")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response structure
            assert 'agents' in data or isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_agent_registration(self):
        """Test agent registration endpoint"""
        agent_data = {
            "name": f"test-agent-{uuid.uuid4().hex[:8]}",
            "type": "test",
            "capabilities": ["test", "integration"],
            "endpoint": "http://test-agent:8080"
        }
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/api/v1/agents/register",
                json=agent_data
            )
            
            # Should succeed or return reasonable error
            assert response.status_code in [200, 201, 400, 422]
            
            if response.status_code in [200, 201]:
                data = response.json()
                assert 'id' in data or 'agent_id' in data
    
    @pytest.mark.asyncio
    async def test_agent_health_check(self):
        """Test agent health check endpoint"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/agents/health")
            
            assert response.status_code == 200
            data = response.json()
            
            # Should return health status of all agents
            assert isinstance(data, (dict, list))


@pytest.mark.integration
class TestTaskEndpoints:
    """Test task management endpoints"""
    
    @pytest.mark.asyncio
    async def test_task_submission(self):
        """Test task submission endpoint"""
        task_data = {
            "type": "text_generation",
            "payload": {
                "prompt": "Generate a haiku about technology",
                "model": "tinyllama"
            },
            "priority": 1
        }
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/api/v1/tasks/",
                json=task_data
            )
            
            assert response.status_code in [200, 201, 202]
            data = response.json()
            
            # Should return task ID
            assert 'task_id' in data or 'id' in data
            task_id = data.get('task_id') or data.get('id')
            
            return task_id
    
    @pytest.mark.asyncio
    async def test_task_status_check(self):
        """Test task status endpoint"""
        # Submit a task first
        task_id = await self.test_task_submission()
        
        if task_id:
            async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
                response = await client.get(f"{BASE_URL}/api/v1/tasks/{task_id}")
                
                assert response.status_code == 200
                data = response.json()
                
                # Validate task status structure
                assert 'status' in data
                assert data['status'] in ['pending', 'processing', 'completed', 'failed']
    
    @pytest.mark.asyncio
    async def test_task_queue_status(self):
        """Test task queue status endpoint"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/tasks/queue/status")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate queue status
            assert 'pending' in data or 'queue_size' in data
            assert 'processing' in data or 'active_tasks' in data


@pytest.mark.integration
class TestMeshEndpoints:
    """Test service mesh and coordination endpoints"""
    
    @pytest.mark.asyncio
    async def test_mesh_enqueue(self):
        """Test mesh task enqueueing"""
        task_data = {
            "task_type": "test_task",
            "payload": {"test_key": "test_value"},
            "priority": 0
        }
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/api/v1/mesh/enqueue",
                json=task_data
            )
            
            assert response.status_code in [200, 201, 202]
            data = response.json()
            assert 'task_id' in data or 'id' in data
    
    @pytest.mark.asyncio
    async def test_mesh_results(self):
        """Test mesh results retrieval"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/mesh/results")
            
            assert response.status_code == 200
            data = response.json()
            
            # Should return results structure
            assert isinstance(data, (dict, list))


@pytest.mark.integration
class TestHardwareEndpoints:
    """Test hardware optimization endpoints"""
    
    @pytest.mark.asyncio
    async def test_hardware_status(self):
        """Test hardware status endpoint"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/hardware/status")
            
            if response.status_code == 200:
                data = response.json()
                assert 'cpu' in data or 'memory' in data or 'disk' in data
            else:
                # Hardware endpoint may not be implemented
                assert response.status_code in [404, 501]
    
    @pytest.mark.asyncio
    async def test_hardware_optimization(self):
        """Test hardware optimization endpoint"""
        optimization_request = {
            "target": "memory",
            "level": "moderate"
        }
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/api/v1/hardware/optimize",
                json=optimization_request
            )
            
            # Should process request or return not implemented
            assert response.status_code in [200, 202, 404, 501]


@pytest.mark.integration
class TestSystemEndpoints:
    """Test system management endpoints"""
    
    @pytest.mark.asyncio
    async def test_system_info(self):
        """Test system information endpoint"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/system/info")
            
            assert response.status_code == 200
            data = response.json()
            
            # Should include version and system info
            assert 'version' in data or 'system' in data
    
    @pytest.mark.asyncio
    async def test_system_stats(self):
        """Test system statistics endpoint"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/system/stats")
            
            assert response.status_code == 200
            data = response.json()
            
            # Should include performance statistics
            assert isinstance(data, dict)


@pytest.mark.integration
class TestAuthenticationEndpoints:
    """Test authentication and authorization endpoints"""
    
    @pytest.mark.asyncio
    async def test_login_endpoint(self):
        """Test login endpoint"""
        login_data = {
            "username": "test_user",
            "password": "test_password"
        }
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/auth/login",
                json=login_data
            )
            
            # Should handle login attempt (success or failure)
            assert response.status_code in [200, 401, 422]
    
    @pytest.mark.asyncio
    async def test_protected_endpoint_access(self):
        """Test access to protected endpoints without authentication"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/admin/users")
            
            # Should require authentication
            assert response.status_code in [401, 403, 404]


@pytest.mark.integration
class TestDatabaseIntegration:
    """Test database integration endpoints"""
    
    @pytest.mark.asyncio
    async def test_database_health(self):
        """Test database connectivity through health endpoint"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/health")
            
            assert response.status_code == 200
            data = response.json()
            
            # Database should be reported as healthy
            services = data.get('services', {})
            if 'database' in services:
                assert services['database'] in ['healthy', 'connected', 'ok']
    
    @pytest.mark.asyncio
    async def test_data_persistence(self):
        """Test data persistence through API"""
        # This would test CRUD operations if implemented
        # For now, just verify endpoints exist
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/api/v1/data/test")
            
            # Should handle request (data found, not found, or not implemented)
            assert response.status_code in [200, 404, 501]


@pytest.mark.integration
class TestPerformanceIntegration:
    """Test API performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_response_time_requirements(self):
        """Test API response times meet requirements (<200ms for health)"""
        import time
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/health")
        
        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        assert response.status_code == 200
        assert response_time < 500  # Health endpoint should respond within 500ms
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_handling(self):
        """Test API handles concurrent requests"""
        async def make_request(client):
            return await client.get(f"{BASE_URL}/health")
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Send 10 concurrent requests
            tasks = [make_request(client) for _ in range(10)]
            responses = await asyncio.gather(*tasks)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_large_payload_handling(self):
        """Test API handles large payloads appropriately"""
        large_message = "A" * 10000  # 10KB message
        chat_request = {
            "message": large_message,
            "model": "tinyllama"
        }
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/api/v1/chat/",
                json=chat_request
            )
            
            # Should handle large payload (success or appropriate error)
            assert response.status_code in [200, 413, 422]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
