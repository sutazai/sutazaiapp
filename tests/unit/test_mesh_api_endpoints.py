"""
Unit tests for mesh API endpoints.
Tests all endpoints in backend/app/api/v1/endpoints/mesh.py with Mocked dependencies.
"""
import json
import pytest
from unittest.Mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
from typing import Dict, Any, List

# Test fixtures
@pytest.fixture
def sample_enqueue_request():
    """Sample enqueue request data."""
    return {
        "topic": "data_processing",
        "task": {
            "task_id": "test-task-123",
            "task_type": "data_analysis",
            "input_data": "sample data for processing",
            "parameters": {
                "algorithm": "nlp_analysis",
                "confidence_threshold": 0.8
            }
        }
    }

@pytest.fixture
def sample_results_data():
    """Sample results data."""
    return [
        {
            "id": "1699999999999-0",
            "data": {
                "task_id": "test-task-123",
                "status": "completed",
                "result": "analysis completed successfully",
                "confidence": 0.95,
                "processing_time": 2.3
            }
        },
        {
            "id": "1699999999999-1", 
            "data": {
                "task_id": "test-task-124",
                "status": "completed",
                "result": "analysis completed successfully",
                "confidence": 0.87,
                "processing_time": 1.8
            }
        }
    ]

@pytest.fixture
def sample_agents_data():
    """Sample agents data."""
    return {
        "count": 2,
        "agents": [
            {
                "agent_id": "data-processor-1",
                "agent_type": "data_processor",
                "meta": {
                    "version": "1.0.0",
                    "capabilities": ["text_analysis", "data_transform"],
                    "max_concurrent_tasks": 5,
                    "status": "active"
                }
            },
            {
                "agent_id": "ai-inference-1",
                "agent_type": "ai_inference",
                "meta": {
                    "version": "1.2.0",
                    "capabilities": ["llm_inference", "embedding_generation"],
                    "max_concurrent_tasks": 3,
                    "status": "active"
                }
            }
        ]
    }

@pytest.fixture
def Mock_redis():
    """Mock Redis client."""
    redis_Mock = Mock()
    redis_Mock.ping.return_value = True
    return redis_Mock

class TestEnqueueEndpoint:
    """Test the /enqueue endpoint."""
    
    @patch('backend.app.api.v1.endpoints.mesh.enqueue_task')
    def test_enqueue_success(self, Mock_enqueue_task, sample_enqueue_request):
        """Test successful task enqueuing."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        Mock_enqueue_task.return_value = "1699999999999-0"
        
        response = client.post("/mesh/enqueue", json=sample_enqueue_request)
        
        assert response.status_code == 200
        assert response.json() == {"id": "1699999999999-0"}
        
        Mock_enqueue_task.assert_called_once_with(
            sample_enqueue_request["topic"],
            sample_enqueue_request["task"]
        )
    
    @patch('backend.app.api.v1.endpoints.mesh.enqueue_task')
    def test_enqueue_redis_error(self, Mock_enqueue_task, sample_enqueue_request):
        """Test enqueue with Redis connection error."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        Mock_enqueue_task.side_effect = Exception("Redis connection failed")
        
        response = client.post("/mesh/enqueue", json=sample_enqueue_request)
        
        assert response.status_code == 500
        assert "Redis connection failed" in response.json()["detail"]
    
    def test_enqueue_invalid_topic_empty(self):
        """Test enqueue with empty topic."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        invalid_request = {
            "topic": "",
            "task": {"key": "value"}
        }
        
        response = client.post("/mesh/enqueue", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    def test_enqueue_invalid_topic_pattern(self):
        """Test enqueue with invalid topic pattern."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        invalid_request = {
            "topic": "invalid topic with spaces!",
            "task": {"key": "value"}
        }
        
        response = client.post("/mesh/enqueue", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    def test_enqueue_topic_too_long(self):
        """Test enqueue with topic exceeding max length."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        invalid_request = {
            "topic": "a" * 65,  # Max length is 64
            "task": {"key": "value"}
        }
        
        response = client.post("/mesh/enqueue", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    def test_enqueue_missing_task(self):
        """Test enqueue with missing task field."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        invalid_request = {
            "topic": "data_processing"
            # Missing task field
        }
        
        response = client.post("/mesh/enqueue", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    def test_enqueue_valid_topic_patterns(self):
        """Test enqueue with various valid topic patterns."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        valid_topics = [
            "data-processing",
            "ai_inference", 
            "nlp:analysis",
            "task_queue_v2",
            "model-training-gpu",
            "data123"
        ]
        
        with patch('backend.app.api.v1.endpoints.mesh.enqueue_task') as Mock_enqueue:
            Mock_enqueue.return_value = "test-id"
            
            for topic in valid_topics:
                request = {"topic": topic, "task": {"test": "data"}}
                response = client.post("/mesh/enqueue", json=request)
                assert response.status_code == 200

class TestResultsEndpoint:
    """Test the /results endpoint."""
    
    @patch('backend.app.api.v1.endpoints.mesh.tail_results')
    def test_get_results_success(self, Mock_tail_results, sample_results_data):
        """Test successful results retrieval."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        # Mock tail_results to return tuples (id, data)
        Mock_data = [
            ("1699999999999-0", sample_results_data[0]["data"]),
            ("1699999999999-1", sample_results_data[1]["data"])
        ]
        Mock_tail_results.return_value = Mock_data
        
        response = client.get("/mesh/results?topic=data_processing&count=2")
        
        assert response.status_code == 200
        assert response.json() == sample_results_data
        
        Mock_tail_results.assert_called_once_with("data_processing", 2)
    
    @patch('backend.app.api.v1.endpoints.mesh.tail_results')
    def test_get_results_empty(self, Mock_tail_results):
        """Test results retrieval with no results."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        Mock_tail_results.return_value = []
        
        response = client.get("/mesh/results?topic=data_processing")
        
        assert response.status_code == 200
        assert response.json() == []
        
        Mock_tail_results.assert_called_once_with("data_processing", 10)  # Default count
    
    @patch('backend.app.api.v1.endpoints.mesh.tail_results')
    def test_get_results_redis_error(self, Mock_tail_results):
        """Test results retrieval with Redis error."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        Mock_tail_results.side_effect = Exception("Redis connection failed")
        
        response = client.get("/mesh/results?topic=data_processing")
        
        assert response.status_code == 500
        assert "Redis connection failed" in response.json()["detail"]
    
    def test_get_results_missing_topic(self):
        """Test results retrieval without topic parameter."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        response = client.get("/mesh/results")
        
        assert response.status_code == 422  # Missing required parameter
    
    def test_get_results_invalid_topic_pattern(self):
        """Test results retrieval with invalid topic pattern."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        response = client.get("/mesh/results?topic=invalid topic!")
        
        assert response.status_code == 422  # Validation error
    
    def test_get_results_count_validation(self):
        """Test results retrieval with count parameter validation."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        # Test count too low
        response = client.get("/mesh/results?topic=data_processing&count=0")
        assert response.status_code == 422
        
        # Test count too high
        response = client.get("/mesh/results?topic=data_processing&count=101")
        assert response.status_code == 422
        
        # Test valid count
        with patch('backend.app.api.v1.endpoints.mesh.tail_results') as Mock_tail:
            Mock_tail.return_value = []
            response = client.get("/mesh/results?topic=data_processing&count=50")
            assert response.status_code == 200
    
    def test_get_results_default_count(self):
        """Test results retrieval uses default count when not specified."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        with patch('backend.app.api.v1.endpoints.mesh.tail_results') as Mock_tail:
            Mock_tail.return_value = []
            response = client.get("/mesh/results?topic=data_processing")
            
            assert response.status_code == 200
            Mock_tail.assert_called_once_with("data_processing", 10)  # Default count

class TestAgentsEndpoint:
    """Test the /agents endpoint."""
    
    @patch('backend.app.api.v1.endpoints.mesh.list_agents')
    def test_get_agents_success(self, Mock_list_agents, sample_agents_data):
        """Test successful agents listing."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        Mock_list_agents.return_value = sample_agents_data["agents"]
        
        response = client.get("/mesh/agents")
        
        assert response.status_code == 200
        assert response.json() == sample_agents_data
        
        Mock_list_agents.assert_called_once()
    
    @patch('backend.app.api.v1.endpoints.mesh.list_agents')
    def test_get_agents_empty(self, Mock_list_agents):
        """Test agents listing with no agents."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        Mock_list_agents.return_value = []
        
        response = client.get("/mesh/agents")
        
        assert response.status_code == 200
        assert response.json() == {"count": 0, "agents": []}
    
    @patch('backend.app.api.v1.endpoints.mesh.list_agents')
    def test_get_agents_redis_error(self, Mock_list_agents):
        """Test agents listing with Redis error."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        Mock_list_agents.side_effect = Exception("Redis connection failed")
        
        response = client.get("/mesh/agents")
        
        assert response.status_code == 500
        assert "Redis connection failed" in response.json()["detail"]

class TestHealthEndpoint:
    """Test the /health endpoint."""
    
    @patch('backend.app.api.v1.endpoints.mesh.get_redis')
    @patch('backend.app.api.v1.endpoints.mesh.list_agents')
    def test_health_success(self, Mock_list_agents, Mock_get_redis, Mock_redis, sample_agents_data):
        """Test successful health check."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        Mock_get_redis.return_value = Mock_redis
        Mock_redis.ping.return_value = True
        Mock_list_agents.return_value = sample_agents_data["agents"]
        
        response = client.get("/mesh/health")
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "ok"
        assert result["redis"] is True
        assert result["agents_count"] == 2
    
    @patch('backend.app.api.v1.endpoints.mesh.get_redis')
    @patch('backend.app.api.v1.endpoints.mesh.list_agents')
    def test_health_redis_degraded(self, Mock_list_agents, Mock_get_redis, Mock_redis):
        """Test health check with degraded Redis."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        Mock_get_redis.return_value = Mock_redis
        Mock_redis.ping.return_value = False
        Mock_list_agents.return_value = []
        
        response = client.get("/mesh/health")
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "degraded"
        assert result["redis"] is False
        assert result["agents_count"] == 0
    
    @patch('backend.app.api.v1.endpoints.mesh.get_redis')
    def test_health_redis_error(self, Mock_get_redis, Mock_redis):
        """Test health check with Redis connection error."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        Mock_get_redis.return_value = Mock_redis
        Mock_redis.ping.side_effect = Exception("Connection failed")
        
        response = client.get("/mesh/health")
        
        assert response.status_code == 503
        assert "mesh unhealthy" in response.json()["detail"]
        assert "Connection failed" in response.json()["detail"]

class TestOllamaGenerateEndpoint:
    """Test the /ollama/generate endpoint."""
    
    @pytest.fixture
    def sample_generate_request(self):
        """Sample generate request."""
        return {
            "model": "tinyllama",
            "prompt": "What is artificial intelligence?",
            "options": {
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }
    
    @pytest.fixture 
    def sample_ollama_response(self):
        """Sample Ollama response."""
        return {
            "model": "tinyllama",
            "created_at": "2024-01-01T00:00:00Z",
            "response": "Artificial intelligence (AI) is a field of computer science focused on creating intelligent machines.",
            "done": True,
            "context": [1, 2, 3, 4, 5],
            "total_duration": 5000000000,
            "load_duration": 1000000000,
            "prompt_eval_count": 15,
            "prompt_eval_duration": 2000000000,
            "eval_count": 25,
            "eval_duration": 2000000000
        }
    
    @patch('backend.app.api.v1.endpoints.mesh.default_ollama_bucket')
    @patch('backend.app.api.v1.endpoints.mesh.get_cache_service')
    @patch('backend.app.api.v1.endpoints.mesh.get_pool_manager')
    def test_ollama_generate_success(self, Mock_pool_manager, Mock_cache_service, 
                                   Mock_bucket, sample_generate_request, sample_ollama_response):
        """Test successful Ollama generation."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        import asyncio
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        # Mock rate limiter
        bucket_Mock = Mock()
        bucket_Mock.try_acquire.return_value = (True, 0)
        Mock_bucket.return_value = bucket_Mock
        
        # Mock cache service
        cache_Mock = AsyncMock()
        cache_Mock.get.return_value = None  # Cache miss
        cache_Mock.set.return_value = None
        Mock_cache_service.return_value = cache_Mock
        
        # Mock HTTP client and pool manager
        http_client_Mock = AsyncMock()
        response_Mock = Mock()
        response_Mock.json.return_value = sample_ollama_response
        response_Mock.raise_for_status.return_value = None
        http_client_Mock.post.return_value = response_Mock
        
        pool_manager_Mock = AsyncMock()
        pool_manager_Mock.get_http_client.return_value.__aenter__.return_value = http_client_Mock
        pool_manager_Mock.get_http_client.return_value.__aexit__.return_value = None
        Mock_pool_manager.return_value = pool_manager_Mock
        
        response = client.post("/mesh/ollama/generate", json=sample_generate_request)
        
        assert response.status_code == 200
        assert response.json() == sample_ollama_response
    
    def test_ollama_generate_prompt_too_large(self):
        """Test Ollama generation with oversized prompt."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        large_request = {
            "model": "tinyllama",
            "prompt": "x" * (32 * 1024 + 1),  # Exceeds 32KB limit
            "options": {}
        }
        
        response = client.post("/mesh/ollama/generate", json=large_request)
        
        assert response.status_code == 400
        assert "prompt too large" in response.json()["detail"]
    
    @patch('backend.app.api.v1.endpoints.mesh.default_ollama_bucket')
    def test_ollama_generate_rate_limited(self, Mock_bucket, sample_generate_request):
        """Test Ollama generation with rate limiting."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        # Mock rate limiter to deny request
        bucket_Mock = Mock()
        bucket_Mock.try_acquire.return_value = (False, 5000)  # Denied with 5s wait
        Mock_bucket.return_value = bucket_Mock
        
        response = client.post("/mesh/ollama/generate", json=sample_generate_request)
        
        assert response.status_code == 429
        assert response.json()["detail"]["retry_after_ms"] == 5000
    
    @patch('backend.app.api.v1.endpoints.mesh.default_ollama_bucket')
    @patch('backend.app.api.v1.endpoints.mesh.get_cache_service')
    def test_ollama_generate_cache_hit(self, Mock_cache_service, Mock_bucket, 
                                     sample_generate_request, sample_ollama_response):
        """Test Ollama generation with cache hit."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        # Mock rate limiter
        bucket_Mock = Mock()
        bucket_Mock.try_acquire.return_value = (True, 0)
        Mock_bucket.return_value = bucket_Mock
        
        # Mock cache service with cache hit
        cache_Mock = AsyncMock()
        cache_Mock.get.return_value = sample_ollama_response
        Mock_cache_service.return_value = cache_Mock
        
        response = client.post("/mesh/ollama/generate", json=sample_generate_request)
        
        assert response.status_code == 200
        assert response.json() == sample_ollama_response
        
        # Should not call cache.set since it was a cache hit
        cache_Mock.set.assert_not_called()
    
    def test_ollama_generate_default_model(self):
        """Test Ollama generation with default model."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        # Request without model field
        request_no_model = {
            "prompt": "Test prompt"
        }
        
        with patch.dict('os.environ', {'OLLAMA_DEFAULT_MODEL': 'test-model'}):
            with patch('backend.app.api.v1.endpoints.mesh.default_ollama_bucket') as Mock_bucket:
                bucket_Mock = Mock()
                bucket_Mock.try_acquire.return_value = (True, 0)
                Mock_bucket.return_value = bucket_Mock
                
                with patch('backend.app.api.v1.endpoints.mesh.get_cache_service') as Mock_cache:
                    cache_Mock = AsyncMock()
                    cache_Mock.get.return_value = None
                    Mock_cache.return_value = cache_Mock
                    
                    with patch('backend.app.api.v1.endpoints.mesh.get_pool_manager') as Mock_pool:
                        http_client_Mock = AsyncMock()
                        response_Mock = Mock()
                        response_Mock.json.return_value = {"response": "test"}
                        response_Mock.raise_for_status.return_value = None
                        http_client_Mock.post.return_value = response_Mock
                        
                        pool_manager_Mock = AsyncMock()
                        pool_manager_Mock.get_http_client.return_value.__aenter__.return_value = http_client_Mock
                        pool_manager_Mock.get_http_client.return_value.__aexit__.return_value = None
                        Mock_pool.return_value = pool_manager_Mock
                        
                        response = client.post("/mesh/ollama/generate", json=request_no_model)
                        
                        # Should use default model
                        call_args = http_client_Mock.post.call_args[1]['json']
                        assert call_args['model'] == 'test-model'

class TestRequestValidation:
    """Test request validation across all endpoints."""
    
    def test_enqueue_request_validation(self):
        """Test EnqueueRequest model validation."""
        from backend.app.api.v1.endpoints.mesh import EnqueueRequest
        from pydantic import ValidationError
        
        # Valid request
        valid_data = {"topic": "data_processing", "task": {"key": "value"}}
        request = EnqueueRequest(**valid_data)
        assert request.topic == "data_processing"
        assert request.task == {"key": "value"}
        
        # Invalid topic (empty)
        with pytest.raises(ValidationError):
            EnqueueRequest(topic="", task={"key": "value"})
        
        # Invalid topic (too long)
        with pytest.raises(ValidationError):
            EnqueueRequest(topic="a" * 65, task={"key": "value"})
        
        # Invalid topic (wrong pattern)
        with pytest.raises(ValidationError):
            EnqueueRequest(topic="invalid topic!", task={"key": "value"})
        
        # Missing task
        with pytest.raises(ValidationError):
            EnqueueRequest(topic="valid_topic")
    
    def test_enqueue_response_validation(self):
        """Test EnqueueResponse model validation."""
        from backend.app.api.v1.endpoints.mesh import EnqueueResponse
        
        response = EnqueueResponse(id="1699999999999-0")
        assert response.id == "1699999999999-0"
    
    def test_generate_request_validation(self):
        """Test GenerateRequest model validation."""
        from backend.app.api.v1.endpoints.mesh import GenerateRequest
        import os
        
        # Test with explicit model
        request = GenerateRequest(model="tinyllama", prompt="test prompt")
        assert request.model == "tinyllama"
        assert request.prompt == "test prompt"
        assert request.options is None
        
        # Test with options
        request_with_options = GenerateRequest(
            model="tinyllama", 
            prompt="test", 
            options={"temperature": 0.7}
        )
        assert request_with_options.options == {"temperature": 0.7}
        
        # Test default model from environment
        with patch.dict(os.environ, {'OLLAMA_DEFAULT_MODEL': 'default-model'}):
            request_default = GenerateRequest(prompt="test prompt")
            assert request_default.model == "default-model"

class TestErrorHandling:
    """Test error handling across all endpoints."""
    
    def test_internal_server_errors(self):
        """Test that internal errors are properly converted to HTTP exceptions."""
        from backend.app.api.v1.endpoints.mesh import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/mesh")
        client = TestClient(app)
        
        # Test various error scenarios
        with patch('backend.app.api.v1.endpoints.mesh.enqueue_task') as Mock_enqueue:
            Mock_enqueue.side_effect = RuntimeError("Internal error")
            response = client.post("/mesh/enqueue", json={"topic": "test", "task": {}})
            assert response.status_code == 500
        
        with patch('backend.app.api.v1.endpoints.mesh.tail_results') as Mock_tail:
            Mock_tail.side_effect = RuntimeError("Internal error")
            response = client.get("/mesh/results?topic=test")
            assert response.status_code == 500
        
        with patch('backend.app.api.v1.endpoints.mesh.list_agents') as Mock_list:
            Mock_list.side_effect = RuntimeError("Internal error")
            response = client.get("/mesh/agents")
            assert response.status_code == 500

if __name__ == "__main__":
    pytest.main([__file__, "-v"])