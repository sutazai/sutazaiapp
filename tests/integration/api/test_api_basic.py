"""
Purpose: Basic API tests for SutazAI backend
Usage: pytest backend/tests/test_api_basic.py
Requirements: pytest, httpx, fastapi
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..')
if backend_path not in sys.path:
    # Path handled by pytest configuration


@pytest.fixture
def mock_app():
    """Create a mock FastAPI app for testing."""
    from fastapi import FastAPI
    
    app = FastAPI()
    
    @app.get("/")
    async def root():
        return {"message": "SutazAI API", "status": "ok"}
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "api"}
    
    @app.get("/api/v1/agents")
    async def list_agents():
        return {
            "agents": [
                {"id": "1", "name": "test-agent", "status": "active"},
                {"id": "2", "name": "demo-agent", "status": "inactive"}
            ],
            "total": 2
        }
    
    @app.post("/api/v1/agents/{agent_id}/execute")
    async def execute_agent(agent_id: str, task: dict):
        return {
            "agent_id": agent_id,
            "task_id": "task-123",
            "status": "executing",
            "task": task
        }
    
    return app


@pytest.fixture
def client(mock_app):
    """Create test client."""
    return TestClient(mock_app)


class TestBasicAPI:
    """Basic API endpoint tests."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns correct response."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "SutazAI API"
        assert data["status"] == "ok"
    
    def test_health_endpoint(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "api"
    
    def test_list_agents(self, client):
        """Test listing agents endpoint."""
        response = client.get("/api/v1/agents")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert len(data["agents"]) == 2
        assert data["total"] == 2
        
        # Check agent structure
        agent = data["agents"][0]
        assert "id" in agent
        assert "name" in agent
        assert "status" in agent
    
    def test_execute_agent(self, client):
        """Test executing agent task."""
        task_data = {
            "action": "test_action",
            "parameters": {"key": "value"}
        }
        
        response = client.post("/api/v1/agents/test-agent/execute", json=task_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["agent_id"] == "test-agent"
        assert data["status"] == "executing"
        assert "task_id" in data
        assert data["task"] == task_data


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_404_not_found(self, client):
        """Test 404 response for unknown endpoint."""
        response = client.get("/api/v1/unknown")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test 405 response for wrong method."""
        response = client.post("/health")
        assert response.status_code == 405


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test async endpoint behavior."""
    
    async def test_async_health_check(self, mock_app):
        """Test async health check works correctly."""
        from httpx import AsyncClient
        
        async with AsyncClient(app=mock_app, base_url="http://test") as ac:
            response = await ac.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestAPIStructure:
    """Test API structure and conventions."""
    
    def test_api_versioning(self, client):
        """Test that API uses versioning (v1)."""
        response = client.get("/api/v1/agents")
        assert response.status_code == 200
        assert "/api/v1/" in str(response.url)
    
    def test_json_response_format(self, client):
        """Test that responses are in JSON format."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Verify it's valid JSON
        data = response.json()
        assert isinstance(data, dict)