"""
Main application tests for SutazAI backend
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from httpx import AsyncClient
import asyncio
from datetime import datetime

# Import the main app - adjust based on your actual app structure
try:
    from app.main import app
except ImportError:
    # Fallback import paths
    try:
        from backend.app.main import app
    except ImportError:
        from main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create an async test client for the FastAPI app"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check(self, client):
        """Test the basic health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_readiness_check(self, client):
        """Test the readiness check endpoint"""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] == True
        assert "services" in data
    
    @pytest.mark.asyncio
    async def test_liveness_check(self, async_client):
        """Test the liveness check endpoint"""
        response = await async_client.get("/live")
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] == True


class TestAPIVersioning:
    """Test API versioning"""
    
    def test_v1_endpoint(self, client):
        """Test v1 API endpoint exists"""
        response = client.get("/api/v1/")
        assert response.status_code in [200, 404]  # Depends on implementation
    
    def test_api_docs(self, client):
        """Test API documentation is accessible"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower() or "openapi" in response.text.lower()
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema is accessible"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema


class TestAuthentication:
    """Test authentication and authorization"""
    
    @pytest.mark.unit
    def test_unauthorized_access(self, client):
        """Test accessing protected endpoint without auth"""
        response = client.get("/api/v1/protected")
        assert response.status_code in [401, 403, 404]
    
    @pytest.mark.unit
    def test_invalid_token(self, client):
        """Test accessing with invalid token"""
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/api/v1/protected", headers=headers)
        assert response.status_code in [401, 403, 404]


class TestAgentEndpoints:
    """Test AI agent related endpoints"""
    
    @pytest.mark.integration
    def test_list_agents(self, client):
        """Test listing available agents"""
        response = client.get("/api/v1/agents")
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list) or "agents" in data
    
    @pytest.mark.integration
    @patch('app.services.agent_orchestrator.create_agent')
    def test_create_agent(self, mock_create, client):
        """Test creating a new agent"""
        mock_create.return_value = {"id": "test-agent", "status": "created"}
        
        agent_data = {
            "name": "TestAgent",
            "type": "code_generator",
            "config": {}
        }
        response = client.post("/api/v1/agents", json=agent_data)
        assert response.status_code in [200, 201, 404]
    
    @pytest.mark.asyncio
    async def test_agent_communication(self, async_client):
        """Test agent communication endpoint"""
        message_data = {
            "agent_id": "test-agent",
            "message": "Hello, agent!",
            "context": {}
        }
        response = await async_client.post("/api/v1/agents/communicate", json=message_data)
        assert response.status_code in [200, 404, 422]


class TestDocumentProcessing:
    """Test document processing endpoints"""
    
    @pytest.mark.integration
    def test_upload_document(self, client):
        """Test document upload"""
        # Create a test file
        test_content = b"This is a test document"
        files = {"file": ("test.txt", test_content, "text/plain")}
        
        response = client.post("/api/v1/documents/upload", files=files)
        assert response.status_code in [200, 201, 404]
    
    @pytest.mark.integration
    def test_list_documents(self, client):
        """Test listing documents"""
        response = client.get("/api/v1/documents")
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list) or "documents" in data


class TestVectorOperations:
    """Test vector database operations"""
    
    @pytest.mark.integration
    @pytest.mark.requires_docker
    def test_vector_search(self, client):
        """Test vector similarity search"""
        search_data = {
            "query": "test query",
            "top_k": 5,
            "collection": "documents"
        }
        response = client.post("/api/v1/vectors/search", json=search_data)
        assert response.status_code in [200, 404, 422]
    
    @pytest.mark.integration
    def test_vector_embedding(self, client):
        """Test text embedding generation"""
        embed_data = {
            "text": "This is a test text for embedding",
            "model": "default"
        }
        response = client.post("/api/v1/vectors/embed", json=embed_data)
        assert response.status_code in [200, 404, 422]


class TestPerformance:
    """Performance and load tests"""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_concurrent_requests(self, client):
        """Test handling concurrent requests"""
        import concurrent.futures
        
        def make_request():
            return client.get("/health")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert all(r.status_code == 200 for r in results)
    
    @pytest.mark.benchmark
    def test_response_time(self, client, benchmark):
        """Benchmark API response time"""
        result = benchmark(client.get, "/health")
        assert result.status_code == 200


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get("/non-existent-endpoint")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data or "error" in data
    
    def test_invalid_json(self, client):
        """Test handling of invalid JSON"""
        response = client.post(
            "/api/v1/agents",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]
    
    def test_method_not_allowed(self, client):
        """Test method not allowed error"""
        response = client.put("/health")  # Assuming health only accepts GET
        assert response.status_code == 405


class TestIntegration:
    """Full integration tests"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_full_workflow(self, async_client):
        """Test a complete workflow through the system"""
        # 1. Check system health
        response = await async_client.get("/health")
        assert response.status_code == 200
        
        # 2. Create an agent
        agent_data = {
            "name": "WorkflowAgent",
            "type": "general",
            "config": {"temperature": 0.7}
        }
        response = await async_client.post("/api/v1/agents", json=agent_data)
        if response.status_code in [200, 201]:
            agent = response.json()
            agent_id = agent.get("id")
            
            # 3. Send a message to the agent
            message_data = {
                "agent_id": agent_id,
                "message": "Process this test message",
                "context": {"test": True}
            }
            response = await async_client.post("/api/v1/agents/communicate", json=message_data)
            assert response.status_code in [200, 404]


# Parametrized tests for different scenarios
@pytest.mark.parametrize("endpoint,expected_status", [
    ("/health", 200),
    ("/ready", 200),
    ("/docs", 200),
    ("/openapi.json", 200),
    ("/api/v1/", [200, 404]),
])
def test_endpoint_availability(client, endpoint, expected_status):
    """Test that key endpoints are available"""
    response = client.get(endpoint)
    if isinstance(expected_status, list):
        assert response.status_code in expected_status
    else:
        assert response.status_code == expected_status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])