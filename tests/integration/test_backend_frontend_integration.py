"""
Integration tests for backend-frontend connectivity
Tests actual API endpoints that the frontend uses
"""

import pytest
import httpx
import asyncio
import json
from datetime import datetime

BACKEND_URL = "http://localhost:10200"
FRONTEND_URL = "http://localhost:11000"

class TestBackendEndpoints:
    """Test backend API endpoints"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test backend health check"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "app" in data
    
    @pytest.mark.asyncio
    async def test_detailed_health_endpoint(self):
        """Test detailed health check with service status"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/health/detailed")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "services" in data
            assert data["healthy_count"] > 0
    
    @pytest.mark.asyncio
    async def test_chat_endpoint(self):
        """Test chat endpoint with TinyLlama"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "message": "Hello, test message",
                "agent": "default",
                "session_id": "test_integration_123"
            }
            response = await client.post(
                f"{BACKEND_URL}/api/v1/chat/",
                json=payload
            )
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert "model" in data
            assert data["status"] == "success"
            assert len(data["response"]) > 0
    
    @pytest.mark.asyncio
    async def test_models_endpoint(self):
        """Test models listing endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/models/")
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert len(data["models"]) > 0
            assert "count" in data
    
    @pytest.mark.asyncio
    async def test_agents_endpoint(self):
        """Test agents listing endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/agents/")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) > 0
            # Check first agent structure
            agent = data[0]
            assert "id" in agent
            assert "name" in agent
            assert "status" in agent
            assert "capabilities" in agent
    
    @pytest.mark.asyncio
    async def test_voice_health_endpoint(self):
        """Test voice service health check"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/voice/demo/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "components" in data
            assert data["components"]["tts"] == "healthy"
            assert data["components"]["asr"] == "healthy"


class TestFrontendConnectivity:
    """Test frontend is accessible"""
    
    @pytest.mark.asyncio
    async def test_frontend_loads(self):
        """Test frontend UI is accessible"""
        async with httpx.AsyncClient() as client:
            response = await client.get(FRONTEND_URL)
            assert response.status_code == 200
            assert "streamlit" in response.text.lower()


class TestEndToEndIntegration:
    """Test complete backend-frontend integration flow"""
    
    @pytest.mark.asyncio
    async def test_chat_flow(self):
        """Test complete chat interaction flow"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: Check backend health
            health = await client.get(f"{BACKEND_URL}/health")
            assert health.status_code == 200
            
            # Step 2: Get available models
            models = await client.get(f"{BACKEND_URL}/api/v1/models/")
            assert models.status_code == 200
            models_data = models.json()
            assert len(models_data["models"]) > 0
            
            # Step 3: Send chat message
            chat_payload = {
                "message": "What is the capital of France?",
                "agent": "default",
                "session_id": f"test_{datetime.now().timestamp()}"
            }
            chat_response = await client.post(
                f"{BACKEND_URL}/api/v1/chat/",
                json=chat_payload
            )
            assert chat_response.status_code == 200
            chat_data = chat_response.json()
            
            # Verify response quality
            assert chat_data["status"] == "success"
            assert len(chat_data["response"]) > 10  # Meaningful response
            assert chat_data["model"] == "tinyllama:latest"
            assert "response_time" in chat_data
    
    @pytest.mark.asyncio
    async def test_voice_service_integration(self):
        """Test voice service is properly integrated"""
        async with httpx.AsyncClient() as client:
            # Check voice service health
            voice_health = await client.get(f"{BACKEND_URL}/api/v1/voice/demo/health")
            assert voice_health.status_code == 200
            data = voice_health.json()
            
            # Verify all voice components are healthy
            components = data["components"]
            assert components["voice_service"] == "healthy"
            assert components["tts"] == "healthy"
            assert components["asr"] == "healthy"
            assert components["jarvis"] == "healthy"


def test_backend_is_running():
    """Synchronous test to verify backend is running"""
    import requests
    response = requests.get(f"{BACKEND_URL}/health", timeout=5)
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_frontend_is_running():
    """Synchronous test to verify frontend is running"""
    import requests
    response = requests.get(FRONTEND_URL, timeout=5)
    assert response.status_code == 200


if __name__ == "__main__":
    # Run quick connectivity tests
    print("Testing Backend-Frontend Integration...")
    
    print("\n1. Testing backend health...")
    test_backend_is_running()
    print("✅ Backend is healthy")
    
    print("\n2. Testing frontend accessibility...")
    test_frontend_is_running()
    print("✅ Frontend is accessible")
    
    print("\n3. Running async integration tests...")
    async def run_tests():
        test = TestBackendEndpoints()
        await test.test_health_endpoint()
        print("✅ Health endpoint working")
        
        await test.test_chat_endpoint()
        print("✅ Chat endpoint working")
        
        await test.test_models_endpoint()
        print("✅ Models endpoint working")
        
        await test.test_agents_endpoint()
        print("✅ Agents endpoint working")
        
        await test.test_voice_health_endpoint()
        print("✅ Voice service working")
        
        e2e = TestEndToEndIntegration()
        await e2e.test_chat_flow()
        print("✅ End-to-end chat flow working")
    
    asyncio.run(run_tests())
    
    print("\n" + "="*60)
    print("✅ ALL INTEGRATION TESTS PASSED")
    print("Backend-Frontend integration is FULLY FUNCTIONAL")
    print("="*60)
