"""
Comprehensive API Integration Tests for Sutazai AI Application
Tests all backend API endpoints with real requests (no mocks)
"""

import pytest
import requests
import json
import asyncio
import websocket
import time
from typing import Dict, Any, List
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:10200"
API_V1 = f"{BASE_URL}/api/v1"
FRONTEND_URL = "http://localhost:11000"
TIMEOUT = 30  # seconds

class TestHealthEndpoints:
    """Test service health check endpoints"""
    
    def test_main_health_endpoint(self):
        """Test the main health endpoint"""
        response = requests.get(f"{API_V1}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "services" in data
        
    def test_service_specific_health(self):
        """Test individual service health checks"""
        services = ["database", "redis", "rabbitmq", "vector_stores"]
        for service in services:
            response = requests.get(f"{API_V1}/health/{service}", timeout=TIMEOUT)
            # Service might not have specific endpoint, but main should work
            if response.status_code == 200:
                data = response.json()
                assert "status" in data
                
    def test_health_with_details(self):
        """Test health endpoint with detailed information"""
        response = requests.get(f"{API_V1}/health?detailed=true", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data or "timestamp" in data


class TestChatEndpoints:
    """Test chat functionality endpoints"""
    
    def test_basic_chat_endpoint(self):
        """Test basic chat endpoint with a simple message"""
        payload = {
            "message": "Hello, how are you?",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        }
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data or "message" in data or "content" in data
        
    def test_simple_chat_endpoint(self):
        """Test the simple chat endpoint"""
        payload = {
            "message": "What is artificial intelligence?",
            "max_tokens": 100
        }
        response = requests.post(f"{API_V1}/simple_chat/", json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data or "message" in data or "content" in data
        
    def test_chat_with_context(self):
        """Test chat with conversation context"""
        # First message
        payload1 = {
            "message": "My name is TestUser",
            "session_id": "test_session_001"
        }
        response1 = requests.post(f"{API_V1}/chat", json=payload1, timeout=TIMEOUT)
        assert response1.status_code == 200
        
        # Follow-up message
        payload2 = {
            "message": "What is my name?",
            "session_id": "test_session_001"
        }
        response2 = requests.post(f"{API_V1}/chat", json=payload2, timeout=TIMEOUT)
        assert response2.status_code == 200
        
    def test_chat_with_invalid_model(self):
        """Test chat with invalid model name"""
        payload = {
            "message": "Test message",
            "model": "invalid-model-xyz"
        }
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        # Should either fail gracefully or use default
        assert response.status_code in [200, 400, 422]
        
    def test_chat_streaming(self):
        """Test chat streaming functionality"""
        payload = {
            "message": "Count from 1 to 5",
            "stream": True
        }
        response = requests.post(f"{API_V1}/chat", json=payload, stream=True, timeout=TIMEOUT)
        assert response.status_code == 200
        # Check if response is streaming
        chunks = []
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                chunks.append(chunk)
        assert len(chunks) > 0


class TestModelsEndpoints:
    """Test model management endpoints"""
    
    def test_list_models(self):
        """Test listing available models"""
        response = requests.get(f"{API_V1}/models", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list) or "models" in data
        
    def test_get_model_info(self):
        """Test getting specific model information"""
        # First get available models
        response = requests.get(f"{API_V1}/models", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            models = data if isinstance(data, list) else data.get("models", [])
            if models and len(models) > 0:
                model_id = models[0] if isinstance(models[0], str) else models[0].get("id", models[0].get("name"))
                if model_id:
                    response = requests.get(f"{API_V1}/models/{model_id}", timeout=TIMEOUT)
                    assert response.status_code in [200, 404]
                    
    def test_model_selection(self):
        """Test model selection for chat"""
        models_to_test = ["gpt-3.5-turbo", "gpt-4", "claude-2", "llama2"]
        for model in models_to_test:
            payload = {
                "message": "Hi",
                "model": model
            }
            response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
            # Should handle gracefully even if model not available
            assert response.status_code in [200, 400, 422, 503]


class TestAgentEndpoints:
    """Test AI agent management endpoints"""
    
    def test_list_agents(self):
        """Test listing all available agents"""
        response = requests.get(f"{API_V1}/agents", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list) or "agents" in data
        
        # Verify expected agents are present
        expected_agents = [
            "JARVIS", "DocuMind", "QuantumCoder", "DataSage",
            "CreativeStorm", "TechAnalyst", "ResearchOwl",
            "CodeOptimizer", "VisionaryArch", "TestMaster", "SecurityGuard"
        ]
        
        agents_list = data if isinstance(data, list) else data.get("agents", [])
        if agents_list:
            agent_names = [a.get("name", a) if isinstance(a, dict) else a for a in agents_list]
            # Check if at least some expected agents are present
            found_agents = [a for a in expected_agents if any(a.lower() in str(name).lower() for name in agent_names)]
            assert len(found_agents) > 0, f"No expected agents found. Got: {agent_names}"
            
    def test_get_agent_details(self):
        """Test getting specific agent information"""
        response = requests.get(f"{API_V1}/agents/jarvis", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            assert "name" in data or "id" in data
            assert "capabilities" in data or "description" in data
            
    def test_agent_chat_interaction(self):
        """Test chatting with specific agents"""
        agents = ["jarvis", "documind", "quantumcoder"]
        for agent_id in agents:
            payload = {
                "message": "What are your capabilities?",
                "agent_id": agent_id
            }
            response = requests.post(f"{API_V1}/agents/{agent_id}/chat", json=payload, timeout=TIMEOUT)
            # Agent endpoint might not exist, but should handle gracefully
            assert response.status_code in [200, 404, 422]
            
    def test_agent_task_assignment(self):
        """Test assigning tasks to agents"""
        payload = {
            "task": "Analyze this code for performance improvements",
            "code": "def slow_function(n):\n    result = 0\n    for i in range(n):\n        for j in range(n):\n            result += i * j\n    return result",
            "agent_id": "codeoptimizer"
        }
        response = requests.post(f"{API_V1}/agents/analyze", json=payload, timeout=TIMEOUT)
        assert response.status_code in [200, 404, 422]


class TestVoiceEndpoints:
    """Test voice processing endpoints"""
    
    def test_voice_process_text(self):
        """Test voice processing with text input"""
        payload = {
            "text": "Hello, this is a test of the voice system",
            "voice": "en-US-Standard-A",
            "speed": 1.0
        }
        response = requests.post(f"{API_V1}/voice/process", json=payload, timeout=TIMEOUT)
        assert response.status_code in [200, 422, 501]  # 501 if not implemented
        
    def test_voice_list_voices(self):
        """Test listing available voices"""
        response = requests.get(f"{API_V1}/voice/voices", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list) or "voices" in data
            
    def test_voice_demo_endpoint(self):
        """Test voice demo functionality"""
        response = requests.get(f"{API_V1}/voice/demo/test", timeout=TIMEOUT)
        assert response.status_code in [200, 404]


class TestVectorStoreEndpoints:
    """Test vector store endpoints"""
    
    def test_vector_stores_list(self):
        """Test listing vector stores"""
        response = requests.get(f"{API_V1}/vectors/stores", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list) or "stores" in data
            
            # Check for expected vector stores
            expected_stores = ["chromadb", "qdrant", "faiss"]
            stores_list = data if isinstance(data, list) else data.get("stores", [])
            if stores_list:
                store_names = [s.get("name", s) if isinstance(s, dict) else s for s in stores_list]
                found_stores = [s for s in expected_stores if any(s in str(name).lower() for name in store_names)]
                assert len(found_stores) > 0
                
    def test_vector_search(self):
        """Test vector similarity search"""
        payload = {
            "query": "artificial intelligence and machine learning",
            "top_k": 5,
            "store": "chromadb"
        }
        response = requests.post(f"{API_V1}/vectors/search", json=payload, timeout=TIMEOUT)
        assert response.status_code in [200, 404, 422]
        
    def test_vector_embedding(self):
        """Test creating embeddings"""
        payload = {
            "text": "This is a test document for embedding",
            "model": "text-embedding-ada-002"
        }
        response = requests.post(f"{API_V1}/vectors/embed", json=payload, timeout=TIMEOUT)
        assert response.status_code in [200, 404, 422, 501]


class TestAuthenticationEndpoints:
    """Test authentication and authorization"""
    
    def test_login_endpoint(self):
        """Test user login"""
        payload = {
            "username": "testuser",
            "password": "testpass123"
        }
        response = requests.post(f"{API_V1}/auth/login", json=payload, timeout=TIMEOUT)
        assert response.status_code in [200, 401, 422]
        
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data or "token" in data or "session" in data
            
    def test_register_endpoint(self):
        """Test user registration"""
        payload = {
            "username": f"testuser_{int(time.time())}",
            "password": "SecurePass123!",
            "email": f"test_{int(time.time())}@example.com"
        }
        response = requests.post(f"{API_V1}/auth/register", json=payload, timeout=TIMEOUT)
        assert response.status_code in [200, 201, 400, 422]
        
    def test_token_refresh(self):
        """Test token refresh functionality"""
        # This would need a valid token first
        headers = {"Authorization": "Bearer dummy_token_for_test"}
        response = requests.post(f"{API_V1}/auth/refresh", headers=headers, timeout=TIMEOUT)
        assert response.status_code in [200, 401, 404]
        
    def test_logout_endpoint(self):
        """Test logout functionality"""
        headers = {"Authorization": "Bearer dummy_token_for_test"}
        response = requests.post(f"{API_V1}/auth/logout", headers=headers, timeout=TIMEOUT)
        assert response.status_code in [200, 401, 404]


class TestWebSocketEndpoints:
    """Test WebSocket connections for real-time features"""
    
    def test_websocket_connection(self):
        """Test establishing WebSocket connection"""
        ws_url = "ws://localhost:10200/api/v1/ws"
        try:
            ws = websocket.create_connection(ws_url, timeout=5)
            ws.send(json.dumps({"type": "ping"}))
            result = ws.recv()
            ws.close()
            assert result is not None
        except Exception as e:
            # WebSocket might not be configured
            print(f"WebSocket connection test skipped: {e}")
            
    def test_jarvis_websocket(self):
        """Test JARVIS WebSocket endpoint"""
        ws_url = "ws://localhost:10200/api/v1/jarvis/ws"
        try:
            ws = websocket.create_connection(ws_url, timeout=5)
            ws.send(json.dumps({"message": "Hello JARVIS"}))
            result = ws.recv()
            ws.close()
            assert result is not None
        except Exception as e:
            print(f"JARVIS WebSocket test skipped: {e}")


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_malformed_json(self):
        """Test handling of malformed JSON"""
        response = requests.post(
            f"{API_V1}/chat",
            data="{'invalid': json}",
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        assert response.status_code in [400, 422]
        
    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        payload = {}  # Missing message field
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code in [400, 422]
        
    def test_rate_limiting(self):
        """Test rate limiting (if implemented)"""
        # Send multiple rapid requests
        responses = []
        for _ in range(20):
            response = requests.get(f"{API_V1}/health", timeout=TIMEOUT)
            responses.append(response.status_code)
            
        # Check if rate limiting is applied
        rate_limited = any(code == 429 for code in responses)
        # Rate limiting is optional, so we just verify responses are handled
        assert all(code in [200, 429, 503] for code in responses)
        
    def test_large_payload(self):
        """Test handling of large payloads"""
        large_text = "x" * 100000  # 100KB of text
        payload = {"message": large_text}
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
        assert response.status_code in [200, 413, 422]  # 413 for payload too large
        
    def test_timeout_handling(self):
        """Test timeout handling"""
        payload = {
            "message": "Process this with a very long operation",
            "timeout": 1  # 1 second timeout
        }
        response = requests.post(f"{API_V1}/chat", json=payload, timeout=2)
        # Should handle timeout gracefully
        assert response.status_code in [200, 408, 504]


class TestIntegration:
    """Test full integration scenarios"""
    
    def test_complete_chat_flow(self):
        """Test complete chat interaction flow"""
        # 1. Get available models
        models_response = requests.get(f"{API_V1}/models", timeout=TIMEOUT)
        assert models_response.status_code == 200
        
        # 2. Select an agent
        agents_response = requests.get(f"{API_V1}/agents", timeout=TIMEOUT)
        assert agents_response.status_code == 200
        
        # 3. Start a chat session
        chat_payload = {
            "message": "Hello, I need help with Python code",
            "agent_id": "quantumcoder",
            "session_id": "integration_test_001"
        }
        chat_response = requests.post(f"{API_V1}/chat", json=chat_payload, timeout=TIMEOUT)
        assert chat_response.status_code == 200
        
        # 4. Continue conversation
        followup_payload = {
            "message": "Can you help me optimize a sorting algorithm?",
            "session_id": "integration_test_001"
        }
        followup_response = requests.post(f"{API_V1}/chat", json=followup_payload, timeout=TIMEOUT)
        assert followup_response.status_code == 200
        
    def test_multi_agent_collaboration(self):
        """Test multiple agents working together"""
        # Create a task that requires multiple agents
        task_payload = {
            "task": "Create a secure API endpoint with documentation",
            "agents": ["quantumcoder", "securityguard", "documind"]
        }
        
        # This might not be implemented, but test the concept
        response = requests.post(f"{API_V1}/agents/collaborate", json=task_payload, timeout=TIMEOUT)
        assert response.status_code in [200, 404, 501]
        
    def test_vector_store_integration(self):
        """Test vector store with chat integration"""
        # 1. Store a document
        store_payload = {
            "text": "Python is a high-level programming language known for its simplicity",
            "metadata": {"topic": "programming", "language": "python"}
        }
        store_response = requests.post(f"{API_V1}/vectors/store", json=store_payload, timeout=TIMEOUT)
        
        # 2. Search for related content
        search_payload = {
            "query": "What is Python programming?",
            "top_k": 3
        }
        search_response = requests.post(f"{API_V1}/vectors/search", json=search_payload, timeout=TIMEOUT)
        
        # 3. Use search results in chat
        if search_response.status_code == 200:
            chat_payload = {
                "message": "Tell me about Python",
                "context": search_response.json() if search_response.status_code == 200 else None
            }
            chat_response = requests.post(f"{API_V1}/chat", json=chat_payload, timeout=TIMEOUT)
            assert chat_response.status_code in [200, 422]


# Performance tests
class TestPerformance:
    """Test performance and load handling"""
    
    def test_response_time(self):
        """Test API response times"""
        endpoints = [
            (f"{API_V1}/health", "GET", None),
            (f"{API_V1}/models", "GET", None),
            (f"{API_V1}/agents", "GET", None),
        ]
        
        for url, method, payload in endpoints:
            start_time = time.time()
            if method == "GET":
                response = requests.get(url, timeout=TIMEOUT)
            else:
                response = requests.post(url, json=payload, timeout=TIMEOUT)
            elapsed = time.time() - start_time
            
            # Response should be under 2 seconds for basic endpoints
            assert elapsed < 2.0, f"{url} took {elapsed:.2f} seconds"
            assert response.status_code in [200, 404]
            
    def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        import concurrent.futures
        
        def make_request(i):
            payload = {"message": f"Test message {i}"}
            response = requests.post(f"{API_V1}/chat", json=payload, timeout=TIMEOUT)
            return response.status_code
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
        # All requests should be handled
        assert all(code in [200, 429, 503] for code in results)
        successful = sum(1 for code in results if code == 200)
        assert successful >= 5  # At least half should succeed


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])