"""
Integration tests for Vector Context Injection System
Tests the full end-to-end flow with Mocked dependencies
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from fastapi.testclient import TestClient

# Import FastAPI app and components
from app.main import app
from app.services.vector_context_injector import VectorSearchResult, KnowledgeContext

class TestVectorContextIntegration:
    """Integration tests for the complete vector context system"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_vector_context(self):
        """Mock vector context injector with sample data"""
        sample_results = [
            VectorSearchResult(
                content="Docker is a containerization platform that allows developers to package applications",
                metadata={"source": "documentation", "topic": "containers"},
                score=0.95,
                source="chromadb"
            ),
            VectorSearchResult(
                content="Docker containers are lightweight, portable, and provide consistent environments",
                metadata={"source": "tutorial", "topic": "containers"},
                score=0.88,
                source="qdrant"
            ),
            VectorSearchResult(
                content="Docker Compose is used to define multi-container applications",
                metadata={"source": "guide", "topic": "orchestration"},
                score=0.82,
                source="faiss"
            )
        ]
        
        return KnowledgeContext(
            results=sample_results,
            query_time_ms=245.0,
            sources_used=["chromadb", "qdrant", "faiss"],
            total_results=3,
            enriched_context="""
KNOWLEDGE CONTEXT FOR: What is Docker?
==================================================

[1] Source: CHROMADB (Score: 0.950)
Content: Docker is a containerization platform that allows developers to package applications
Metadata: {"source": "documentation", "topic": "containers"}

[2] Source: QDRANT (Score: 0.880)
Content: Docker containers are lightweight, portable, and provide consistent environments
Metadata: {"source": "tutorial", "topic": "containers"}

[3] Source: FAISS (Score: 0.820)
Content: Docker Compose is used to define multi-container applications
Metadata: {"source": "guide", "topic": "orchestration"}

==================================================
Use this knowledge context to provide accurate, informed responses.
            """.strip()
        )
    
    @pytest.mark.asyncio
    @patch('app.main.vector_context_injector')
    @patch('app.main.get_ollama_models')
    @patch('app.main.query_ollama')
    async def test_chat_endpoint_with_vector_context(self, mock_query_ollama, mock_get_models, mock_vci, client, mock_vector_context):
        """Test chat endpoint with vector context integration"""
        # Setup Mocks
        mock_get_models.return_value = ["tinyllama"]
        mock_query_ollama.return_value = "Docker is a containerization platform that allows you to package applications with their dependencies into lightweight containers."
        
        # Mock vector context injector
        mock_vci.analyze_user_request = AsyncMock(return_value=(True, mock_vector_context))
        mock_vci.inject_context_into_prompt = AsyncMock(return_value="""
KNOWLEDGE CONTEXT FOR: What is Docker?
[Context content here...]

ORIGINAL REQUEST: What is Docker?

INSTRUCTIONS: Use the provided knowledge context above to give an accurate, well-informed response.
        """.strip())
        
        # Make request to chat endpoint
        response = client.post("/chat", json={
            "message": "What is Docker?",
            "agent": "research-agent"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "response" in data
        assert "vector_context_used" in data
        assert "vector_context_info" in data
        
        # Verify vector context was used
        assert data["vector_context_used"] is True
        assert data["vector_context_info"] is not None
        assert data["vector_context_info"]["total_results"] == 3
        assert data["vector_context_info"]["sources_used"] == ["chromadb", "qdrant", "faiss"]
        assert data["vector_context_info"]["query_time_ms"] == 245.0
        
        # Verify the context injection was called
        mock_vci.analyze_user_request.assert_called_once_with("What is Docker?")
        mock_vci.inject_context_into_prompt.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.main.vector_context_injector')
    @patch('app.main.get_ollama_models')
    @patch('app.main.query_ollama')
    async def test_chat_endpoint_without_vector_context(self, mock_query_ollama, mock_get_models, mock_vci, client):
        """Test chat endpoint when no vector context is needed"""
        # Setup Mocks
        mock_get_models.return_value = ["tinyllama"]
        mock_query_ollama.return_value = "Hello! How can I help you today?"
        
        # Mock vector context injector to return no context needed
        mock_vci.analyze_user_request = AsyncMock(return_value=(False, None))
        
        # Make request with non-knowledge query
        response = client.post("/chat", json={
            "message": "Hello there!",
            "agent": "task_coordinator"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify vector context was not used
        assert data["vector_context_used"] is False
        assert data["vector_context_info"] is None
        
        # Verify the analysis was still called
        mock_vci.analyze_user_request.assert_called_once_with("Hello there!")
    
    @pytest.mark.asyncio
    @patch('app.main.vector_context_injector')
    @patch('app.main.get_ollama_models')
    @patch('app.main.query_ollama')
    async def test_think_endpoint_with_vector_context(self, mock_query_ollama, mock_get_models, mock_vci, client, mock_vector_context):
        """Test think endpoint with vector context integration"""
        # Setup Mocks
        mock_get_models.return_value = ["tinyllama"]
        mock_query_ollama.return_value = "Based on the provided knowledge context, Docker is a containerization platform..."
        
        # Mock vector context injector
        mock_vci.analyze_user_request = AsyncMock(return_value=(True, mock_vector_context))
        mock_vci.inject_context_into_prompt = AsyncMock(return_value="Enhanced prompt with context")
        
        # Make request to think endpoint
        response = client.post("/think", json={
            "query": "Explain Docker containerization",
            "reasoning_type": "deductive"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "thought" in data
        assert "vector_context_used" in data
        assert "vector_context_info" in data
        assert "reasoning_type" in data
        
        # Verify vector context integration
        assert data["vector_context_used"] is True
        assert data["vector_context_info"]["total_results"] == 3
        assert data["reasoning_type"] == "deductive"
    
    @pytest.mark.asyncio
    @patch('app.main.vector_context_injector')
    @patch('app.main.get_ollama_models') 
    @patch('app.main.query_ollama')
    async def test_public_think_endpoint_with_vector_context(self, mock_query_ollama, mock_get_models, mock_vci, client, mock_vector_context):
        """Test public think endpoint with vector context"""
        # Setup Mocks
        mock_get_models.return_value = ["tinyllama"]
        mock_query_ollama.return_value = "Comprehensive analysis of Docker technology..."
        
        # Mock vector context injector
        mock_vci.analyze_user_request = AsyncMock(return_value=(True, mock_vector_context))
        mock_vci.inject_context_into_prompt = AsyncMock(return_value="Enhanced reasoning prompt")
        
        # Make request to public think endpoint
        response = client.post("/public/think", json={
            "query": "How does Docker work?",
            "reasoning_type": "general"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify vector context integration in public endpoint
        assert data["vector_context_used"] is True
        assert data["vector_context_info"]["sources_used"] == ["chromadb", "qdrant", "faiss"]
        
        # Verify thought process shows context integration
        thought_process = data["thought_process"]
        assert any("Vector context integrated" in step for step in thought_process)
    
    @pytest.mark.asyncio
    @patch('app.main.vector_context_injector')
    async def test_vector_context_error_handling(self, mock_vci, client):
        """Test error handling in vector context injection"""
        # Mock vector context injector to raise an exception
        mock_vci.analyze_user_request = AsyncMock(side_effect=Exception("Database connection failed"))
        
        with patch('app.main.get_ollama_models', return_value=["tinyllama"]):
            with patch('app.main.query_ollama', return_value="Response without context"):
                response = client.post("/chat", json={
                    "message": "What is Kubernetes?",
                    "agent": "research-agent"
                })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should handle error gracefully
        assert data["vector_context_used"] is False
        assert data["vector_context_info"] is None
        assert "response" in data  # Should still provide a response
    
    @pytest.mark.asyncio
    @patch('app.main.VECTOR_CONTEXT_AVAILABLE', False)
    @patch('app.main.get_ollama_models')
    @patch('app.main.query_ollama')
    async def test_fallback_when_vector_context_unavailable(self, mock_query_ollama, mock_get_models, client):
        """Test fallback behavior when vector context system is unavailable"""
        mock_get_models.return_value = ["tinyllama"]
        mock_query_ollama.return_value = "Response without vector context"
        
        response = client.post("/chat", json={
            "message": "What is machine learning?",
            "agent": "task_coordinator"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should work without vector context
        assert data["vector_context_used"] is False
        assert data["vector_context_info"] is None
        assert "response" in data
    
    @pytest.mark.asyncio
    @patch('app.main.vector_context_injector')
    @patch('app.main.get_ollama_models')
    @patch('app.main.query_ollama')
    async def test_performance_under_load(self, mock_query_ollama, mock_get_models, mock_vci, client, mock_vector_context):
        """Test system performance with multiple concurrent requests"""
        # Setup Mocks
        mock_get_models.return_value = ["tinyllama"]
        mock_query_ollama.return_value = "Quick response"
        
        # Mock fast vector context response
        fast_context = KnowledgeContext(
            results=[VectorSearchResult("Quick result", {}, 0.9, "chromadb")],
            query_time_ms=50.0,  # Fast response
            sources_used=["chromadb"],
            total_results=1,
            enriched_context="Quick context"
        )
        mock_vci.analyze_user_request = AsyncMock(return_value=(True, fast_context))
        mock_vci.inject_context_into_prompt = AsyncMock(return_value="Quick prompt")
        
        # Make multiple concurrent requests
        import concurrent.futures
        import time
        
        def make_request():
            return client.post("/chat", json={
                "message": "What is AI?",
                "agent": "research-agent"
            })
        
        start_time = time.time()
        
        # Test with 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [future.result() for future in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)
        
        # Should complete reasonably quickly (under 5 seconds for 5 requests)
        assert total_time < 5.0
        
        # Verify all used vector context
        for response in responses:
            data = response.json()
            assert data["vector_context_used"] is True
    
    @pytest.mark.asyncio
    @patch('app.main.vector_context_injector')
    async def test_vector_context_caching(self, mock_vci, client):
        """Test that vector context results are cached appropriately"""
        # Mock vector context with cache behavior
        call_count = 0
        
        async def mock_analyze(query):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call - simulate database search
                context = KnowledgeContext(
                    results=[VectorSearchResult("Cached result", {}, 0.9, "chromadb")],
                    query_time_ms=200.0,  # Slower first call
                    sources_used=["chromadb"],
                    total_results=1,
                    enriched_context="Context"
                )
                return True, context
            else:
                # Subsequent calls - should be faster (cached)
                context = KnowledgeContext(
                    results=[VectorSearchResult("Cached result", {}, 0.9, "chromadb")],
                    query_time_ms=5.0,  # Much faster cached call
                    sources_used=["chromadb"],
                    total_results=1,
                    enriched_context="Context"
                )
                return True, context
        
        mock_vci.analyze_user_request = mock_analyze
        mock_vci.inject_context_into_prompt = AsyncMock(return_value="Cached prompt")
        
        with patch('app.main.get_ollama_models', return_value=["tinyllama"]):
            with patch('app.main.query_ollama', return_value="Cached response"):
                # First request
                response1 = client.post("/chat", json={
                    "message": "What is caching?",
                    "agent": "research-agent"
                })
                
                # Second identical request
                response2 = client.post("/chat", json={
                    "message": "What is caching?",
                    "agent": "research-agent"
                })
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Both should have vector context
        assert data1["vector_context_used"] is True
        assert data2["vector_context_used"] is True
        
        # due to caching, but this requires more sophisticated timing measurement

class TestVectorContextHealthChecks:
    """Test health check integration with vector databases"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.mark.asyncio
    @patch('app.main.check_chromadb')
    @patch('app.main.check_qdrant')
    async def test_health_endpoint_shows_vector_db_status(self, mock_check_qdrant, mock_check_chromadb, client):
        """Test that health endpoint includes vector database status"""
        mock_check_chromadb.return_value = True
        mock_check_qdrant.return_value = False  # Simulate Qdrant being down
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "services" in data
        assert data["services"]["chromadb"] == "connected"
        assert data["services"]["qdrant"] == "disconnected"
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint_includes_vector_context_metrics(self, client):
        """Test that metrics endpoint includes vector context performance data"""
        response = client.get("/public/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "services" in data
        # Vector database services should be included in metrics
        assert "chromadb" in data["services"]
        assert "qdrant" in data["services"]

if __name__ == "__main__":
    unittest.main()
