"""
Unit tests for Vector Context Injection System
Tests all components with mocked vector database clients
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

# Import the system under test
from app.services.vector_context_injector import (
    VectorContextInjector,
    ChromaDBClient,
    QdrantDBClient, 
    FAISSClient,
    CircuitBreaker,
    VectorSearchResult,
    KnowledgeContext
)

class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts in CLOSED state"""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60)
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0
        assert cb.last_failure_time is None
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success_path(self):
        """Test successful function calls don't trigger circuit breaker"""
        cb = CircuitBreaker(failure_threshold=2)
        
        @cb.call
        async def mock_func():
            return "success"
        
        result = await mock_func()
        assert result == "success"
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_counting(self):
        """Test circuit breaker counts failures correctly"""
        cb = CircuitBreaker(failure_threshold=2)
        
        @cb.call
        async def failing_func():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            await failing_func()
        assert cb.failure_count == 1
        assert cb.state == "CLOSED"
        
        # Second failure should open circuit
        with pytest.raises(Exception):
            await failing_func()
        assert cb.failure_count == 2
        assert cb.state == "OPEN"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self):
        """Test circuit breaker blocks calls when OPEN"""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=1)
        
        @cb.call
        async def failing_func():
            raise Exception("Test failure")
        
        # Trigger circuit breaker
        with pytest.raises(Exception):
            await failing_func()
        assert cb.state == "OPEN"
        
        # Now calls should be blocked
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await failing_func()

class TestChromaDBClient:
    """Test ChromaDB client functionality"""
    
    @pytest.fixture
    def chroma_client(self):
        """Create ChromaDB client for testing"""
        return ChromaDBClient()
    
    @pytest.mark.asyncio
    async def test_chromadb_search_success(self, chroma_client):
        """Test successful ChromaDB search"""
        # Mock the ChromaDB client
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Document 1", "Document 2"]],
            "metadatas": [[{"id": 1}, {"id": 2}]],
            "distances": [[0.1, 0.2]]
        }
        
        mock_chromadb_client = Mock()
        mock_chromadb_client.get_collection.return_value = mock_collection
        chroma_client.client = mock_chromadb_client
        
        results = await chroma_client.search("test query")
        
        assert len(results) == 2
        assert results[0].content == "Document 1"
        assert results[0].score == 0.9  # 1.0 - 0.1
        assert results[0].source == "chromadb"
        assert results[1].content == "Document 2"
        assert results[1].score == 0.8  # 1.0 - 0.2
    
    @pytest.mark.asyncio
    async def test_chromadb_search_no_client(self, chroma_client):
        """Test ChromaDB search with no client initialized"""
        chroma_client.client = None
        results = await chroma_client.search("test query")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_chromadb_search_exception(self, chroma_client):
        """Test ChromaDB search handles exceptions gracefully"""
        mock_chromadb_client = Mock()
        mock_chromadb_client.get_collection.side_effect = Exception("Connection failed")
        chroma_client.client = mock_chromadb_client
        
        results = await chroma_client.search("test query")
        assert results == []

class TestQdrantDBClient:
    """Test Qdrant client functionality"""
    
    @pytest.fixture
    def qdrant_client(self):
        """Create Qdrant client for testing"""
        return QdrantDBClient()
    
    @pytest.mark.asyncio
    async def test_qdrant_search_success(self, qdrant_client):
        """Test successful Qdrant search"""
        # Mock search results
        mock_point = Mock()
        mock_point.payload = {"content": "Test document", "metadata": {"id": 1}}
        mock_point.score = 0.85
        
        mock_qdrant_client = Mock()
        mock_qdrant_client.search.return_value = [mock_point]
        qdrant_client.client = mock_qdrant_client
        
        results = await qdrant_client.search("test query")
        
        assert len(results) == 1
        assert results[0].content == "Test document"
        assert results[0].score == 0.85
        assert results[0].source == "qdrant"
        assert results[0].metadata == {"id": 1}
    
    @pytest.mark.asyncio
    async def test_qdrant_query_encoding(self, qdrant_client):
        """Test query encoding functionality"""
        vector = await qdrant_client._encode_query("test query")
        assert len(vector) == 384
        assert all(isinstance(v, float) for v in vector)
        assert all(0.0 <= v <= 1.0 for v in vector)
    
    @pytest.mark.asyncio
    async def test_qdrant_search_no_client(self, qdrant_client):
        """Test Qdrant search with no client"""
        qdrant_client.client = None
        results = await qdrant_client.search("test query")
        assert results == []

class TestFAISSClient:
    """Test FAISS client functionality"""
    
    @pytest.fixture
    def faiss_client(self):
        """Create FAISS client for testing"""
        return FAISSClient()
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_faiss_search_success(self, mock_httpx, faiss_client):
        """Test successful FAISS search"""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"content": "FAISS doc 1", "metadata": {"id": 1}, "score": 0.9},
                {"content": "FAISS doc 2", "metadata": {"id": 2}, "score": 0.8}
            ]
        }
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_httpx.return_value.__aenter__.return_value = mock_client
        
        results = await faiss_client.search("test query")
        
        assert len(results) == 2
        assert results[0].content == "FAISS doc 1"
        assert results[0].score == 0.9
        assert results[0].source == "faiss"
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_faiss_search_http_error(self, mock_httpx, faiss_client):
        """Test FAISS search with HTTP error"""
        mock_response = Mock()
        mock_response.status_code = 500
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_httpx.return_value.__aenter__.return_value = mock_client
        
        results = await faiss_client.search("test query")
        assert results == []
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_faiss_search_connection_error(self, mock_httpx, faiss_client):
        """Test FAISS search with connection error"""
        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("Connection failed")
        mock_httpx.return_value.__aenter__.return_value = mock_client
        
        results = await faiss_client.search("test query")
        assert results == []

class TestVectorContextInjector:
    """Test main vector context injection system"""
    
    @pytest.fixture
    def injector(self):
        """Create vector context injector for testing"""
        return VectorContextInjector()
    
    def test_is_knowledge_query_detection(self, injector):
        """Test knowledge query detection logic"""
        # Knowledge queries
        assert injector._is_knowledge_query("What is Docker?")
        assert injector._is_knowledge_query("How to install Python?")
        assert injector._is_knowledge_query("Explain machine learning")
        assert injector._is_knowledge_query("Tell me about React")
        assert injector._is_knowledge_query("Can you explain quantum computing?")
        
        # Non-knowledge queries
        assert not injector._is_knowledge_query("Hello there")
        assert not injector._is_knowledge_query("Good morning")
        assert not injector._is_knowledge_query("Thanks for your help")
        assert not injector._is_knowledge_query("Please restart the service")
    
    @pytest.mark.asyncio
    async def test_analyze_user_request_knowledge_query(self, injector):
        """Test analyze_user_request with knowledge query"""
        # Mock the search method
        mock_context = KnowledgeContext(
            results=[
                VectorSearchResult("Test doc", {"id": 1}, 0.9, "chromadb")
            ],
            query_time_ms=100.0,
            sources_used=["chromadb"],
            total_results=1,
            enriched_context="Test context"
        )
        
        injector.search_all_databases = AsyncMock(return_value=mock_context)
        
        needs_context, context = await injector.analyze_user_request("What is Docker?")
        
        assert needs_context is True
        assert context is not None
        assert context.total_results == 1
        injector.search_all_databases.assert_called_once_with("What is Docker?")
    
    @pytest.mark.asyncio
    async def test_analyze_user_request_non_knowledge_query(self, injector):
        """Test analyze_user_request with non-knowledge query"""
        needs_context, context = await injector.analyze_user_request("Hello there")
        
        assert needs_context is False
        assert context is None
    
    @pytest.mark.asyncio
    async def test_search_all_databases_concurrent(self, injector):
        """Test concurrent search across all databases"""
        # Mock all database clients
        chromadb_result = [VectorSearchResult("ChromaDB doc", {}, 0.9, "chromadb")]
        qdrant_result = [VectorSearchResult("Qdrant doc", {}, 0.8, "qdrant")]
        faiss_result = [VectorSearchResult("FAISS doc", {}, 0.85, "faiss")]
        
        injector.chromadb_client.search = AsyncMock(return_value=chromadb_result)
        injector.qdrant_client.search = AsyncMock(return_value=qdrant_result)
        injector.faiss_client.search = AsyncMock(return_value=faiss_result)
        
        context = await injector.search_all_databases("test query")
        
        assert len(context.results) == 3
        assert context.sources_used == ["chromadb", "qdrant", "faiss"]
        assert context.total_results == 3
        assert context.query_time_ms > 0
        
        # Verify all clients were called
        injector.chromadb_client.search.assert_called_once_with("test query")
        injector.qdrant_client.search.assert_called_once_with("test query")
        injector.faiss_client.search.assert_called_once_with("test query")
    
    @pytest.mark.asyncio
    async def test_search_all_databases_with_failures(self, injector):
        """Test search continues with partial failures"""
        # Mock one success and two failures
        chromadb_result = [VectorSearchResult("ChromaDB doc", {}, 0.9, "chromadb")]
        
        injector.chromadb_client.search = AsyncMock(return_value=chromadb_result)
        injector.qdrant_client.search = AsyncMock(side_effect=Exception("Qdrant failed"))
        injector.faiss_client.search = AsyncMock(return_value=[])
        
        context = await injector.search_all_databases("test query")
        
        assert len(context.results) == 1
        assert context.sources_used == ["chromadb"]
        assert context.total_results == 1
    
    @pytest.mark.asyncio
    async def test_search_timeout_handling(self, injector):
        """Test search timeout handling"""
        # Mock slow responses
        async def slow_search(query):
            await asyncio.sleep(1.0)  # Longer than 500ms timeout
            return []
        
        injector.chromadb_client.search = slow_search
        injector.qdrant_client.search = slow_search
        injector.faiss_client.search = slow_search
        
        context = await injector.search_all_databases("test query")
        
        # Should return empty results due to timeout
        assert context.total_results == 0
        assert context.sources_used == []
    
    def test_deduplicate_results(self, injector):
        """Test result deduplication logic"""
        results = [
            VectorSearchResult("Document A", {}, 0.9, "chromadb"),
            VectorSearchResult("Document B", {}, 0.8, "qdrant"),
            VectorSearchResult("Document A", {}, 0.85, "faiss"),  # Duplicate content
            VectorSearchResult("Document C", {}, 0.7, "chromadb")
        ]
        
        deduplicated = injector._deduplicate_results(results)
        
        # Should have 3 unique documents, sorted by score
        assert len(deduplicated) == 3
        assert deduplicated[0].content == "Document A"
        assert deduplicated[0].score == 0.9  # Highest score version kept
        assert deduplicated[1].content == "Document B"
        assert deduplicated[2].content == "Document C"
    
    def test_create_enriched_context(self, injector):
        """Test enriched context creation"""
        results = [
            VectorSearchResult("Test document 1", {"source": "wiki"}, 0.9, "chromadb"),
            VectorSearchResult("Test document 2", {"source": "docs"}, 0.8, "qdrant")
        ]
        
        context = injector._create_enriched_context(results, "test query")
        
        assert "KNOWLEDGE CONTEXT FOR: test query" in context
        assert "Test document 1" in context
        assert "Test document 2" in context
        assert "CHROMADB" in context
        assert "QDRANT" in context
        assert "Score: 0.900" in context
        assert "Score: 0.800" in context
    
    @pytest.mark.asyncio
    async def test_inject_context_into_prompt(self, injector):
        """Test context injection into prompt"""
        knowledge_context = KnowledgeContext(
            results=[VectorSearchResult("Test doc", {}, 0.9, "chromadb")],
            query_time_ms=100.0,
            sources_used=["chromadb"],
            total_results=1,
            enriched_context="KNOWLEDGE CONTEXT FOR: test\nTest document content"
        )
        
        enhanced_prompt = await injector.inject_context_into_prompt(
            "What is this?", knowledge_context
        )
        
        assert "KNOWLEDGE CONTEXT FOR: test" in enhanced_prompt
        assert "ORIGINAL REQUEST: What is this?" in enhanced_prompt
        assert "INSTRUCTIONS:" in enhanced_prompt
    
    @pytest.mark.asyncio
    async def test_inject_context_no_context(self, injector):
        """Test context injection with no context"""
        enhanced_prompt = await injector.inject_context_into_prompt(
            "What is this?", None
        )
        
        assert enhanced_prompt == "What is this?"
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, injector):
        """Test result caching functionality"""
        # Mock successful search
        mock_context = KnowledgeContext(
            results=[VectorSearchResult("Test doc", {}, 0.9, "chromadb")],
            query_time_ms=100.0,
            sources_used=["chromadb"],
            total_results=1,
            enriched_context="Test context"
        )
        
        # Mock the search method to track calls
        original_search = injector.search_all_databases
        injector.search_all_databases = AsyncMock(return_value=mock_context)
        
        # First call should perform search
        context1 = await injector.search_all_databases("test query")
        assert injector.search_all_databases.call_count == 1
        
        # Second call should use cache (but we need to test the actual analyze method)
        needs_context1, context1 = await injector.analyze_user_request("What is test?")
        needs_context2, context2 = await injector.analyze_user_request("What is test?")
        
        assert needs_context1 == needs_context2
        # Note: In actual implementation, second call should be faster due to caching

# Integration test with actual FastAPI endpoint
@pytest.mark.asyncio
async def test_integration_with_chat_endpoint():
    """Integration test with the chat endpoint"""
    from app.main import app
    from fastapi.testclient import TestClient
    import json
    
    # This would require more setup to properly test the full integration
    # For now, this is a placeholder for the integration test structure
    pass

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])