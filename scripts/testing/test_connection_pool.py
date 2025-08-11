#!/usr/bin/env python3
"""
Comprehensive unit tests for OllamaConnectionPool
Tests connection management, pooling, model caching, and resource optimization
"""

import pytest
import asyncio
import httpx
import time
import sys
import os
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add the agents directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents'))

from core.ollama_pool import (
    OllamaConnectionPool, PoolConnection, ModelCache, PoolStats,
    ConnectionState, create_ollama_pool
)


class TestOllamaConnectionPool:
    """Test suite for OllamaConnectionPool class"""
    
    @pytest.fixture
    def pool_config(self):
        """Standard pool configuration for testing"""
        return {
            "base_url": "http://test-ollama:10104",
            "max_connections": 2,
            "min_connections": 1,
            "default_model": "tinyllama",
            "connection_timeout": 10,
            "request_timeout": 30,
            "cleanup_interval": 60,
            "model_cache_size": 3
        }
    
    @pytest.fixture
    def pool(self, pool_config):
        """Create OllamaConnectionPool instance for testing"""
        return OllamaConnectionPool(**pool_config)
    
    def test_initialization(self, pool):
        """Test pool initialization"""
        assert pool.base_url == "http://test-ollama:10104"
        assert pool.max_connections == 2
        assert pool.min_connections == 1
        assert pool.default_model == "tinyllama"
        assert pool.connection_timeout == 10
        assert pool.request_timeout == 30
        assert len(pool._connections) == 0
        assert isinstance(pool.stats, PoolStats)
    
    @pytest.mark.asyncio
    async def test_create_connection_success(self, pool):
        """Test successful connection creation"""
        # Mock httpx client and successful health check
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        
        with patch('httpx.AsyncClient', return_value=mock_client):
            connection = await pool._create_connection()
            
            assert isinstance(connection, PoolConnection)
            assert connection.client == mock_client
            assert connection.state == ConnectionState.IDLE
            assert connection.error_count == 0
            assert len(pool._connections) == 1
            assert pool.stats.total_connections == 1
    
    @pytest.mark.asyncio
    async def test_create_connection_health_check_failure(self, pool):
        """Test connection creation with health check failure"""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 500
        mock_client.get.return_value = mock_response
        
        with patch('httpx.AsyncClient', return_value=mock_client):
            with pytest.raises(Exception):
                await pool._create_connection()
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self, pool):
        """Test successful connection testing"""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        
        connection = PoolConnection(client=mock_client)
        
        await pool._test_connection(connection)
        
        assert connection.state == ConnectionState.IDLE
        assert connection.error_count == 0
    
    @pytest.mark.asyncio
    async def test_test_connection_failure(self, pool):
        """Test connection testing failure"""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection failed")
        
        connection = PoolConnection(client=mock_client)
        
        with pytest.raises(Exception):
            await pool._test_connection(connection)
        
        assert connection.state == ConnectionState.ERROR
        assert connection.error_count == 1
    
    @pytest.mark.asyncio
    async def test_get_connection_idle_available(self, pool):
        """Test getting connection when idle connection is available"""
        # Create a mock connection
        mock_client = AsyncMock()
        connection = PoolConnection(client=mock_client, state=ConnectionState.IDLE)
        pool._connections.append(connection)
        
        async with pool._get_connection() as conn:
            assert conn == connection
            assert conn.state == ConnectionState.BUSY
            assert pool.stats.pool_hits == 1
        
        assert connection.state == ConnectionState.IDLE
    
    @pytest.mark.asyncio
    async def test_get_connection_create_new(self, pool):
        """Test getting connection when no idle connections available"""
        # Mock connection creation
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        
        with patch('httpx.AsyncClient', return_value=mock_client):
            async with pool._get_connection() as conn:
                assert isinstance(conn, PoolConnection)
                assert conn.state == ConnectionState.BUSY
                assert pool.stats.pool_misses == 1
        
        assert conn.state == ConnectionState.IDLE
    
    @pytest.mark.asyncio
    async def test_get_connection_at_max_capacity(self, pool):
        """Test getting connection when at max capacity"""
        # Fill pool to capacity with busy connections
        for _ in range(pool.max_connections):
            mock_client = AsyncMock()
            connection = PoolConnection(client=mock_client, state=ConnectionState.BUSY)
            pool._connections.append(connection)
        
        # Should raise exception when no connections available
        with pytest.raises(Exception, match="No connections available"):
            async with pool._get_connection():
                pass
    
    @pytest.mark.asyncio
    async def test_refresh_available_models_success(self, pool):
        """Test successful model list refresh"""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "tinyllama:latest"},
                {"name": "tinyllama2.5-coder:7b"}
            ]
        }
        
        # Create a connection to use
        connection = PoolConnection(client=mock_client)
        pool._connections.append(connection)
        
        await pool._refresh_available_models()
        
        assert pool._available_models == {"tinyllama", "tinyllama2.5-coder"}
        assert pool.stats.models_cached == 2
    
    @pytest.mark.asyncio
    async def test_refresh_available_models_failure(self, pool):
        """Test model list refresh failure"""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 500
        
        connection = PoolConnection(client=mock_client)
        pool._connections.append(connection)
        
        await pool._refresh_available_models()
        
        # Should handle error gracefully
        assert pool._available_models is None
    
    @pytest.mark.asyncio
    async def test_ensure_model_available_cached(self, pool):
        """Test model availability check with cached model"""
        model_name = "tinyllama"
        
        # Add model to cache
        cache_entry = ModelCache(model_name=model_name, is_warmed=True)
        pool._model_cache[model_name] = cache_entry
        
        result = await pool._ensure_model_available(model_name)
        assert result is True
        
        # Should update last_used
        assert cache_entry.last_used > datetime.utcnow() - timedelta(seconds=1)
    
    @pytest.mark.asyncio
    async def test_ensure_model_available_in_registry(self, pool):
        """Test model availability check with model in registry"""
        model_name = "tinyllama"
        
        # Mock available models
        pool._available_models = {"tinyllama", "other-model"}
        pool._models_last_checked = datetime.utcnow()
        
        result = await pool._ensure_model_available(model_name)
        assert result is True
        assert model_name in pool._model_cache
    
    @pytest.mark.asyncio
    async def test_ensure_model_available_needs_pull(self, pool):
        """Test model availability when model needs to be pulled"""
        model_name = "new-model"
        
        # Mock model not in available list
        pool._available_models = {"tinyllama"}
        pool._models_last_checked = datetime.utcnow()
        
        # Mock successful pull
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        
        connection = PoolConnection(client=mock_client)
        pool._connections.append(connection)
        
        with patch.object(mock_client, 'post', return_value=mock_response):
            result = await pool._ensure_model_available(model_name)
            assert result is True
            assert model_name in pool._available_models
            assert model_name in pool._model_cache
    
    @pytest.mark.asyncio
    async def test_pull_model_success(self, pool):
        """Test successful model pull"""
        model_name = "new-model"
        
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        
        connection = PoolConnection(client=mock_client)
        pool._connections.append(connection)
        
        with patch.object(mock_client, 'post', return_value=mock_response):
            result = await pool._pull_model(model_name)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_pull_model_failure(self, pool):
        """Test failed model pull"""
        model_name = "invalid-model"
        
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 404
        
        connection = PoolConnection(client=mock_client)
        pool._connections.append(connection)
        
        with patch.object(mock_client, 'post', return_value=mock_response):
            result = await pool._pull_model(model_name)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_warm_model_success(self, pool):
        """Test successful model warming"""
        model_name = "tinyllama"
        
        # Mock model availability and generation
        with patch.object(pool, '_ensure_model_available', return_value=True), \
             patch.object(pool, 'generate', return_value="warmup response"):
            
            await pool._warm_model(model_name)
            
            # Should mark model as warmed
            assert model_name in pool._model_cache
            assert pool._model_cache[model_name].is_warmed is True
    
    @pytest.mark.asyncio
    async def test_generate_success(self, pool):
        """Test successful text generation"""
        prompt = "Test prompt"
        model = "tinyllama"
        
        # Mock model availability
        with patch.object(pool, '_ensure_model_available', return_value=True):
            # Mock connection and response
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "Generated text"}
            
            connection = PoolConnection(client=mock_client)
            pool._connections.append(connection)
            
            with patch.object(mock_client, 'post', return_value=mock_response):
                result = await pool.generate(prompt, model)
                
                assert result == "Generated text"
                assert pool.stats.total_requests == 1
                assert pool.stats.failed_requests == 0
    
    @pytest.mark.asyncio
    async def test_generate_model_unavailable(self, pool):
        """Test generation when model is unavailable"""
        with patch.object(pool, '_ensure_model_available', return_value=False):
            result = await pool.generate("Test prompt")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_generate_api_error(self, pool):
        """Test generation with API error"""
        with patch.object(pool, '_ensure_model_available', return_value=True):
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            
            connection = PoolConnection(client=mock_client)
            pool._connections.append(connection)
            
            with patch.object(mock_client, 'post', return_value=mock_response):
                result = await pool.generate("Test prompt")
                
                assert result is None
                assert pool.stats.failed_requests == 1
    
    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, pool):
        """Test generation with system prompt"""
        with patch.object(pool, '_ensure_model_available', return_value=True):
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "Response with system"}
            
            connection = PoolConnection(client=mock_client)
            pool._connections.append(connection)
            
            with patch.object(mock_client, 'post', return_value=mock_response) as mock_post:
                await pool.generate(
                    prompt="User prompt",
                    system="You are a helpful assistant"
                )
                
                # Verify system prompt was included
                call_args = mock_post.call_args
                payload = call_args[1]['json']
                assert payload['system'] == "You are a helpful assistant"
    
    @pytest.mark.asyncio
    async def test_chat_success(self, pool):
        """Test successful chat completion"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(pool, '_ensure_model_available', return_value=True):
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "message": {"content": "Hello! How can I help?"}
            }
            
            connection = PoolConnection(client=mock_client)
            pool._connections.append(connection)
            
            with patch.object(mock_client, 'post', return_value=mock_response):
                result = await pool.chat(messages)
                
                assert result == "Hello! How can I help?"
                assert pool.stats.total_requests == 1
    
    @pytest.mark.asyncio
    async def test_chat_api_error(self, pool):
        """Test chat with API error"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(pool, '_ensure_model_available', return_value=True):
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 400
            
            connection = PoolConnection(client=mock_client)
            pool._connections.append(connection)
            
            with patch.object(mock_client, 'post', return_value=mock_response):
                result = await pool.chat(messages)
                
                assert result is None
                assert pool.stats.failed_requests == 1
    
    @pytest.mark.asyncio
    async def test_embeddings_success(self, pool):
        """Test successful embeddings generation"""
        text = "Test text for embeddings"
        
        with patch.object(pool, '_ensure_model_available', return_value=True):
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "embedding": [0.1, 0.2, 0.3, 0.4]
            }
            
            connection = PoolConnection(client=mock_client)
            pool._connections.append(connection)
            
            with patch.object(mock_client, 'post', return_value=mock_response):
                result = await pool.embeddings(text)
                
                assert result == [0.1, 0.2, 0.3, 0.4]
                assert pool.stats.total_requests == 1
    
    @pytest.mark.asyncio
    async def test_embeddings_api_error(self, pool):
        """Test embeddings with API error"""
        with patch.object(pool, '_ensure_model_available', return_value=True):
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 500
            
            connection = PoolConnection(client=mock_client)
            pool._connections.append(connection)
            
            with patch.object(mock_client, 'post', return_value=mock_response):
                result = await pool.embeddings("Test text")
                
                assert result is None
                assert pool.stats.failed_requests == 1
    
    def test_update_average_response_time(self, pool):
        """Test average response time calculation"""
        # First request
        pool.stats.total_requests = 1
        pool._update_average_response_time(2.0)
        assert pool.stats.average_response_time == 2.0
        
        # Second request
        pool.stats.total_requests = 2
        pool._update_average_response_time(4.0)
        assert pool.stats.average_response_time == 3.0  # (2.0 + 4.0) / 2
    
    @pytest.mark.asyncio
    async def test_cleanup_connections(self, pool):
        """Test connection cleanup"""
        # Add connections with different states and ages
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()
        mock_client3 = AsyncMock()
        
        # Error connection - should be removed
        error_conn = PoolConnection(client=mock_client1, state=ConnectionState.ERROR)
        
        # Old idle connection - should be removed if above minimum
        old_idle_conn = PoolConnection(
            client=mock_client2,
            state=ConnectionState.IDLE,
            last_used=datetime.utcnow() - timedelta(hours=1)
        )
        
        # Recent idle connection - should be kept
        recent_idle_conn = PoolConnection(
            client=mock_client3,
            state=ConnectionState.IDLE
        )
        
        pool._connections = [error_conn, old_idle_conn, recent_idle_conn]
        pool.stats.total_connections = 3
        
        await pool._cleanup_connections()
        
        # Should remove error connection and old idle connection
        assert len(pool._connections) == 1
        assert pool._connections[0] == recent_idle_conn
    
    @pytest.mark.asyncio
    async def test_cleanup_model_cache(self, pool):
        """Test model cache cleanup"""
        # Add more models than cache size allows
        for i in range(pool.model_cache_size + 2):
            model_name = f"model-{i}"
            cache_entry = ModelCache(
                model_name=model_name,
                last_used=datetime.utcnow() - timedelta(minutes=i)
            )
            pool._model_cache[model_name] = cache_entry
        
        await pool._cleanup_model_cache()
        
        # Should keep only the most recently used models
        assert len(pool._model_cache) == pool.model_cache_size
        
        # Should keep the most recently used models
        remaining_models = list(pool._model_cache.keys())
        assert "model-0" in remaining_models  # Most recent
        assert f"model-{pool.model_cache_size + 1}" not in remaining_models  # Oldest
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, pool):
        """Test successful health check"""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        
        connection = PoolConnection(client=mock_client)
        pool._connections.append(connection)
        
        with patch.object(mock_client, 'get', return_value=mock_response):
            result = await pool.health_check()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, pool):
        """Test health check failure"""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection failed")
        
        connection = PoolConnection(client=mock_client)
        pool._connections.append(connection)
        
        result = await pool.health_check()
        assert result is False
    
    def test_get_stats(self, pool):
        """Test statistics retrieval"""
        # Set up some test statistics
        pool.stats.total_requests = 100
        pool.stats.failed_requests = 5
        pool.stats.pool_hits = 80
        pool.stats.pool_misses = 20
        pool.stats.average_response_time = 1.5
        pool._model_cache = {"model1": Mock(), "model2": Mock()}
        pool._available_models = {"model1", "model2", "model3"}
        
        stats = pool.get_stats()
        
        assert stats["total_requests"] == 100
        assert stats["failed_requests"] == 5
        assert stats["success_rate"] == 0.95  # (100-5)/100
        assert stats["pool_hits"] == 80
        assert stats["pool_misses"] == 20
        assert stats["hit_rate"] == 0.8  # 80/(80+20)
        assert stats["average_response_time"] == 1.5
        assert stats["models_cached"] == 2
        assert len(stats["available_models"]) == 3
    
    @pytest.mark.asyncio
    async def test_context_manager(self, pool_config):
        """Test async context manager functionality"""
        async with OllamaConnectionPool(**pool_config) as pool:
            assert isinstance(pool, OllamaConnectionPool)
            # Pool should be initialized
            assert pool._cleanup_task is not None
            assert pool._health_task is not None
    
    @pytest.mark.asyncio
    async def test_close(self, pool):
        """Test pool closure"""
        # Set up some connections
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()
        
        connection1 = PoolConnection(client=mock_client1)
        connection2 = PoolConnection(client=mock_client2)
        
        pool._connections = [connection1, connection2]
        
        # Mock background tasks
        pool._cleanup_task = AsyncMock()
        pool._health_task = AsyncMock()
        
        await pool.close()
        
        # Should cancel background tasks
        pool._cleanup_task.cancel.assert_called_once()
        pool._health_task.cancel.assert_called_once()
        
        # Should close all connections
        mock_client1.aclose.assert_called_once()
        mock_client2.aclose.assert_called_once()
        
        # Should clear connections list
        assert len(pool._connections) == 0
        
        # Should set shutdown event
        assert pool._shutdown_event.is_set()


class TestPoolConnection:
    """Test PoolConnection dataclass"""
    
    def test_pool_connection_creation(self):
        """Test PoolConnection creation"""
        mock_client = AsyncMock()
        connection = PoolConnection(client=mock_client)
        
        assert connection.client == mock_client
        assert connection.state == ConnectionState.IDLE
        assert connection.request_count == 0
        assert connection.error_count == 0
        assert connection.current_model is None
        assert isinstance(connection.created_at, datetime)
        assert isinstance(connection.last_used, datetime)


class TestModelCache:
    """Test ModelCache dataclass"""
    
    def test_model_cache_creation(self):
        """Test ModelCache creation"""
        cache = ModelCache(model_name="test-model")
        
        assert cache.model_name == "test-model"
        assert cache.request_count == 0
        assert cache.average_response_time == 0.0
        assert cache.is_warmed is False
        assert isinstance(cache.last_used, datetime)


class TestPoolStats:
    """Test PoolStats dataclass"""
    
    def test_pool_stats_defaults(self):
        """Test PoolStats default values"""
        stats = PoolStats()
        
        assert stats.total_connections == 0
        assert stats.active_connections == 0
        assert stats.idle_connections == 0
        assert stats.error_connections == 0
        assert stats.total_requests == 0
        assert stats.failed_requests == 0
        assert stats.average_response_time == 0.0
        assert stats.models_cached == 0
        assert stats.pool_hits == 0
        assert stats.pool_misses == 0


class TestConnectionState:
    """Test ConnectionState enum"""
    
    def test_connection_state_values(self):
        """Test ConnectionState enum values"""
        assert ConnectionState.IDLE.value == "idle"
        assert ConnectionState.BUSY.value == "busy"
        assert ConnectionState.ERROR.value == "error"
        assert ConnectionState.CLOSED.value == "closed"


class TestFactoryFunction:
    """Test factory function"""
    
    def test_create_ollama_pool(self):
        """Test create_ollama_pool factory function"""
        pool = create_ollama_pool(
            max_connections=3,
            default_model="custom-model",
            connection_timeout=15
        )
        
        assert isinstance(pool, OllamaConnectionPool)
        assert pool.max_connections == 3
        assert pool.default_model == "custom-model"
        assert pool.connection_timeout == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])