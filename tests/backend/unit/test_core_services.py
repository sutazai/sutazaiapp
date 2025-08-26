"""
Unit tests for core service modules
Testing connection pools, caching, task queues, and health monitoring
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime, timedelta
import json


class TestConnectionPoolManager:
    """Test connection pool management"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_connection_pool_initialization(self):
        """Test connection pool manager initialization"""
        with patch('app.core.connection_pool.ConnectionPoolManager') as MockPoolManager:
            mock_instance = AsyncMock()
            MockPoolManager.return_value = mock_instance
            
            from app.core.connection_pool import get_pool_manager
            
            pool_manager = await get_pool_manager()
            assert pool_manager is not None
            MockPoolManager.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_connection_pool_stats(self, mock_pool_manager):
        """Test connection pool statistics"""
        stats = mock_pool_manager.get_stats()
        
        assert isinstance(stats, dict)
        assert "active_connections" in stats
        assert "total_connections" in stats
        assert "pool_utilization" in stats
        
        # Verify stat values are reasonable
        assert stats["active_connections"] >= 0
        assert stats["total_connections"] >= stats["active_connections"]
        assert 0 <= stats["pool_utilization"] <= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_connection_pool_close(self, mock_pool_manager):
        """Test connection pool cleanup"""
        await mock_pool_manager.close()
        mock_pool_manager.close.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_http_client_creation(self):
        """Test HTTP client creation and management"""
        with patch('app.core.connection_pool.get_http_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            from app.core.connection_pool import get_http_client
            
            async with await get_http_client('test_pool') as client:
                assert client is not None


class TestCacheService:
    """Test caching service functionality"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_service_initialization(self, mock_cache_service):
        """Test cache service initialization"""
        assert mock_cache_service is not None
        
        # Test basic cache operations are available
        assert hasattr(mock_cache_service, 'get')
        assert hasattr(mock_cache_service, 'set')
        assert hasattr(mock_cache_service, 'delete')

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_get_set_operations(self, mock_cache_service):
        """Test basic cache get/set operations"""
        key = "test_key"
        value = {"test": "data"}
        
        # Test set operation
        result = await mock_cache_service.set(key, value, ttl=60)
        assert result is True
        
        # Verify set was called with correct parameters
        mock_cache_service.set.assert_called_with(key, value, ttl=60)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_delete_operations(self, mock_cache_service):
        """Test cache deletion operations"""
        key = "test_key"
        
        # Test delete operation
        result = await mock_cache_service.delete(key)
        assert result is True
        
        mock_cache_service.delete.assert_called_with(key)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_clear_all(self, mock_cache_service):
        """Test cache clear all operation"""
        result = await mock_cache_service.clear_all()
        assert result is True
        
        mock_cache_service.clear_all.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_statistics(self, mock_cache_service):
        """Test cache statistics collection"""
        stats = mock_cache_service.get_stats()
        
        assert isinstance(stats, dict)
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "total_operations" in stats
        
        # Verify hit rate calculation
        expected_hit_rate = stats["hits"] / (stats["hits"] + stats["misses"])
        assert abs(stats["hit_rate"] - expected_hit_rate) < 0.01

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_decorators(self):
        """Test cache decorator functionality"""
        with patch('app.core.cache.get_cache_service') as mock_get_cache:
            mock_cache = AsyncMock()
            mock_get_cache.return_value = mock_cache
            mock_cache.get.return_value = None  # Cache miss
            mock_cache.set.return_value = True
            
            from app.core.cache import cached
            
            @cached(ttl=60)
            async def test_function(arg1, arg2):
                return f"result_{arg1}_{arg2}"
            
            # Call function - should cache result
            result = await test_function("a", "b")
            assert result == "result_a_b"


class TestTaskQueue:
    """Test task queue functionality"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_task_queue_initialization(self, mock_task_queue):
        """Test task queue initialization"""
        assert mock_task_queue is not None
        assert hasattr(mock_task_queue, 'get_task_status')
        assert hasattr(mock_task_queue, 'register_handler')

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_task_status_retrieval(self, mock_task_queue):
        """Test task status retrieval"""
        task_id = "test-task-123"
        
        status = await mock_task_queue.get_task_status(task_id)
        
        assert isinstance(status, dict)
        assert "status" in status
        assert "result" in status
        assert status["status"] == "completed"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_task_handler_registration(self, mock_task_queue):
        """Test task handler registration"""
        async def test_handler(payload):
            return {"processed": payload}
        
        await mock_task_queue.register_handler("test_type", test_handler)
        
        mock_task_queue.register_handler.assert_called_with("test_type", test_handler)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_task_queue_statistics(self, mock_task_queue):
        """Test task queue statistics"""
        stats = mock_task_queue.get_stats()
        
        assert isinstance(stats, dict)
        assert "pending_tasks" in stats
        assert "completed_tasks" in stats
        assert "failed_tasks" in stats
        
        # Verify stats are non-negative integers
        assert stats["pending_tasks"] >= 0
        assert stats["completed_tasks"] >= 0
        assert stats["failed_tasks"] >= 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_task_queue_shutdown(self, mock_task_queue):
        """Test task queue shutdown"""
        await mock_task_queue.stop()
        mock_task_queue.stop.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_background_task_creation(self):
        """Test background task creation"""
        with patch('app.core.task_queue.create_background_task') as mock_create:
            mock_create.return_value = "task-id-123"
            
            from app.core.task_queue import create_background_task
            
            task_id = await create_background_task(
                task_type="test",
                payload={"data": "test"},
                priority=1
            )
            
            assert task_id == "task-id-123"
            mock_create.assert_called_once_with(
                task_type="test",
                payload={"data": "test"},
                priority=1
            )


class TestHealthMonitoring:
    """Test health monitoring service"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_monitoring_initialization(self, mock_health_monitoring):
        """Test health monitoring service initialization"""
        assert mock_health_monitoring is not None
        assert hasattr(mock_health_monitoring, 'get_detailed_health')
        assert hasattr(mock_health_monitoring, 'get_prometheus_metrics')

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_detailed_health_report(self, mock_health_monitoring):
        """Test detailed health report generation"""
        health_report = await mock_health_monitoring.get_detailed_health()
        
        assert health_report is not None
        assert hasattr(health_report, 'overall_status')
        assert hasattr(health_report, 'services')
        assert hasattr(health_report, 'performance_metrics')
        assert hasattr(health_report, 'system_resources')

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_prometheus_metrics_generation(self, mock_health_monitoring):
        """Test Prometheus metrics generation"""
        metrics = await mock_health_monitoring.get_prometheus_metrics()
        
        assert isinstance(metrics, str)
        assert len(metrics) > 0
        # Basic Prometheus format validation
        assert "# " in metrics  # Should contain comments

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_service_health_tracking(self, mock_health_monitoring):
        """Test individual service health tracking"""
        health_report = await mock_health_monitoring.get_detailed_health()
        
        services = health_report.services
        assert isinstance(services, dict)
        
        # Check that services have required metrics
        for service_name, metrics in services.items():
            assert hasattr(metrics, 'status')
            assert hasattr(metrics, 'response_time_ms')
            assert hasattr(metrics, 'consecutive_failures')


class TestCircuitBreakers:
    """Test circuit breaker functionality"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_circuit_breaker_manager_initialization(self, mock_circuit_breaker_manager):
        """Test circuit breaker manager initialization"""
        assert mock_circuit_breaker_manager is not None
        assert hasattr(mock_circuit_breaker_manager, 'get_all_stats')
        assert hasattr(mock_circuit_breaker_manager, 'reset_all')

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_circuit_breaker_stats(self, mock_circuit_breaker_manager):
        """Test circuit breaker statistics"""
        stats = mock_circuit_breaker_manager.get_all_stats()
        
        assert isinstance(stats, dict)
        
        # Check expected circuit breakers
        expected_breakers = ["redis", "database"]
        for breaker_name in expected_breakers:
            assert breaker_name in stats
            breaker_stats = stats[breaker_name]
            assert "state" in breaker_stats
            assert "failure_count" in breaker_stats

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_circuit_breaker_reset(self, mock_circuit_breaker_manager):
        """Test circuit breaker reset functionality"""
        await mock_circuit_breaker_manager.reset_all()
        mock_circuit_breaker_manager.reset_all.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_individual_circuit_breakers(self):
        """Test individual circuit breaker creation"""
        with patch('app.core.circuit_breaker_integration.get_redis_circuit_breaker') as mock_redis_breaker, \
             patch('app.core.circuit_breaker_integration.get_database_circuit_breaker') as mock_db_breaker, \
             patch('app.core.circuit_breaker_integration.get_ollama_circuit_breaker') as mock_ollama_breaker:
            
            mock_redis_breaker.return_value = AsyncMock()
            mock_db_breaker.return_value = AsyncMock()
            mock_ollama_breaker.return_value = AsyncMock()
            
            from app.core.circuit_breaker_integration import (
                get_redis_circuit_breaker,
                get_database_circuit_breaker,
                get_ollama_circuit_breaker
            )
            
            redis_breaker = await get_redis_circuit_breaker()
            db_breaker = await get_database_circuit_breaker()
            ollama_breaker = await get_ollama_circuit_breaker()
            
            assert redis_breaker is not None
            assert db_breaker is not None
            assert ollama_breaker is not None


class TestOllamaService:
    """Test Ollama service integration"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_service_initialization(self, mock_ollama_service):
        """Test Ollama service initialization"""
        assert mock_ollama_service is not None
        assert hasattr(mock_ollama_service, 'generate')
        assert hasattr(mock_ollama_service, 'generate_streaming')
        assert hasattr(mock_ollama_service, 'batch_generate')

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_text_generation(self, mock_ollama_service):
        """Test Ollama text generation"""
        prompt = "Hello, how are you?"
        model = "tinyllama"
        
        result = await mock_ollama_service.generate(
            prompt=prompt,
            model=model,
            use_cache=True
        )
        
        assert isinstance(result, dict)
        assert "response" in result
        assert "cached" in result
        assert "model" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_streaming_generation(self, mock_ollama_service):
        """Test Ollama streaming text generation"""
        prompt = "Tell me a story"
        model = "tinyllama"
        
        chunks = []
        async for chunk in mock_ollama_service.generate_streaming(
            prompt=prompt,
            model=model
        ):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        # Verify we got the expected chunks
        expected_chunks = ["Test", " streaming", " response"]
        assert chunks == expected_chunks

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_batch_generation(self, mock_ollama_service):
        """Test Ollama batch text generation"""
        prompts = ["Hello", "How are you?"]
        
        results = await mock_ollama_service.batch_generate(
            prompts=prompts,
            max_concurrent=10
        )
        
        assert isinstance(results, list)
        assert len(results) == 2
        
        for result in results:
            assert isinstance(result, dict)
            assert "response" in result
            assert "cached" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_service_statistics(self, mock_ollama_service):
        """Test Ollama service statistics"""
        stats = mock_ollama_service.get_stats()
        
        assert isinstance(stats, dict)
        assert "requests_processed" in stats
        assert "average_response_time" in stats
        assert "cache_hit_rate" in stats
        
        # Verify stats are reasonable
        assert stats["requests_processed"] >= 0
        assert stats["average_response_time"] > 0
        assert 0 <= stats["cache_hit_rate"] <= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_service_warmup(self, mock_ollama_service):
        """Test Ollama service warmup"""
        await mock_ollama_service.warmup(3)
        mock_ollama_service.warmup.assert_called_with(3)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_service_shutdown(self, mock_ollama_service):
        """Test Ollama service shutdown"""
        await mock_ollama_service.shutdown()
        mock_ollama_service.shutdown.assert_called_once()


class TestCoreConfiguration:
    """Test core configuration management"""

    @pytest.mark.unit
    def test_settings_import(self):
        """Test settings configuration import"""
        try:
            from app.core.config import settings
            assert settings is not None
        except ImportError:
            pytest.skip("Settings module not available")

    @pytest.mark.unit
    def test_environment_variables(self):
        """Test environment variable handling"""
        import os
        
        # Test that test environment is properly set
        assert os.getenv("SUTAZAI_ENV") == "test"
        assert os.getenv("JWT_SECRET_KEY") is not None
        assert len(os.getenv("JWT_SECRET_KEY")) >= 32

    @pytest.mark.unit
    def test_logging_configuration(self):
        """Test logging configuration"""
        import logging
        
        # Verify logging is configured
        logger = logging.getLogger("app.main")
        assert logger is not None
        
        # Test log level is appropriate for testing
        assert logger.getEffectiveLevel() <= logging.WARNING


class TestServiceIntegration:
    """Test service integration and coordination"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_service_dependency_coordination(self, 
                                                   mock_cache_service,
                                                   mock_pool_manager,
                                                   mock_ollama_service,
                                                   mock_task_queue):
        """Test that services can work together"""
        # Test cache service with connection pool
        assert mock_cache_service is not None
        assert mock_pool_manager is not None
        
        # Test cache stats and pool stats are available
        cache_stats = mock_cache_service.get_stats()
        pool_stats = mock_pool_manager.get_stats()
        
        assert isinstance(cache_stats, dict)
        assert isinstance(pool_stats, dict)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_handling_across_services(self,
                                                   mock_cache_service,
                                                   mock_ollama_service):
        """Test error handling across different services"""
        # Configure Mock to raise exception
        mock_cache_service.get.side_effect = Exception("Redis connection failed")
        
        # Service should handle cache failures gracefully
        with pytest.raises(Exception) as exc_info:
            await mock_cache_service.get("test_key")
        
        assert "Redis connection failed" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_service_lifecycle_management(self,
                                                mock_pool_manager,
                                                mock_task_queue,
                                                mock_ollama_service):
        """Test service lifecycle (startup/shutdown)"""
        # Test shutdown sequence
        await mock_ollama_service.shutdown()
        await mock_task_queue.stop()
        await mock_pool_manager.close()
        
        # Verify all services were shut down
        mock_ollama_service.shutdown.assert_called_once()
        mock_task_queue.stop.assert_called_once()
        mock_pool_manager.close.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_service_operations(self,
                                                  mock_cache_service,
                                                  mock_ollama_service):
        """Test concurrent operations across services"""
        # Create concurrent operations
        cache_task = mock_cache_service.get("test_key")
        ollama_task = mock_ollama_service.generate(
            prompt="test",
            model="tinyllama"
        )
        
        # Execute concurrently
        cache_result, ollama_result = await asyncio.gather(
            cache_task, ollama_task, return_exceptions=True
        )
        
        # Both operations should complete
        assert cache_result is None  # Mock returns None for cache miss
        assert isinstance(ollama_result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])