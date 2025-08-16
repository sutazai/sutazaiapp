#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Backend Core Components
Follows Rules 1-19 for professional test implementation
"""

import pytest
import asyncio
import os
import sys
from unittest.Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test import Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test, AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test, patch, MagicRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test
from datetime import datetime, timedelta
import json
import uuid

# Add backend to path for imports
# Path handled by pytest configuration, '..', '..', 'backend'))

# Test backend core modules
from app.core.config import settings
from app.core.cache import get_cache_service, cached
from app.core.connection_pool import get_pool_manager
from app.core.task_queue import get_task_queue, create_background_task
from app.core.database import get_database_connection
from app.core.security import SecurityManager
from app.core.metrics import MetricsCollector


@pytest.mark.unit
class TestBackendConfig:
    """Test configuration management"""
    
    def test_settings_default_values(self):
        """Test default configuration values are properly set"""
        assert settings.DEFAULT_MODEL == "tinyllama"
        assert settings.CORS_ORIGINS is not None
        assert settings.DATABASE_URL is not None
        assert settings.REDIS_URL is not None
        
    def test_environment_variable_override(self):
        """Test environment variables override defaults"""
        with patch.dict(os.environ, {'DEFAULT_MODEL': 'test-model'}):
            # Re-import to get new env vars
            from app.core.config import Settings
            test_settings = Settings()
            assert test_settings.DEFAULT_MODEL == 'test-model'
    
    def test_invalid_database_url_handling(self):
        """Test handling of invalid database URLs"""
        with patch.dict(os.environ, {'DATABASE_URL': 'invalid-url'}):
            with pytest.raises(Exception):
                from app.core.config import Settings
                Settings()


@pytest.mark.unit
class TestCacheService:
    """Test Redis cache service functionality"""
    
    @pytest.mark.asyncio
    async def test_cache_service_initialization(self):
        """Test cache service initializes correctly"""
        with patch('app.core.cache.Redis') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis:
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.return_value.ping = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=True)
            cache_service = await get_cache_service()
            assert cache_service is not None
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_set_get_operations(self):
        """Test basic cache set and get operations"""
        with patch('app.core.cache.Redis') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis:
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_instance = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_instance.set = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=True)
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_instance.get = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=b'test_value')
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_instance
            
            cache_service = await get_cache_service()
            
            # Test set operation
            await cache_service.set('test_key', 'test_value')
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_instance.set.assert_called_once()
            
            # Test get operation
            value = await cache_service.get('test_key')
            assert value == b'test_value'
    
    @pytest.mark.asyncio
    async def test_cached_decorator(self):
        """Test the @cached decorator functionality"""
        call_count = 0
        
        @cached(ttl=300, key_prefix='test')
        async def test_function(arg1, arg2):
            nonlocal call_count
            call_count += 1
            return f"{arg1}_{arg2}"
        
        with patch('app.core.cache.get_cache_service') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_cache:
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_cache = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_cache.get = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=None)  # Cache miss
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_cache.set = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_cache.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_cache
            
            # First call should execute function
            result1 = await test_function('hello', 'world')
            assert result1 == 'hello_world'
            assert call_count == 1
            
            # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test cache hit for second call
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_cache.get = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=b'"cached_result"')
            result2 = await test_function('hello', 'world')
            assert call_count == 1  # Function not called again
    
    @pytest.mark.asyncio
    async def test_cache_error_handling(self):
        """Test cache service handles Redis errors gracefully"""
        with patch('app.core.cache.Redis') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis:
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_instance = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_instance.set = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(side_effect=Exception('Redis error'))
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_instance
            
            cache_service = await get_cache_service()
            
            # Should not raise exception, just log and continue
            result = await cache_service.set('key', 'value', ignore_errors=True)
            assert result is False


@pytest.mark.unit
class TestConnectionPool:
    """Test connection pool management"""
    
    @pytest.mark.asyncio
    async def test_pool_manager_initialization(self):
        """Test connection pool manager initializes correctly"""
        with patch('app.core.connection_pool.aiohttp.TCPConnector') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connector:
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connector.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            pool_manager = await get_pool_manager()
            assert pool_manager is not None
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connector.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_http_client_connection_reuse(self):
        """Test HTTP client reuses connections"""
        with patch('app.core.connection_pool.aiohttp.ClientSession') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_session:
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_session_instance = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_session.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_session_instance
            
            from app.core.connection_pool import get_http_client
            client1 = await get_http_client()
            client2 = await get_http_client()
            
            # Should return the same client instance (singleton)
            assert client1 is client2
    
    @pytest.mark.asyncio
    async def test_pool_connection_limits(self):
        """Test connection pool respects connection limits"""
        with patch('app.core.connection_pool.aiohttp.TCPConnector') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connector:
            await get_pool_manager()
            
            # Verify connector called with proper limits
            call_args = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connector.call_args
            assert 'limit' in call_args.kwargs
            assert 'limit_per_host' in call_args.kwargs


@pytest.mark.unit
class TestTaskQueue:
    """Test background task queue functionality"""
    
    @pytest.mark.asyncio
    async def test_task_queue_initialization(self):
        """Test task queue initializes correctly"""
        with patch('app.core.task_queue.asyncio.Queue') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_queue:
            task_queue = await get_task_queue()
            assert task_queue is not None
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_queue.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_background_task_creation(self):
        """Test background task creation and execution"""
        async def sample_task(arg1, arg2):
            return f"{arg1}_{arg2}"
        
        with patch('app.core.task_queue.asyncio.create_task') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_create_task:
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_task = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_create_task.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_task
            
            task = create_background_task(sample_task, 'hello', 'world')
            assert task is not None
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_create_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_task_queue_priority_handling(self):
        """Test task queue handles priority correctly"""
        with patch('app.core.task_queue.heapq') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_heapq:
            from app.core.task_queue import PriorityTaskQueue
            
            queue = PriorityTaskQueue()
            await queue.put_task('low_priority', priority=10)
            await queue.put_task('high_priority', priority=1)
            
            # High priority task should be retrieved first
            task = await queue.get_task()
            assert 'high_priority' in str(task)


@pytest.mark.unit
class TestDatabaseConnection:
    """Test database connection management"""
    
    @pytest.mark.asyncio
    async def test_database_connection_success(self):
        """Test successful database connection"""
        with patch('app.core.database.asyncpg.connect') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connect:
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connection = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connect.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connection
            
            connection = await get_database_connection()
            assert connection is not None
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_database_connection_retry_logic(self):
        """Test database connection retry on failure"""
        with patch('app.core.database.asyncpg.connect') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connect:
            # First two calls fail, third succeeds
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connect.side_effect = [
                Exception('Connection failed'),
                Exception('Connection failed'),
                AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            ]
            
            connection = await get_database_connection(max_retries=3)
            assert connection is not None
            assert Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connect.call_count == 3
    
    @pytest.mark.asyncio
    async def test_database_connection_pool_exhaustion(self):
        """Test handling of connection pool exhaustion"""
        with patch('app.core.database.asyncpg.create_pool') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_create_pool:
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool.acquire.side_effect = Exception('Pool exhausted')
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_create_pool.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool
            
            from app.core.database import DatabasePool
            pool = DatabasePool()
            
            with pytest.raises(Exception):
                await pool.get_connection()


@pytest.mark.unit
class TestSecurityManager:
    """Test security management functionality"""
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        security_manager = SecurityManager()
        password = 'test_password_123'
        
        # Test hashing
        hashed = security_manager.hash_password(password)
        assert hashed != password
        assert len(hashed) > 50  # Proper hash length
        
        # Test verification
        assert security_manager.verify_password(password, hashed)
        assert not security_manager.verify_password('wrong_password', hashed)
    
    def test_jwt_token_generation(self):
        """Test JWT token generation and validation"""
        security_manager = SecurityManager()
        payload = {'user_id': 'test_user', 'role': 'admin'}
        
        # Generate token
        token = security_manager.create_jwt_token(payload)
        assert isinstance(token, str)
        assert len(token) > 100  # Proper JWT length
        
        # Validate token
        decoded = security_manager.decode_jwt_token(token)
        assert decoded['user_id'] == 'test_user'
        assert decoded['role'] == 'admin'
    
    def test_jwt_token_expiration(self):
        """Test JWT token expiration handling"""
        security_manager = SecurityManager()
        payload = {'user_id': 'test_user'}
        
        # Create expired token
        expired_time = datetime.utcnow() - timedelta(hours=1)
        with patch('app.core.security.datetime') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_datetime:
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_datetime.utcnow.return_value = expired_time
            expired_token = security_manager.create_jwt_token(payload)
        
        # Should raise exception when decoding expired token
        with pytest.raises(Exception):
            security_manager.decode_jwt_token(expired_token)
    
    def test_input_sanitization(self):
        """Test input sanitization against XSS"""
        security_manager = SecurityManager()
        
        # Test XSS payload sanitization
        malicious_input = '<script>alert("xss")</script>'
        sanitized = security_manager.sanitize_input(malicious_input)
        assert '<script>' not in sanitized
        assert 'alert' not in sanitized
        
        # Test SQL injection prevention
        sql_injection = "'; DROP TABLE users; --"
        sanitized_sql = security_manager.sanitize_sql_input(sql_injection)
        assert 'DROP TABLE' not in sanitized_sql


@pytest.mark.unit
class TestMetricsCollector:
    """Test metrics collection functionality"""
    
    def test_metrics_initialization(self):
        """Test metrics collector initializes correctly"""
        metrics = MetricsCollector()
        assert metrics.get_metric('requests_total') == 0
        assert metrics.get_metric('errors_total') == 0
    
    def test_counter_metrics(self):
        """Test counter metric operations"""
        metrics = MetricsCollector()
        
        # Test increment
        metrics.increment('api_requests')
        assert metrics.get_metric('api_requests') == 1
        
        # Test increment by value
        metrics.increment('api_requests', 5)
        assert metrics.get_metric('api_requests') == 6
    
    def test_histogram_metrics(self):
        """Test histogram metric operations"""
        metrics = MetricsCollector()
        
        # Record response times
        metrics.record_histogram('response_time', 0.1)
        metrics.record_histogram('response_time', 0.2)
        metrics.record_histogram('response_time', 0.15)
        
        histogram_data = metrics.get_histogram('response_time')
        assert histogram_data['count'] == 3
        assert histogram_data['sum'] == 0.45
        assert 0.1 <= histogram_data['avg'] <= 0.2
    
    def test_gauge_metrics(self):
        """Test gauge metric operations"""
        metrics = MetricsCollector()
        
        # Set gauge value
        metrics.set_gauge('active_connections', 10)
        assert metrics.get_gauge('active_connections') == 10
        
        # Update gauge value
        metrics.set_gauge('active_connections', 15)
        assert metrics.get_gauge('active_connections') == 15
    
    def test_metrics_export_format(self):
        """Test metrics export in Prometheus format"""
        metrics = MetricsCollector()
        metrics.increment('requests_total', 100)
        metrics.set_gauge('memory_usage', 512)
        
        prometheus_output = metrics.export_prometheus_format()
        assert 'requests_total 100' in prometheus_output
        assert 'memory_usage 512' in prometheus_output
        assert prometheus_output.startswith('# TYPE')


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling across core components"""
    
    @pytest.mark.asyncio
    async def test_graceful_service_degradation(self):
        """Test system handles service failures gracefully"""
        with patch('app.core.cache.get_cache_service') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_cache:
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_cache.side_effect = Exception('Cache service unavailable')
            
            # System should continue without cache
            from app.core.cache import cached_with_fallback
            
            @cached_with_fallback(ttl=300)
            async def test_function():
                return 'result'
            
            result = await test_function()
            assert result == 'result'  # Function executed despite cache failure
    
    def test_exception_logging_and_metrics(self):
        """Test exceptions are properly logged and tracked in metrics"""
        with patch('app.core.metrics.logger') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_logger:
            metrics = MetricsCollector()
            
            try:
                raise ValueError('Test exception')
            except ValueError as e:
                metrics.record_exception(e)
                metrics.increment('errors_total')
            
            assert metrics.get_metric('errors_total') == 1
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_logger.error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test proper timeout handling in async operations"""
        async def slow_operation():
            await asyncio.sleep(10)  # Simulate slow operation
        
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_operation(), timeout=0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
