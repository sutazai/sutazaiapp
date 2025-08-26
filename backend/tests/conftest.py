"""
Test configuration and fixtures for SutazAI backend testing
Professional-grade test infrastructure with comprehensive fixtures
"""

import asyncio
import os
import pytest
import logging
from typing import AsyncGenerator, Dict, Any
from unittest. import Async, Magic, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient
import tempfile
import json

# Set test environment variables before importing app
os.environ.update({
    "SUTAZAI_ENV": "test",
    "JWT_SECRET_KEY": "test_secret_key_for_testing_minimum_32_chars",
    "DATABASE_URL": "postgresql://test:test@localhost:5432/sutazai_test",
    "REDIS_URL": "redis://localhost:6379/15",
    "OLLAMA_HOST": "http://localhost:11434",
    "DISABLE_AUTH": "false"
})

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def _cache_service():
    """ cache service for testing"""
    cache_ = Async()
    cache_.get.return_value = None
    cache_.set.return_value = True
    cache_.delete.return_value = True
    cache_.clear_all.return_value = True
    cache_.get_stats.return_value = {
        "hits": 100,
        "misses": 20,
        "hit_rate": 0.83,
        "total_operations": 120
    }
    return cache_


@pytest.fixture
async def _pool_manager():
    """ connection pool manager for testing"""
    pool_ = Async()
    pool_.get_stats.return_value = {
        "active_connections": 5,
        "idle_connections": 10,
        "total_connections": 15
    }
    pool_.close.return_value = None
    return pool_


@pytest.fixture
async def _ollama_service():
    """ Ollama service for testing"""
    ollama_ = Async()
    ollama_.generate.return_value = {
        "response": "Test response from Ollama",
        "model": "tinyllama",
        "duration": 0.5
    }
    ollama_.generate_streaming.return_value = asyncio.iterate([
        {"chunk": "Test streaming response"}
    ])
    ollama_.batch_generate.return_value = [
        {"response": "Batch response 1"},
        {"response": "Batch response 2"}
    ]
    ollama_.get_stats.return_value = {
        "requests_processed": 100,
        "average_response_time": 0.5,
        "cache_hit_rate": 0.7
    }
    ollama_.warmup.return_value = None
    ollama_.shutdown.return_value = None
    return ollama_


@pytest.fixture
async def _task_queue():
    """ task queue for testing"""
    queue_ = Async()
    queue_.get_task_status.return_value = {
        "status": "completed",
        "result": {"test": "result"}
    }
    queue_.get_stats.return_value = {
        "queued": 5,
        "processing": 2,
        "completed": 100
    }
    queue_.register_handler.return_value = None
    queue_.stop.return_value = None
    return queue_


@pytest.fixture
async def _health_monitoring():
    """ health monitoring service for testing"""
    health_ = Async()
    health_.get_detailed_health.return_value = Magic(
        overall_status=Magic(value="healthy"),
        timestamp=Magic(),
        services={
            "redis": Magic(
                status=Magic(value="healthy"),
                response_time_ms=5,
                details={"ping": "pong"}
            )
        },
        performance_metrics={
            "api_response_p95": 100,
            "cache_hit_rate": 0.85,
            "db_connection_pool_usage": 0.5
        },
        system_resources={
            "memory_percent": 65.2,
            "disk_percent": 45.8,
            "network_io": {"bytes_sent": 1000000, "bytes_recv": 2000000}
        },
        alerts=[],
        recommendations=[]
    )
    health_.get_prometheus_metrics.return_value = "# Prometheus metrics test data"
    return health_


@pytest.fixture
async def _circuit_breaker_manager():
    """ circuit breaker manager for testing"""
    breaker_ = Async()
    breaker_.get_all_stats.return_value = {
        "redis": {
            "state": "closed",
            "success_count": 100,
            "failure_count": 2,
            "fail_max": 5,
            "reset_timeout": 60
        }
    }
    breaker_.reset_all.return_value = None
    return breaker_


@pytest.fixture
async def app_with_s(
    _cache_service,
    _pool_manager,
    _ollama_service,
    _task_queue,
    _health_monitoring,
    _circuit_breaker_manager
):
    """Create FastAPI app with all dependencies ed"""
    
    # Patch all dependencies before importing app
    with patch('app.core.cache.get_cache_service', Async(return_value=_cache_service)), \
         patch('app.core.cache._cache_service', _cache_service), \
         patch('app.core.connection_pool.get_pool_manager', Async(return_value=_pool_manager)), \
         patch('app.core.connection_pool.get_http_client', Async()), \
         patch('app.services.consolidated_ollama_service.get_ollama_service', Async(return_value=_ollama_service)), \
         patch('app.core.task_queue.get_task_queue', Async(return_value=_task_queue)), \
         patch('app.core.task_queue.create_background_task', Async(return_value="test-task-id")), \
         patch('app.core.health_monitoring.get_health_monitoring_service', Async(return_value=_health_monitoring)), \
         patch('app.core.circuit_breaker_integration.get_circuit_breaker_manager', Async(return_value=_circuit_breaker_manager)), \
         patch('app.core.circuit_breaker_integration.get_redis_circuit_breaker', Async()), \
         patch('app.core.circuit_breaker_integration.get_database_circuit_breaker', Async()), \
         patch('app.core.circuit_breaker_integration.get_ollama_circuit_breaker', Async()):
        
        # Import after patching to ensure s are in place
        from app.main import app
        yield app


@pytest.fixture
async def async_client(app_with_s) -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing"""
    async with AsyncClient(app=app_with_s, base_url="http://test") as client:
        yield client


@pytest.fixture
def sync_client(app_with_s) -> TestClient:
    """Create sync HTTP client for testing"""
    return TestClient(app_with_s)


@pytest.fixture
def sample_chat_request():
    """Sample chat request for testing"""
    return {
        "message": "Hello, test!",
        "model": "tinyllama",
        "use_cache": True
    }


@pytest.fixture
def sample_task_request():
    """Sample task request for testing"""
    return {
        "task_type": "automation",
        "payload": {"test": "data"},
        "priority": 1
    }


@pytest.fixture
def sample_agent_data():
    """Sample agent data for testing"""
    return {
        "id": "test-agent-001",
        "name": "Test Agent",
        "status": "active",
        "capabilities": ["test", "automation"]
    }


@pytest.fixture
def temp_test_files():
    """Create temporary test files"""
    files = []
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"test": "data"}, f)
        files.append(f.name)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content")
        files.append(f.name)
    
    yield files
    
    # Cleanup
    for file in files:
        try:
            os.unlink(file)
        except:
            pass


@pytest.fixture
def _psutil():
    """ psutil for system metrics testing"""
    with patch('psutil.cpu_percent') as _cpu, \
         patch('psutil.virtual_memory') as _memory, \
         patch('psutil.disk_usage') as _disk, \
         patch('psutil.net_io_counters') as _net:
        
        _cpu.return_value = 25.5
        _memory.return_value = Magic(
            percent=65.2,
            available=4000000000,
            total=16000000000
        )
        _disk.return_value = Magic(percent=45.8)
        _net.return_value = Magic(
            bytes_sent=1000000,
            bytes_recv=2000000
        )
        
        yield


@pytest.fixture
def _validation():
    """ validation utilities"""
    return Magic(
        validate_model_name=Magic(return_value="tinyllama"),
        validate_agent_id=Magic(side_effect=lambda x: x),
        validate_task_id=Magic(side_effect=lambda x: x),
        validate_cache_pattern=Magic(side_effect=lambda x: x),
        sanitize_user_input=Magic(side_effect=lambda x, max_length=None: x)
    )


@pytest.fixture
def _jwt_auth():
    """ JWT authentication for testing"""
    with patch('app.auth.router.get_current_user') as _user:
        _user.return_value = {
            "username": "testuser",
            "email": "test@example.com",
            "is_active": True
        }
        yield _user