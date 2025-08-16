"""
Test configuration and fixtures for SutazAI backend testing
Professional-grade test infrastructure with comprehensive fixtures
"""

import asyncio
import os
import pytest
import logging
from typing import AsyncGenerator, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
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
async def mock_cache_service():
    """Mock cache service for testing"""
    cache_mock = AsyncMock()
    cache_mock.get.return_value = None
    cache_mock.set.return_value = True
    cache_mock.delete.return_value = True
    cache_mock.clear_all.return_value = True
    cache_mock.get_stats.return_value = {
        "hits": 100,
        "misses": 20,
        "hit_rate": 0.83,
        "total_operations": 120
    }
    return cache_mock


@pytest.fixture
async def mock_pool_manager():
    """Mock connection pool manager for testing"""
    pool_mock = AsyncMock()
    pool_mock.get_stats.return_value = {
        "active_connections": 5,
        "idle_connections": 10,
        "total_connections": 15
    }
    pool_mock.close.return_value = None
    return pool_mock


@pytest.fixture
async def mock_ollama_service():
    """Mock Ollama service for testing"""
    ollama_mock = AsyncMock()
    ollama_mock.generate.return_value = {
        "response": "Test response from Ollama",
        "model": "tinyllama",
        "duration": 0.5
    }
    ollama_mock.generate_streaming.return_value = asyncio.iterate([
        {"chunk": "Test streaming response"}
    ])
    ollama_mock.batch_generate.return_value = [
        {"response": "Batch response 1"},
        {"response": "Batch response 2"}
    ]
    ollama_mock.get_stats.return_value = {
        "requests_processed": 100,
        "average_response_time": 0.5,
        "cache_hit_rate": 0.7
    }
    ollama_mock.warmup.return_value = None
    ollama_mock.shutdown.return_value = None
    return ollama_mock


@pytest.fixture
async def mock_task_queue():
    """Mock task queue for testing"""
    queue_mock = AsyncMock()
    queue_mock.get_task_status.return_value = {
        "status": "completed",
        "result": {"test": "result"}
    }
    queue_mock.get_stats.return_value = {
        "queued": 5,
        "processing": 2,
        "completed": 100
    }
    queue_mock.register_handler.return_value = None
    queue_mock.stop.return_value = None
    return queue_mock


@pytest.fixture
async def mock_health_monitoring():
    """Mock health monitoring service for testing"""
    health_mock = AsyncMock()
    health_mock.get_detailed_health.return_value = MagicMock(
        overall_status=MagicMock(value="healthy"),
        timestamp=MagicMock(),
        services={
            "redis": MagicMock(
                status=MagicMock(value="healthy"),
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
    health_mock.get_prometheus_metrics.return_value = "# Prometheus metrics test data"
    return health_mock


@pytest.fixture
async def mock_circuit_breaker_manager():
    """Mock circuit breaker manager for testing"""
    breaker_mock = AsyncMock()
    breaker_mock.get_all_stats.return_value = {
        "redis": {
            "state": "closed",
            "success_count": 100,
            "failure_count": 2,
            "fail_max": 5,
            "reset_timeout": 60
        }
    }
    breaker_mock.reset_all.return_value = None
    return breaker_mock


@pytest.fixture
async def app_with_mocks(
    mock_cache_service,
    mock_pool_manager,
    mock_ollama_service,
    mock_task_queue,
    mock_health_monitoring,
    mock_circuit_breaker_manager
):
    """Create FastAPI app with all dependencies mocked"""
    
    # Patch all dependencies before importing app
    with patch('app.core.cache.get_cache_service', AsyncMock(return_value=mock_cache_service)), \
         patch('app.core.cache._cache_service', mock_cache_service), \
         patch('app.core.connection_pool.get_pool_manager', AsyncMock(return_value=mock_pool_manager)), \
         patch('app.core.connection_pool.get_http_client', AsyncMock()), \
         patch('app.services.consolidated_ollama_service.get_ollama_service', AsyncMock(return_value=mock_ollama_service)), \
         patch('app.core.task_queue.get_task_queue', AsyncMock(return_value=mock_task_queue)), \
         patch('app.core.task_queue.create_background_task', AsyncMock(return_value="test-task-id")), \
         patch('app.core.health_monitoring.get_health_monitoring_service', AsyncMock(return_value=mock_health_monitoring)), \
         patch('app.core.circuit_breaker_integration.get_circuit_breaker_manager', AsyncMock(return_value=mock_circuit_breaker_manager)), \
         patch('app.core.circuit_breaker_integration.get_redis_circuit_breaker', AsyncMock()), \
         patch('app.core.circuit_breaker_integration.get_database_circuit_breaker', AsyncMock()), \
         patch('app.core.circuit_breaker_integration.get_ollama_circuit_breaker', AsyncMock()):
        
        # Import after patching to ensure mocks are in place
        from app.main import app
        yield app


@pytest.fixture
async def async_client(app_with_mocks) -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing"""
    async with AsyncClient(app=app_with_mocks, base_url="http://test") as client:
        yield client


@pytest.fixture
def sync_client(app_with_mocks) -> TestClient:
    """Create sync HTTP client for testing"""
    return TestClient(app_with_mocks)


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
def mock_psutil():
    """Mock psutil for system metrics testing"""
    with patch('psutil.cpu_percent') as mock_cpu, \
         patch('psutil.virtual_memory') as mock_memory, \
         patch('psutil.disk_usage') as mock_disk, \
         patch('psutil.net_io_counters') as mock_net:
        
        mock_cpu.return_value = 25.5
        mock_memory.return_value = MagicMock(
            percent=65.2,
            available=4000000000,
            total=16000000000
        )
        mock_disk.return_value = MagicMock(percent=45.8)
        mock_net.return_value = MagicMock(
            bytes_sent=1000000,
            bytes_recv=2000000
        )
        
        yield


@pytest.fixture
def mock_validation():
    """Mock validation utilities"""
    return MagicMock(
        validate_model_name=MagicMock(return_value="tinyllama"),
        validate_agent_id=MagicMock(side_effect=lambda x: x),
        validate_task_id=MagicMock(side_effect=lambda x: x),
        validate_cache_pattern=MagicMock(side_effect=lambda x: x),
        sanitize_user_input=MagicMock(side_effect=lambda x, max_length=None: x)
    )


@pytest.fixture
def mock_jwt_auth():
    """Mock JWT authentication for testing"""
    with patch('app.auth.router.get_current_user') as mock_user:
        mock_user.return_value = {
            "username": "testuser",
            "email": "test@example.com",
            "is_active": True
        }
        yield mock_user