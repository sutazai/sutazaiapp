# SutazAI Comprehensive Test Configuration
# Professional pytest fixtures and configuration per Rules 1-19

import pytest
import asyncio
import aiohttp
import logging
import os
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch

# Configure test logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# Test environment configuration
TEST_BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:10010")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:10011")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:10104")

@pytest.fixture(scope="session")
def event_loop():
    """Create session-wide event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def aiohttp_session() -> AsyncGenerator[aiohttp.ClientSession, None]:
    """Provide aiohttp session for integration tests."""
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        yield session

@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response for unit tests."""
    return {
        "model": "tinyllama",
        "created_at": "2024-01-01T00:00:00Z",
        "response": "Test response from Ollama",
        "done": True,
        "context": [1, 2, 3],
        "total_duration": 1000000,
        "load_duration": 500000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 200000,
        "eval_count": 20,
        "eval_duration": 300000
    }

@pytest.fixture
def mock_database():
    """Mock database connection for unit tests."""
    db_Mock = Mock()
    db_Mock.execute = AsyncMock()
    db_Mock.fetch = AsyncMock(return_value=[])
    db_Mock.fetchrow = AsyncMock(return_value=None)
    db_Mock.fetchval = AsyncMock(return_value=None)
    return db_Mock

@pytest.fixture
def mock_redis():
    """Mock Redis connection for unit tests."""
    redis_Mock = Mock()
    redis_Mock.get = AsyncMock(return_value=None)
    redis_Mock.set = AsyncMock(return_value=True)
    redis_Mock.delete = AsyncMock(return_value=1)
    redis_Mock.exists = AsyncMock(return_value=False)
    redis_Mock.expire = AsyncMock(return_value=True)
    return redis_Mock

@pytest.fixture
def sample_chat_message():
    """Sample chat message for testing."""
    return {
        "message": "Hello, how are you?",
        "model": "tinyllama",
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.7,
            "top_p": 0.9
        }
    }

@pytest.fixture
def sample_agent_config():
    """Sample agent configuration for testing."""
    return {
        "name": "test-agent",
        "type": "generic",
        "capabilities": ["chat", "analysis"],
        "model": "tinyllama",
        "max_tokens": 1000,
        "temperature": 0.7
    }

@pytest.fixture
def mock_agent_registry():
    """Mock agent registry for unit tests."""
    registry_Mock = Mock()
    registry_Mock.get_agent = AsyncMock()
    registry_Mock.register_agent = AsyncMock()
    registry_Mock.list_agents = AsyncMock(return_value=[])
    registry_Mock.remove_agent = AsyncMock()
    return registry_Mock

@pytest.fixture
def health_check_response():
    """Expected health check response structure."""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "1.0.0",
        "services": {
            "database": "healthy",
            "redis": "healthy",
            "ollama": "healthy"
        },
        "system": {
            "cpu_usage": 25.5,
            "memory_usage": 45.2,
            "disk_usage": 30.1
        }
    }

# Performance test fixtures
@pytest.fixture
def performance_thresholds():
    """Performance testing thresholds."""
    return {
        "health_endpoint_max_time": 0.2,  # 200ms
        "chat_endpoint_max_time": 5.0,    # 5 seconds
        "model_load_max_time": 10.0,      # 10 seconds
        "concurrent_users": 50,
        "max_memory_mb": 1024,
        "max_cpu_percent": 80
    }

# Security test fixtures
@pytest.fixture
def xss_payloads():
    """XSS attack payloads for security testing."""
    return [
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        "<img src=x onerror=alert('XSS')>",
        "';alert('XSS');//",
        "<svg onload=alert('XSS')>",
        "%3Cscript%3Ealert('XSS')%3C/script%3E"
    ]

@pytest.fixture
def sql_injection_payloads():
    """SQL injection payloads for security testing."""
    return [
        "'; DROP TABLE users; --",
        "' OR '1'='1",
        "' UNION SELECT * FROM users --",
        "admin'--",
        "' OR 1=1--",
        "'; EXEC xp_cmdshell('dir'); --"
    ]

# Database test fixtures
@pytest.fixture
async def test_database():
    """Test database connection (integration tests)."""
    try:
        # Import here to avoid import errors in unit tests
        from app.core.database import get_db
        async for db in get_db():
            yield db
            break
    except ImportError:
        # Fallback Mock for unit tests
        yield mock_database()

# Integration test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "requires_system: Tests that require running system services"
    )
    config.addinivalue_line(
        "markers", "requires_ollama: Tests that require Ollama service"
    )
    config.addinivalue_line(
        "markers", "requires_database: Tests that require database connection"
    )

# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Auto-mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            
        # Auto-mark performance tests
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            
        # Auto-mark security tests
        if "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
            
        # Auto-mark unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
            
        # Auto-mark e2e tests
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

# Session fixtures for system health
@pytest.fixture(scope="session", autouse=True)
async def ensure_test_environment():
    """Ensure test environment is ready before running tests."""
    # Check if we're in test mode
    if not os.getenv("TESTING"):
        os.environ["TESTING"] = "true"
    
    # Set test-specific configurations
    os.environ["LOG_LEVEL"] = "WARNING"
    os.environ["PYTEST_CURRENT_TEST"] = "true"
    
    yield
    
    # Cleanup after all tests
    # Any session-level cleanup would go here