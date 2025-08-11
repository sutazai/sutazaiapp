#!/usr/bin/env python3
"""
Global pytest configuration and fixtures for Ollama integration tests
Provides common fixtures, utilities, and test configuration
"""

import pytest
import asyncio
import os
import sys
import tempfile
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import logging

# Add agents directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents'))

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file for testing"""
    config_data = {
        "capabilities": ["test", "example"],
        "max_retries": 3,
        "timeout": 300,
        "batch_size": 10
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except OSError:
        pass


@pytest.fixture
def mock_environment():
    """Mock environment variables for testing"""
    test_env = {
        'AGENT_NAME': 'test-agent',
        'AGENT_TYPE': 'test',
        'BACKEND_URL': 'http://test-backend:8000',
        'OLLAMA_URL': 'http://test-ollama:10104',
        'LOG_LEVEL': 'WARNING'
    }
    
    with patch.dict(os.environ, test_env):
        yield test_env


@pytest.fixture
def mock_ollama_service():
    """Mock Ollama service responses"""
    class MockOllamaService:
        def __init__(self):
            self.models = ["tinyllama"]
            self.responses = {}
        
        def set_response(self, prompt, response):
            """Set a specific response for a prompt"""
            self.responses[prompt] = response
        
        def get_response(self, prompt):
            """Get response for a prompt"""
            return self.responses.get(prompt, f"Mock response for: {prompt[:50]}...")
        
        def mock_generate(self, prompt, **kwargs):
            """Mock generate method"""
            return self.get_response(prompt)
        
        def mock_chat(self, messages, **kwargs):
            """Mock chat method"""
            if messages:
                last_message = messages[-1].get('content', '')
                return self.get_response(last_message)
            return "Mock chat response"
        
        def mock_embeddings(self, text, **kwargs):
            """Mock embeddings method"""
            # Return mock embedding vector
            return [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # 500-dimensional mock vector
        
        def mock_model_list(self):
            """Mock model list response"""
            return {
                "models": [{"name": f"{model}:latest"} for model in self.models]
            }
    
    return MockOllamaService()


@pytest.fixture
def mock_backend_service():
    """Mock backend coordinator responses"""
    class MockBackendService:
        def __init__(self):
            self.agents = {}
            self.tasks = []
            self.completions = []
        
        def register_agent(self, agent_data):
            """Mock agent registration"""
            agent_name = agent_data.get('agent_name')
            self.agents[agent_name] = agent_data
            return Mock(status_code=200)
        
        def get_next_task(self, agent_type):
            """Mock task retrieval"""
            if self.tasks:
                task = self.tasks.pop(0)
                response = Mock()
                response.status_code = 200
                response.json.return_value = task
                return response
            else:
                response = Mock()
                response.status_code = 204  # No tasks available
                return response
        
        def add_task(self, task):
            """Add a task to the queue"""
            self.tasks.append(task)
        
        def complete_task(self, completion_data):
            """Mock task completion"""
            self.completions.append(completion_data)
            return Mock(status_code=200)
        
        def heartbeat(self, heartbeat_data):
            """Mock heartbeat"""
            return Mock(status_code=200)
        
        def health_check(self):
            """Mock health check"""
            return Mock(status_code=200)
    
    return MockBackendService()


@pytest.fixture
async def base_agent():
    """Create a base agent for testing"""
    from agents.core.base_agent import BaseAgentV2
    
    with patch.dict(os.environ, {
        'AGENT_NAME': 'test-base-agent',
        'AGENT_TYPE': 'test-base'
    }):
        agent = BaseAgentV2()
        await agent._setup_async_components()
        
        yield agent
        
        await agent._cleanup_async_components()


@pytest.fixture
def sample_task():
    """Create a sample task for testing"""
    return {
        "id": "test-task-001",
        "type": "test",
        "data": {
            "prompt": "This is a test prompt",
            "model": "tinyllama",
            "max_tokens": 100
        },
        "created_at": datetime.utcnow().isoformat()
    }


@pytest.fixture
def sample_task_result():
    """Create a sample task result for testing"""
    from agents.core.base_agent import TaskResult
    
    return TaskResult(
        task_id="test-task-001",
        status="completed",
        result={
            "output": "Test task completed successfully",
            "model_used": "tinyllama",
            "tokens_used": 50
        },
        processing_time=1.5
    )


@pytest.fixture
def mock_circuit_breaker():
    """Mock circuit breaker for testing"""
    from core.circuit_breaker import CircuitBreaker, CircuitBreakerState
    
    mock_breaker = Mock(spec=CircuitBreaker)
    mock_breaker.state = CircuitBreakerState.CLOSED
    mock_breaker.call = AsyncMock(side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))
    mock_breaker.trip_count = 0
    
    return mock_breaker


@pytest.fixture
def mock_connection_pool():
    """Mock connection pool for testing"""
    from core.ollama_pool import OllamaConnectionPool
    
    mock_pool = Mock(spec=OllamaConnectionPool)
    mock_pool.generate = AsyncMock(return_value="Mock pool response")
    mock_pool.chat = AsyncMock(return_value="Mock chat response")
    mock_pool.embeddings = AsyncMock(return_value=[0.1, 0.2, 0.3])
    mock_pool.health_check = AsyncMock(return_value=True)
    mock_pool.get_stats = Mock(return_value={
        "total_requests": 10,
        "successful_requests": 9,
        "failed_requests": 1,
        "average_response_time": 0.5
    })
    mock_pool.close = AsyncMock()
    
    return mock_pool


@pytest.fixture
def mock_request_queue():
    """Mock request queue for testing"""
    from core.request_queue import RequestQueue
    
    mock_queue = Mock(spec=RequestQueue)
    mock_queue.submit = AsyncMock(side_effect=lambda task: task)
    mock_queue.close = AsyncMock()
    mock_queue.size = Mock(return_value=0)
    
    return mock_queue


@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """Automatically cleanup test environment after each test"""
    yield
    
    # Cleanup any test artifacts
    import gc
    gc.collect()
    
    # Close any remaining async tasks
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            pending = asyncio.all_tasks(loop)
            for task in pending:
                if not task.done():
                    task.cancel()
    except RuntimeError:
        pass  # No event loop running


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "regression: Regression tests")
    config.addinivalue_line("markers", "failure: Failure scenario tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "network: Tests requiring network")
    config.addinivalue_line("markers", "ollama: Tests requiring Ollama service")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file names"""
    for item in items:
        # Add markers based on test file names
        if "test_performance" in item.fspath.basename:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        elif "test_integration" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
        elif "test_failure" in item.fspath.basename:
            item.add_marker(pytest.mark.failure)
        elif "test_regression" in item.fspath.basename:
            item.add_marker(pytest.mark.regression)
        else:
            item.add_marker(pytest.mark.unit)
        
        # Add ollama marker for tests that use Ollama
        if "ollama" in item.name.lower() or "ollama" in str(item.fspath).lower():
            item.add_marker(pytest.mark.ollama)


# Pytest report hooks
def pytest_runtest_makereport(item, call):
    """Generate detailed test reports"""
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item):
    """Setup for incremental tests"""
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail(f"previous test failed ({previousfailed.name})")


# Custom assertions for testing
class TestAssertions:
    """Custom assertions for Ollama integration tests"""
    
    @staticmethod
    def assert_valid_agent_metrics(metrics):
        """Assert that agent metrics are valid"""
        assert hasattr(metrics, 'tasks_processed')
        assert hasattr(metrics, 'tasks_failed')
        assert hasattr(metrics, 'ollama_requests')
        assert hasattr(metrics, 'ollama_failures')
        
        assert metrics.tasks_processed >= 0
        assert metrics.tasks_failed >= 0
        assert metrics.ollama_requests >= 0
        assert metrics.ollama_failures >= 0
    
    @staticmethod
    def assert_valid_task_result(result):
        """Assert that task result is valid"""
        from agents.core.base_agent import TaskResult
        
        assert isinstance(result, TaskResult)
        assert result.task_id is not None
        assert result.status in ["completed", "failed"]
        assert isinstance(result.result, dict)
        assert result.processing_time >= 0
    
    @staticmethod
    def assert_performance_within_bounds(execution_time, max_time, operation_name="Operation"):
        """Assert that performance is within acceptable bounds"""
        assert execution_time <= max_time, \
            f"{operation_name} took {execution_time:.3f}s, exceeds limit of {max_time}s"
    
    @staticmethod
    def assert_memory_usage_reasonable(memory_mb, max_memory_mb=500):
        """Assert that memory usage is reasonable"""
        assert memory_mb <= max_memory_mb, \
            f"Memory usage {memory_mb:.1f}MB exceeds limit of {max_memory_mb}MB"


# Make custom assertions available globally
@pytest.fixture
def assertions():
    """Provide custom test assertions"""
    return TestAssertions