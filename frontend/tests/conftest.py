"""
Pytest configuration and fixtures for Sutazai test suite
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
API_BASE_URL = "http://localhost:10200"
FRONTEND_URL = "http://localhost:11000"
WS_BASE_URL = "ws://localhost:10200"

@pytest.fixture(scope="session")
def api_base_url():
    """Provide base API URL"""
    return API_BASE_URL

@pytest.fixture(scope="session")
def frontend_url():
    """Provide frontend URL"""
    return FRONTEND_URL

@pytest.fixture(scope="session")
def ws_base_url():
    """Provide WebSocket base URL"""
    return WS_BASE_URL

@pytest.fixture(scope="function")
def session_id():
    """Generate unique session ID for each test"""
    import uuid
    return str(uuid.uuid4())

@pytest.fixture(scope="session")
def test_timeout():
    """Default timeout for tests"""
    return 30

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "websocket: mark test as websocket test"
    )
    config.addinivalue_line(
        "markers", "agent: mark test as agent-specific test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )

# Test collection configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        if "websocket" in item.name.lower():
            item.add_marker(pytest.mark.websocket)
        if "agent" in item.name.lower():
            item.add_marker(pytest.mark.agent)
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)