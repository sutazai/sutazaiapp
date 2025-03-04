"""
Pytest configuration for the test suite.
"""
import os
import sys
import pytest
import pytest_asyncio

# Add the parent directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "asyncio: mark test as an asyncio test")

@pytest_asyncio.fixture
async def orchestrator():
    # Create and return an actual orchestrator instance
    orchestrator_instance = Orchestrator()
    await orchestrator_instance.initialize()
    yield orchestrator_instance
    await orchestrator_instance.cleanup()
