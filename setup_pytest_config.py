#!/usr/bin/env python3
"""
Set up proper pytest configuration for the project.
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_fixes.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("setup_pytest")

def create_pyproject_toml():
    """Create or update the pyproject.toml file with pytest configuration."""
    file_path = Path("/opt/sutazaiapp/pyproject.toml")
    
    pytest_config = """[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_mode = "auto"
markers = [
    "unit: marks a test as a unit test",
    "integration: marks a test as an integration test",
    "slow: marks a test as a slow test",
]
"""
    
    try:
        if file_path.exists():
            # Read existing content
            with open(file_path, "r") as f:
                content = f.read()
            
            # Check if pytest config already exists
            if "[tool.pytest.ini_options]" in content:
                logger.info("pytest configuration already exists in pyproject.toml")
                return True
            
            # Append pytest config
            content += "\n" + pytest_config
            
            with open(file_path, "w") as f:
                f.write(content)
        else:
            # Create new file
            with open(file_path, "w") as f:
                f.write(pytest_config)
        
        logger.info("Created/updated pytest configuration in pyproject.toml")
        return True
    except Exception as e:
        logger.error(f"Error creating/updating pyproject.toml: {e}")
        return False

def create_conftest_py():
    """Create or update the conftest.py file with pytest fixtures."""
    file_path = Path("/opt/sutazaiapp/tests/conftest.py")
    
    conftest_content = """import pytest
import asyncio
from unittest.mock import MagicMock, patch

# Ensure pytest knows about asyncio markers
pytest_plugins = ["pytest_asyncio"]

@pytest.fixture
def event_loop():
        """Create an instance of the default event loop for each test."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_db_session():
    """Fixture to provide a mock database session."""
    return MagicMock()

@pytest.fixture
def sync_manager():
    """Fixture to provide a sync manager with mocked dependencies."""
    from core_system.orchestrator.sync_manager import SyncManager
    
    # Mock the dependencies
    manager = SyncManager()
    manager.api_client = MagicMock()
    manager.config = MagicMock()
    manager.exception_handler = MagicMock()
    
    return manager

@pytest.fixture
def agent_manager():
    """Fixture to provide an agent manager with mocked dependencies."""
    from core_system.orchestrator.agent_manager import AgentManager
    
    with patch('core_system.orchestrator.agent_manager.AgentManager._initialize_event_system') as mock_init:
        # Mock the initialization
        mock_init.return_value = None
        
        # Create the manager
        manager = AgentManager()
        
        # Mock the dependencies
        manager.event_system = MagicMock()
        manager.heartbeat_task = MagicMock()
        
        return manager
"""
    
    try:
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.exists():
            # Read existing content
            with open(file_path, "r") as f:
                content = f.read()
            
            # Check if content already has the fixtures
            if "def sync_manager():" in content and "def agent_manager():" in content:
                logger.info("Required fixtures already exist in conftest.py")
                return True
            
            # Append new fixtures if they don't exist
            if "def sync_manager():" not in content:
                content += "\n" + conftest_content.split("@pytest.fixture\ndef sync_manager():")[1].split("@pytest.fixture\ndef agent_manager():")[0]
            
            if "def agent_manager():" not in content:
                content += "\n" + conftest_content.split("@pytest.fixture\ndef agent_manager():")[1]
            
            with open(file_path, "w") as f:
                f.write(content)
        else:
            # Create new file
            with open(file_path, "w") as f:
                f.write(conftest_content)
        
        logger.info("Created/updated pytest fixtures in conftest.py")
        return True
    except Exception as e:
        logger.error(f"Error creating/updating conftest.py: {e}")
        return False

def main():
    """Main function to set up pytest configuration."""
    success_pyproject = create_pyproject_toml()
    success_conftest = create_conftest_py()
    
    if success_pyproject and success_conftest:
        logger.info("Successfully set up pytest configuration")
    else:
        logger.warning("Some issues occurred during pytest configuration setup")

if __name__ == "__main__":
    main() 