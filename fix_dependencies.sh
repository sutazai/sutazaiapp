#!/bin/bash

# Script to fix dependencies and ensure tests can run successfully
# This addresses all missing modules and prepares the environment for 100% test success

echo "Starting dependency fix process..."

# SSH key to use
SSH_KEY="/root/.ssh/sutazaiapp_sync_key"
REMOTE_SERVER="root@192.168.100.100"

# Install all necessary dependencies
echo "Installing all necessary dependencies..."
ssh -i $SSH_KEY $REMOTE_SERVER "cd /opt/sutazaiapp && source venv/bin/activate && \
    pip install requests psutil fastapi httpx sqlalchemy aiohttp mock pandas numpy pytest pytest-asyncio pytest-cov pytest-xdist pytest-html"

# Create necessary directories
echo "Creating necessary directories..."
ssh -i $SSH_KEY $REMOTE_SERVER "mkdir -p /opt/sutazaiapp/logs /opt/sutazaiapp/tmp /opt/sutazaiapp/coverage /opt/sutazaiapp/test_reports"

# Fix permissions
echo "Fixing permissions..."
ssh -i $SSH_KEY $REMOTE_SERVER "chmod -R 755 /opt/sutazaiapp/scripts"

# Create log files
echo "Creating log files..."
ssh -i $SSH_KEY $REMOTE_SERVER "touch /opt/sutazaiapp/logs/code_audit.log"

# Update conftest.py
echo "Updating test configuration..."
ssh -i $SSH_KEY $REMOTE_SERVER "cd /opt/sutazaiapp && cat > tests/conftest.py << 'EOF'
\"\"\"Common test fixtures for all tests.\"\"\"
import os
import pytest
from pathlib import Path
import tempfile
import shutil
import json

@pytest.fixture
def temp_dir():
    \"\"\"Create a temporary directory for tests.\"\"\"
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def test_config(temp_dir):
    \"\"\"Create a test configuration file.\"\"\"
    # Create directories
    log_dir = temp_dir / 'logs'
    data_dir = temp_dir / 'data'
    backup_dir = temp_dir / 'backups'
    log_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file
    (log_dir / 'code_audit.log').touch()
    
    # Create config file
    config = {
        'log_dir': str(log_dir),
        'data_dir': str(data_dir),
        'backup_dir': str(backup_dir),
        'max_log_age_days': 7,
        'max_backup_age_days': 30,
        'backup_retention': 5
    }
    config_path = temp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    return config_path

@pytest.fixture
def mock_environment(monkeypatch):
    \"\"\"Set up mock environment variables for tests.\"\"\"
    monkeypatch.setenv('SUTAZAI_ENV', 'test')
    monkeypatch.setenv('SUTAZAI_LOG_LEVEL', 'DEBUG')
    monkeypatch.setenv('SUTAZAI_CONFIG_PATH', '/tmp/test_config.json')
    
@pytest.fixture
def sample_data():
    \"\"\"Return sample data for tests.\"\"\"
    return {
        'id': 1,
        'name': 'Test Item',
        'value': 42,
        'active': True,
        'tags': ['test', 'sample', 'fixture']
    }
EOF"

# Run tests to validate fixes
echo "Running tests to validate fixes..."
ssh -i $SSH_KEY $REMOTE_SERVER "cd /opt/sutazaiapp && source venv/bin/activate && ./scripts/run_tests.sh"

echo "Dependency fix process complete!" 