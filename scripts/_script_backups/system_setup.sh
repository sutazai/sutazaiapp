#!/bin/bash

# Comprehensive system setup script
set -e
set -o pipefail

# Create log directory
mkdir -p /var/log/sutazai

# Ensure we're in the correct virtual environment
source sutazai_env/bin/activate

# Upgrade pip with verbose output and no cache
python -m pip install --upgrade pip setuptools wheel --no-cache-dir --verbose 2>&1 | tee /var/log/sutazai/pip_upgrade.log

# Run dependency management with comprehensive logging
bash scripts/dependency_management.sh 2>&1 | tee /var/log/sutazai/system_setup.log

# Run system validator
python scripts/system_validator.py 2>&1 | tee /var/log/sutazai/system_validator.log

# Run database migrations
alembic upgrade head 2>&1 | tee /var/log/sutazai/database_migration.log

# Start application with comprehensive logging
uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --reload 2>&1 | tee /var/log/sutazai/application.log 