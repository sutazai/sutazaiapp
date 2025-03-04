#!/bin/bash
# Script to run targeted coverage tests

set -e  # Exit on error

echo "Activating virtual environment..."
source venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Install any missing dependencies
echo "Installing required dependencies..."
pip install pytest pytest-asyncio pytest-cov pytest-mock aiohttp fastapi httpx sqlalchemy

# Make sure the coverage directory exists
mkdir -p coverage

echo "Running coverage tests for targeted files..."
python -m pytest --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail tests/test_agent_manager_complete_coverage.py \
       tests/test_supreme_ai_complete_coverage.py \
       tests/test_task_queue_complete_coverage.py \
       tests/test_sync_manager_complete_coverage.py \
       -v \
       --no-header \
       --cov=core_system.orchestrator \
       --cov-report=html:coverage \
       --cov-report=term

echo "All tests completed!"
echo "Coverage report is available at coverage/index.html"
