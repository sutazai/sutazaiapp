#!/bin/bash
# Simple script to run all tests with coverage

set -e  # Exit on error

echo "Activating virtual environment..."
source venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

echo "Running tests with coverage..."
python -m pytest --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail tests/ \
    --cov=core_system.orchestrator \
    --cov-report=html:coverage \
    --cov-report=term \
    -v

echo "All tests completed successfully!"
echo "Coverage report is available at coverage/index.html" 