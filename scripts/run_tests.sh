#!/bin/bash
set -e

# Activate virtual environment
source /opt/venv-sutazaiapp/bin/activate

# Run unit tests
echo "Running unit tests..."
pytest tests/unit -v

# Run integration tests (only if explicitly requested)
if [ "${RUN_INTEGRATION_TESTS:-false}" = "true" ]; then
    echo "Running integration tests..."
    pytest tests/integration -v
else
    echo "Skipping integration tests. Set RUN_INTEGRATION_TESTS=true to run them."
fi

echo "Tests completed." 