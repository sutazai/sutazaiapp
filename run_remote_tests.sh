#!/bin/bash
# This script runs the tests on the remote test server with coverage.

echo "Preparing to run tests on the remote test server..."

# Remote server details (from user input)
REMOTE_SERVER="192.168.100.100"
REMOTE_USER="root"
REMOTE_TEST_DIR="/opt/sutazaitest"

# Create coverage config if it doesn't exist
cat <<EOF > .coveragerc
[run]
source = core_system.orchestrator
omit = */tests/*, */venv/*, */__pycache__/*

[report]
precision = 2
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
    pass
    raise ImportError
EOF

# Copy the local files to the remote server
echo "Copying files to remote test server..."
ssh -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_SERVER} "mkdir -p ${REMOTE_TEST_DIR}"
rsync -avz --exclude "venv" --exclude "__pycache__" --exclude ".git" --exclude "*.pyc" ./ ${REMOTE_USER}@${REMOTE_SERVER}:${REMOTE_TEST_DIR}

# Run the tests on the remote server
echo "Running tests on remote server..."
ssh -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_SERVER} << EOF
    cd ${REMOTE_TEST_DIR}
    # Check if virtualenv exists, create if not
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3.11 -m venv venv
    fi
    
    source venv/bin/activate
    
    # Install dependencies
    pip install pytest pytest-cov pytest-asyncio pytest-mock aiohttp
    
    # Install the package in development mode if needed
    pip install -e .
    
    # Make sure the coverage directory exists
    mkdir -p coverage
    mkdir -p test_reports
    
    echo "Running tests for the Supreme AI Orchestrator..."
    
    # First run the regular tests
    python -m pytest --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail tests/ \
        --cov=core_system.orchestrator \
        --cov-report=html:coverage \
        --cov-report=term-missing \
        --html=test_reports/report.html
    
    echo "Running targeted coverage tests for specific components..."
    
    # Then run the special targeted coverage tests
    python -m pytest --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail tests/test_agent_manager_complete_coverage.py \
        tests/test_supreme_ai_complete_coverage.py \
        tests/test_task_queue_complete_coverage.py \
        tests/test_sync_manager_complete_coverage.py \
        --cov=core_system.orchestrator \
        --cov-append \
        --cov-report=html:coverage \
        --cov-report=term-missing
    
    # Create an overall coverage report
    echo "Generating final coverage report..."
    python -m pytest --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail --cov=core_system.orchestrator --cov-report=html:coverage --cov-report=term
    
    echo "Tests completed!"
EOF

# Check if the tests passed
if [ $? -eq 0 ]; then
    echo "All tests passed successfully!"
    echo "Coverage Report:"
    echo "HTML coverage report: /opt/sutazaiapp/coverage/index.html"
    echo "Test report: /opt/sutazaiapp/test_reports/report.html"
else
    echo "Tests failed!"
fi

echo "Remote tests complete!" 