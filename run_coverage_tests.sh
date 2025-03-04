#!/bin/bash

# Script to run tests specifically for reaching 100% coverage

echo "Running coverage tests..."

# Activate virtual environment
cd /opt/sutazaiapp
source venv/bin/activate

# Create coverage configuration
cat > .coveragerc << EOF
[run]
source = core_system.orchestrator
omit = */__pycache__/*,*/tests/*,*/venv/*

[report]
exclude_lines = 
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
    pass
    raise ImportError
EOF

# Run tests with detailed reporting of missing lines
python -m pytest tests/ \
    --cov=core_system.orchestrator \
    --cov-config=.coveragerc \
    --cov-report=html:coverage \
    --cov-report=term-missing \
    -v

# Specifically run our targeted tests to ensure they're included
echo "Running targeted tests for better coverage..."
python -m pytest tests/test_agent_manager_targeted.py tests/test_supreme_ai_targeted.py \
    --cov=core_system.orchestrator \
    --cov-config=.coveragerc \
    --cov-report=html:coverage \
    --cov-report=term-missing \
    --cov-append \
    -v

echo "Coverage tests completed!"
echo "Check the coverage report at: /opt/sutazaiapp/coverage/index.html"
