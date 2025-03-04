#!/bin/bash
# Script to run a specific test file for diagnosis

if [ $# -eq 0 ]; then
    echo "Usage: ./run_test.sh <test_file_path>"
    echo "Example: ./run_test.sh tests/test_agent_manager_targeted.py"
    exit 1
fi

source venv/bin/activate
python -m pytest --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail $1 -v
