#!/bin/bash
# Universal health check script for base image

# Check if Python is available
if command -v python &> /dev/null; then
    python -c "import sys; sys.exit(0)"
    PYTHON_STATUS=$?
else
    PYTHON_STATUS=1
fi

# Check if Node.js is available
if command -v node &> /dev/null; then
    node -e "process.exit(0)"
    NODE_STATUS=$?
else
    NODE_STATUS=1
fi

# If either Python or Node is healthy, consider the container healthy
if [ $PYTHON_STATUS -eq 0 ] || [ $NODE_STATUS -eq 0 ]; then
    exit 0
else
    exit 1
fi