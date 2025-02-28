#!/bin/bash
set -e

# Validate deployment environment
echo "[DEPLOYMENT VALIDATION]"

# Check Python version
echo "Python Version:"
python3 --version

# Verify virtual environment
echo "Virtual Environment:"
which python
python -c "import sys; print(sys.executable)"

# Check key dependencies
echo "Dependency Validation:"
pip list | grep -E "Django|google-cloud|grpcio|protobuf"

# Test critical imports
python3 -c "
import django
import google.cloud
import grpc
import protobuf
print('All critical imports successful')
"

# Final deployment status
echo "[DEPLOYMENT VALIDATION COMPLETE]"
