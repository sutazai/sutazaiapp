#!/bin/bash

# Set strict error handling
set -e
set -o pipefail

# Ensure we're in the correct virtual environment
source sutazai_env/bin/activate

# Safely clear pip cache with error handling
pip cache purge || true

# Remove existing package index (with error suppression)
rm -rf sutazai_env/lib/python3.11/site-packages/pip/_internal/index-url || true

# Upgrade pip and setuptools with verbose output
python -m pip install --upgrade pip setuptools wheel --verbose

# Create a comprehensive log directory
mkdir -p /var/log/sutazai

# Run advanced dependency resolver with logging
python scripts/advanced_dependency_resolver.py 2>&1 | tee /var/log/sutazai/dependency_resolution.log

# Verify package installations
pip list

# Optional: Run safety check with relaxed constraints
safety check --ignore 44715 --ignore 51668 || true 