#!/bin/bash

set -euo pipefail

# Activate virtual environment
source /opt/sutazaiapp/venv/bin/activate

# Code Audit Script
echo "Starting Comprehensive Code Audit"

# Semgrep Scan
echo "Running Semgrep..."
semgrep scan --config=auto /opt/sutazaiapp

# Pylint Scan
echo "Running Pylint..."
find /opt/sutazaiapp -name "*.py" | xargs pylint

# Mypy Type Checking
echo "Running Mypy Type Checking..."
mypy /opt/sutazaiapp

# Bandit Security Scan
echo "Running Bandit Security Scan..."
bandit -r /opt/sutazaiapp -f custom

echo "Code Audit Complete"
