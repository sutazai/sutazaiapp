#!/bin/bash
set -eo pipefail

echo "Running Python 3.11 compatibility checks..."
pyupgrade --py311-plus --exit-zero-even-if-changed $(find . -name "*.py" -not -path "./.venv/*" -not -path "./**/__pycache__/*")

echo "Running flake8..."
flake8 .

echo "Running mypy..."
mypy --strict --python-version 3.11 .

echo "Running black check..."
black --check --diff .

echo "All code quality checks passed successfully"
