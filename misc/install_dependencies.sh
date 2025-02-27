#!/bin/bash
set -e

# Ensure we're using Python 3.11
PYTHON_CMD="python3.11"

# Function to check command success
check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    fi
}

# Upgrade pip and setuptools safely
${PYTHON_CMD} -m ensurepip --upgrade
check_success "pip upgrade"

${PYTHON_CMD} -m pip install --upgrade pip setuptools wheel
check_success "setuptools and wheel upgrade"

# Verify Python and pip versions
echo "Python version:"
${PYTHON_CMD} --version
check_success "Python version check"

echo "Pip version:"
${PYTHON_CMD} -m pip --version
check_success "Pip version check"

# Install the project in editable mode with development dependencies
${PYTHON_CMD} -m pip install -e '.[dev]'
check_success "Project installation"

# Additional safety checks
${PYTHON_CMD} -m pip install safety
check_success "Safety installation"

${PYTHON_CMD} -m safety check
check_success "Safety dependency check"

# Run linters
${PYTHON_CMD} -m pylint core_system misc system_integration
check_success "Pylint checks"

echo "🎉 Dependency installation and checks completed successfully!" 