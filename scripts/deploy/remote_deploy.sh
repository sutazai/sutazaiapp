#!/bin/bash
# Remote deployment script for SutazAI

# Run on deployment server
set -e

echo "==============================================="
echo "SutazAI Remote Deployment"
echo "==============================================="
echo "Start time: $(date)"
echo "Running on: $(hostname)"
echo

# Project root
PROJECT_ROOT="/opt/sutazaiapp"

# Check if we are on the deployment server
if [[ "$(hostname -I | awk '{print $1}')" != "192.168.100.100" ]]; then
    echo "ERROR: This script should only be run on the deployment server."
    exit 1
fi

# Function to handle errors
handle_error() {
    echo "ERROR: $1"
    exit 1
}

# Step 0: Set up Python environment
echo "Step 0: Setting up Python environment..."
cd $PROJECT_ROOT

# Remove existing virtual environment if it exists
if [ -d "/opt/venv-sutazaiapp" ]; then
    echo "Removing existing virtual environment..."
    rm -rf /opt/venv-sutazaiapp
fi

# Create a new virtual environment
echo "Creating new virtual environment with Python 3.11..."
python3.11 -m venv /opt/venv-sutazaiapp || handle_error "Failed to create virtual environment"

# Activate the virtual environment
echo "Activating virtual environment..."
source /opt/venv-sutazaiapp/bin/activate || handle_error "Failed to activate virtual environment"

# Upgrade pip and essential tools
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel || handle_error "Failed to upgrade pip"

# Install PyYAML first to ensure correct version
echo "Installing PyYAML 6.0 specifically..."
pip install PyYAML==6.0 || handle_error "Failed to install PyYAML"

# Install superagi-tools separately to ensure compatibility
echo "Installing superagi-tools 1.0.8..."
pip install superagi-tools==1.0.8 || handle_error "Failed to install superagi-tools"

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt || handle_error "Failed to install dependencies"

echo "Python environment setup completed successfully."
echo

# Step 1: Build the application
echo "Step 1: Building the application..."
cd $PROJECT_ROOT
if [ -f "./scripts/build.sh" ]; then
    ./scripts/build.sh || handle_error "Build failed"
    echo "Build completed successfully."
else
    echo "No build script found, skipping build step."
fi
echo

# Step 2: Run tests (optional)
echo "Step 2: Running tests..."
if [ -f "./scripts/run_tests.sh" ]; then
    ./scripts/run_tests.sh || echo "Tests failed but continuing deployment."
    echo "Tests completed."
else
    echo "No test script found, skipping tests."
fi
echo

# Step 3: Start the application
echo "Step 3: Starting the application..."
if [ -f "./scripts/start.sh" ]; then
    ./scripts/start.sh || handle_error "Start failed"
    echo "Application started successfully."
else
    echo "No start script found, skipping application start."
fi
echo

echo "==============================================="
echo "Remote deployment completed successfully."
echo "End time: $(date)"
echo "==============================================="
