#!/bin/bash
# SutazAI Comprehensive Dependency Installation Script

# Exit on any error
set -e

# Logging
LOG_FILE="/opt/sutazaiapp/logs/dependency_install.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Error handling
handle_error() {
    log "ERROR: $*"
    exit 1
}

# Ensure we're using Python 3.11
PYTHON_CMD="python3.11"

# Check Python version
log "Checking Python version..."
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" != "3.11" ]]; then
    handle_error "Python 3.11 is required. Current version: $PYTHON_VERSION"
fi

# Ensure virtual environment is activated or create one
if [ -z "$VIRTUAL_ENV" ]; then
    log "Creating virtual environment..."
    $PYTHON_CMD -m venv /opt/sutazaiapp/venv || handle_error "Failed to create virtual environment"
    # shellcheck source=/dev/null
    source /opt/sutazaiapp/venv/bin/activate
fi

# Upgrade pip, setuptools, and wheel
log "Upgrading pip, setuptools, and wheel..."
$PYTHON_CMD -m pip install --upgrade pip setuptools wheel || handle_error "Failed to upgrade pip, setuptools, and wheel"

# Install system dependencies
log "Installing system dependencies..."
apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev \
    libmagic1 \
    tesseract-ocr \
    libgl1-mesa-glx \
    || handle_error "Failed to install system dependencies"

# Ensure core system directories exist
log "Checking core system directories..."
mkdir -p /opt/sutazaiapp/core_system
mkdir -p /opt/sutazaiapp/misc
mkdir -p /opt/sutazaiapp/system_integration

# Install dependencies without resolving to avoid conflicts
log "Installing Python dependencies (first pass)..."
$PYTHON_CMD -m pip install --no-deps -r /opt/sutazaiapp/requirements.txt || handle_error "Failed to install dependencies (first pass)"

# Install remaining dependencies
log "Installing remaining dependencies (second pass)..."
$PYTHON_CMD -m pip install -r /opt/sutazaiapp/requirements.txt || handle_error "Failed to install dependencies (second pass)"

# Verify installation
log "Verifying dependencies..."
$PYTHON_CMD -m pip list

# Run safety check
log "Running dependency safety check..."
$PYTHON_CMD -m safety check || log "WARNING: Safety check found potential issues"

log "SutazAI dependencies installed successfully!"

# Create a compatibility summary
log "Generating compatibility summary..."
$PYTHON_CMD -c "
import sys
import platform

print('=============== COMPATIBILITY CHECK SUMMARY ===============')
print(f'Python Version: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Python Implementation: {platform.python_implementation()}')
print('=========================================================')
" > /opt/sutazaiapp/logs/compatibility_summary.log

exit 0 