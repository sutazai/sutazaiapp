#!/bin/bash
set -e

# Deployment script for SutazAI

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[DEPLOY]${NC} $1"
}

# Error handling function
error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Validate system requirements
validate_system() {
    log "Checking system requirements..."
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    if [[ ! "$PYTHON_VERSION" =~ ^3\.(9|10|11)\. ]]; then
        error "Unsupported Python version. Required: 3.9-3.11, Current: $PYTHON_VERSION"
    fi
    
    # Check virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        error "Virtual environment not activated"
    fi
}

# Create virtual environment
setup_venv() {
    log "Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
}

# Install dependencies
install_dependencies() {
    log "Installing dependencies..."
    pip install -r requirements.txt --verbose --no-cache-dir
}

# Run system validation
run_system_checks() {
    log "Running system validation..."
    python scripts/dependency_validator.py
    python system_optimizer.py
}

# Main deployment function
deploy() {
    log "Starting SutazAI deployment..."
    
    validate_system
    setup_venv
    install_dependencies
    run_system_checks
    
    log "Deployment completed successfully! 🚀"
}

# Execute deployment
deploy