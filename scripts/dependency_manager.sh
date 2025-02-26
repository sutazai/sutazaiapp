#!/usr/bin/env bash
# Comprehensive Dependency Management Script for SutazAI

set -euo pipefail

# Logging
LOG_FILE="/var/log/sutazai_dependency_management.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="/opt/sutazaiapp"

# Function to log messages
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Update system packages
update_system_packages() {
    log "${YELLOW}Updating System Packages${NC}"
    sudo apt-get update
    sudo apt-get upgrade -y
    log "${GREEN}System packages updated${NC}"
}

# Install system dependencies
install_system_dependencies() {
    log "${YELLOW}Installing System Dependencies${NC}"
    
    # Essential build tools
    sudo apt-get install -y \
        build-essential \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        git \
        curl \
        wget \
        software-properties-common

    # OCR and document processing dependencies
    sudo apt-get install -y \
        tesseract-ocr \
        libtesseract-dev \
        poppler-utils \
        libmagic-dev

    # Graphics and image processing libraries
    sudo apt-get install -y \
        libopencv-dev \
        libgl1-mesa-glx

    log "${GREEN}System dependencies installed${NC}"
}

# Create Python virtual environment
create_python_venv() {
    log "${YELLOW}Creating Python Virtual Environment${NC}"
    
    # Remove existing venv if it exists
    rm -rf "$PROJECT_ROOT/venv"
    
    # Create new virtual environment with Python 3.11
    python3.11 -m venv "$PROJECT_ROOT/venv"
    
    # Activate venv and upgrade pip
    source "$PROJECT_ROOT/venv/bin/activate"
    pip install --upgrade pip setuptools wheel
    deactivate
    
    log "${GREEN}Python virtual environment created${NC}"
}

# Install Python dependencies
install_python_dependencies() {
    log "${YELLOW}Installing Python Dependencies${NC}"
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Install dependencies with timeout and retry
    pip install --timeout 100 -r "$PROJECT_ROOT/requirements.txt" || \
    pip install --timeout 100 -r "$PROJECT_ROOT/requirements.txt"
    
    # Install additional performance packages
    pip install \
        cython \
        numpy \
        numba \
        py-spy \
        memory_profiler
    
    deactivate
    
    log "${GREEN}Python dependencies installed${NC}"
}

# Install Node.js dependencies
install_nodejs_dependencies() {
    log "${YELLOW}Installing Node.js Dependencies${NC}"
    
    # Check if package.json exists
    if [ -f "$PROJECT_ROOT/web_ui/package.json" ]; then
        cd "$PROJECT_ROOT/web_ui"
        npm install
        npm audit fix
        cd "$PROJECT_ROOT"
    else
        log "${RED}No package.json found in web_ui directory${NC}"
    fi
    
    log "${GREEN}Node.js dependencies installed${NC}"
}

# Verify Python 3.11
verify_python_version() {
    log "${YELLOW}Verifying Python Version${NC}"
    
    # Check if Python 3.11 is installed
    if command -v python3.11 >/dev/null 2>&1; then
        log "${GREEN}Python 3.11 is installed${NC}"
    else
        log "${RED}Python 3.11 is not installed. Installing...${NC}"
        sudo add-apt-repository ppa:deadsnakes/ppa -y
        sudo apt-get update
        sudo apt-get install -y python3.11 python3.11-dev python3.11-venv
    fi
    
    # Verify Python version
    PYTHON_VERSION=$(python3.11 --version)
    log "${GREEN}Using $PYTHON_VERSION${NC}"
}

# Main dependency management function
main() {
    log "${GREEN}Starting SutazAI Dependency Management${NC}"
    
    update_system_packages
    verify_python_version
    install_system_dependencies
    create_python_venv
    install_python_dependencies
    install_nodejs_dependencies
    
    log "${GREEN}Dependency Management Complete!${NC}"
}

# Run the main function
main 