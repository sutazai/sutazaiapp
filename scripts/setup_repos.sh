#!/bin/bash

# Sutazaiapp Repository Setup Script
# Manages manual repository synchronization and setup

set -euo pipefail

# Configuration
SUTAZAIAPP_HOME="/opt/sutazaiapp"
REPO_URL="https://github.com/chrissuta/sutazaiapp.git"
SETUP_LOG="/var/log/sutazaiapp_repo_setup.log"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SETUP_LOG"
}

# Error handling function
handle_error() {
    log "ERROR: An error occurred on line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Validate repository
validate_repository() {
    log "Validating repository"
    
    if [[ ! -d "$SUTAZAIAPP_HOME/.git" ]]; then
        log "Repository not found. Cloning from remote."
        git clone "$REPO_URL" "$SUTAZAIAPP_HOME"
    else
        cd "$SUTAZAIAPP_HOME"
        git fetch origin
        git reset --hard origin/main
    fi
}

# Setup virtual environment
setup_venv() {
    log "Setting up Python virtual environment"
    
    if [[ ! -d "$SUTAZAIAPP_HOME/venv" ]]; then
        python3.11 -m venv "$SUTAZAIAPP_HOME/venv"
    fi
    
    source "$SUTAZAIAPP_HOME/venv/bin/activate"
    pip install --upgrade pip setuptools wheel
    pip install -r "$SUTAZAIAPP_HOME/requirements.txt"
}

# Perform system checks
system_checks() {
    log "Performing system checks"
    
    # Check Python version
    python_version=$(python3 --version)
    if [[ ! "$python_version" =~ "3.11" ]]; then
        log "WARNING: Python 3.11 not detected. Current version: $python_version"
    fi
    
    # Check required directories
    required_dirs=(
        "ai_agents"
        "model_management"
        "backend"
        "web_ui"
        "scripts"
        "packages"
        "logs"
        "doc_data"
        "docs"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$SUTAZAIAPP_HOME/$dir" ]]; then
            mkdir -p "$SUTAZAIAPP_HOME/$dir"
            log "Created missing directory: $dir"
        fi
    done
}

# Main execution
main() {
    log "Starting Sutazaiapp Repository Setup"
    
    validate_repository
    setup_venv
    system_checks
    
    log "Repository Setup Complete"
}

main 