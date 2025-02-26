#!/usr/bin/env bash
# SutazAI Documentation Update Script

set -euo pipefail

# Logging configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/opt/sutazaiapp/logs/documentation"
UPDATE_LOG="${LOG_DIR}/doc_update_${TIMESTAMP}.log"

# Create log directory
mkdir -p "${LOG_DIR}"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${UPDATE_LOG}"
}

# Error handling function
handle_error() {
    log "ERROR: Documentation update failed at stage: $1"
    exit 1
}

# Python version verification
verify_python_version() {
    log "Verifying Python 3.11"
    
    if command -v python3.11 >/dev/null 2>&1; then
        log "Python 3.11 is installed"
    else
        log "Python 3.11 not found. Installing..."
        sudo add-apt-repository ppa:deadsnakes/ppa -y
        sudo apt-get update
        sudo apt-get install -y python3.11 python3.11-dev python3.11-venv
    fi
    
    PYTHON_VERSION=$(python3.11 --version)
    log "Using ${PYTHON_VERSION}"
}

# Main documentation update workflow
main() {
    log "Starting SutazAI Documentation Update"

    # 0. Verify Python version
    verify_python_version \
        || handle_error "Python Version Verification"

    # 1. Activate virtual environment
    log "Stage 1: Activating Virtual Environment"
    source /opt/sutazaiapp/venv/bin/activate \
        || handle_error "Virtual Environment Activation"

    # 2. Install documentation requirements
    log "Stage 2: Installing Documentation Requirements"
    pip install -r /opt/sutazaiapp/requirements-docs.txt \
        || handle_error "Documentation Requirements Installation"

    # 3. Run documentation management script
    log "Stage 3: Running Documentation Management"
    python /opt/sutazaiapp/core_system/documentation_manager.py \
        || handle_error "Documentation Management"

    # 4. Generate Sphinx Documentation
    log "Stage 4: Generating Sphinx Documentation"
    cd /opt/sutazaiapp/docs
    sphinx-build -b html . _build/html \
        || handle_error "Sphinx Documentation Generation"

    log "Documentation Update Completed Successfully"
}

# Execute main documentation update function
main 2>&1 | tee -a "${UPDATE_LOG}"