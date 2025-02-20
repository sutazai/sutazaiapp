#!/usr/bin/env bash
# SutazAI Documentation Update Script

set -euo pipefail

# Logging configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/opt/sutazai_project/SutazAI/logs/documentation"
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

# Main documentation update workflow
main() {
    log "Starting SutazAI Documentation Update"

    # 1. Activate virtual environment
    log "Stage 1: Activating Virtual Environment"
    source /opt/sutazai_project/SutazAI/venv/bin/activate \
        || handle_error "Virtual Environment Activation"

    # 2. Install documentation requirements
    log "Stage 2: Installing Documentation Requirements"
    pip install -r /opt/sutazai_project/SutazAI/requirements-docs.txt \
        || handle_error "Documentation Requirements Installation"

    # 3. Run documentation management script
    log "Stage 3: Running Documentation Management"
    python3 /opt/sutazai_project/SutazAI/core_system/documentation_manager.py \
        || handle_error "Documentation Management"

    # 4. Generate Sphinx Documentation
    log "Stage 4: Generating Sphinx Documentation"
    cd /opt/sutazai_project/SutazAI/docs
    sphinx-build -b html . _build/html \
        || handle_error "Sphinx Documentation Generation"

    log "Documentation Update Completed Successfully"
}

# Execute main documentation update function
main 2>&1 | tee -a "${UPDATE_LOG}"