#!/usr/bin/env bash
# SutazAI Comprehensive Deployment Script

set -euo pipefail

# Logging configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/opt/sutazai_project/SutazAI/logs/deployment"
DEPLOYMENT_LOG="${LOG_DIR}/deployment_${TIMESTAMP}.log"
AUDIT_LOG="${LOG_DIR}/audit_${TIMESTAMP}.log"

# Create log directories
mkdir -p "${LOG_DIR}"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${DEPLOYMENT_LOG}"
}

# Error handling function
handle_error() {
    log "ERROR: Deployment failed at stage: $1"
    exit 1
}

# Deployment stages
main() {
    log "Starting SutazAI Comprehensive Deployment"

    # 1. Pre-deployment system audit
    log "Stage 1: Pre-Deployment System Audit"
    python3 /opt/sutazai_project/SutazAI/core_system/scripts/system_comprehensive_audit.py \
        || handle_error "Pre-Deployment Audit"

    # 2. Dependency Management
    log "Stage 2: Dependency Management"
    python3 -m venv /opt/sutazai_project/SutazAI/venv \
        || handle_error "Virtual Environment Creation"
    
    source /opt/sutazai_project/SutazAI/venv/bin/activate \
        || handle_error "Virtual Environment Activation"
    
    pip install --upgrade pip setuptools wheel \
        || handle_error "Pip Upgrade"
    
    pip install -r /opt/sutazai_project/SutazAI/requirements.txt \
        || handle_error "Dependency Installation"

    # 3. Security Checks
    log "Stage 3: Security Vulnerability Scan"
    safety check -r /opt/sutazai_project/SutazAI/requirements.txt \
        || log "WARNING: Security vulnerabilities detected"

    # 4. Model and Agent Initialization
    log "Stage 4: AI Model and Agent Initialization"
    python3 -m ai_agents.supreme_ai.initialize \
        || handle_error "AI Agent Initialization"

    # 5. Backend Service Deployment
    log "Stage 5: Backend Service Deployment"
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!

    # 6. Web UI Deployment
    log "Stage 6: Web UI Deployment"
    cd /opt/sutazai_project/SutazAI/web_ui
    npm install || handle_error "NPM Dependencies"
    npm run build || handle_error "Web UI Build"
    npm start &
    WEB_UI_PID=$!

    # 7. Post-Deployment Verification
    log "Stage 7: Post-Deployment Verification"
    python3 /opt/sutazai_project/SutazAI/scripts/system_verify.py \
        || handle_error "Post-Deployment Verification"

    # 8. Final System Health Check
    log "Stage 8: Final System Health Check"
    python3 /opt/sutazai_project/SutazAI/core_system/scripts/system_comprehensive_audit.py \
        || log "WARNING: Post-Deployment Audit Detected Issues"

    log "Deployment Completed Successfully"

    # Keep track of background process PIDs
    echo "Backend PID: ${BACKEND_PID}" >> "${DEPLOYMENT_LOG}"
    echo "Web UI PID: ${WEB_UI_PID}" >> "${DEPLOYMENT_LOG}"
}

# Trap signals for graceful shutdown
trap 'kill ${BACKEND_PID} ${WEB_UI_PID}; exit 0' SIGINT SIGTERM

# Execute main deployment function
main 2>&1 | tee -a "${DEPLOYMENT_LOG}"