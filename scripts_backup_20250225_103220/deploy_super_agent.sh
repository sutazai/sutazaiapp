#!/bin/bash
set -euo pipefail

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/deploy_utils.sh"

# Add dependency checks
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi

log "INFO" "Starting Super Agent deployment"

# Verify virtual environment
verify_virtualenv || handle_error "Virtual environment verification failed"

# Verify super agent deployment
if grep -i 'sutazai' deploy_super_agent.sh; then
    echo "SutazAi found in deploy_super_agent.sh"
    exit 1
fi

# Start Super Agent
{
    log "DEBUG" "Initializing Super Agent"
    python3 "$SCRIPT_DIR/super_agent/init.py" --sutazai-acceleration=true || handle_error "Super Agent initialization failed"
    
    log "DEBUG" "Starting Super Agent service"
    python3 "$SCRIPT_DIR/super_agent/main.py" --sutazai-acceleration=true || handle_error "Super Agent service failed to start"
    
    log "DEBUG" "Starting monitoring service"
    python3 "$SCRIPT_DIR/super_agent/monitor.py" --sutazai-acceleration=true || handle_error "Monitoring service failed to start"
} | modern_progress_bar "Super Agent" "Initialization" "Service Start"

log "INFO" "Super Agent deployed successfully" 