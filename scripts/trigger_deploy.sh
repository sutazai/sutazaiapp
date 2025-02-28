#!/bin/bash

# Sutazaiapp Deployment Trigger Script
# Manages automated deployment and repository synchronization

set -euo pipefail

# Configuration
SUTAZAIAPP_HOME="/opt/sutazaiapp"
CODE_SERVER="192.168.100.28"
DEPLOY_SERVER="192.168.100.100"
SUTAZAIAPP_USER="sutazaiapp_dev"
DEPLOY_LOG="/var/log/sutazaiapp_deploy.log"
SYNC_TIMESTAMP_FILE="$SUTAZAIAPP_HOME/.last_sync"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DEPLOY_LOG"
}

# Error handling function
handle_error() {
    log "ERROR: An error occurred on line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Validate prerequisites
validate_prerequisites() {
    if [[ ! -f "$SUTAZAIAPP_HOME/.ssh/sutazaiapp_sync_key" ]]; then
        log "ERROR: SSH key not found. Run server_sync.sh first."
        exit 1
    fi
}

# Synchronize repositories
sync_repositories() {
    log "Starting repository synchronization"
    
    # Ensure local repository is up to date
    cd "$SUTAZAIAPP_HOME"
    git fetch origin
    git pull origin main
    
    # Synchronize to deployment server
    rsync -avz \
        -e "ssh -i $SUTAZAIAPP_HOME/.ssh/sutazaiapp_sync_key" \
        --exclude='.git' \
        --exclude='venv' \
        --exclude='logs' \
        "$SUTAZAIAPP_HOME/" \
        "$SUTAZAIAPP_USER@$DEPLOY_SERVER:$SUTAZAIAPP_HOME/"
    
    # Update last sync timestamp
    date +%s > "$SYNC_TIMESTAMP_FILE"
    
    log "Repository synchronization complete"
}

# Trigger deployment on remote server
trigger_remote_deployment() {
    log "Triggering remote deployment"
    
    ssh -i "$SUTAZAIAPP_HOME/.ssh/sutazaiapp_sync_key" \
        "$SUTAZAIAPP_USER@$DEPLOY_SERVER" \
        "bash $SUTAZAIAPP_HOME/scripts/setup_repos.sh"
}

# Main execution
main() {
    log "Starting Sutazaiapp Deployment Process"
    
    validate_prerequisites
    sync_repositories
    trigger_remote_deployment
    
    log "Deployment Process Complete"
}

main 