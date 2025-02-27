#!/bin/bash

# Sutazaiapp Synchronization Testing Script
# Validates server synchronization and deployment processes

set -euo pipefail

# Configuration
CODE_SERVER="192.168.100.28"
DEPLOY_SERVER="192.168.100.100"
SUTAZAIAPP_USER="sutazaiapp_dev"
SUTAZAIAPP_HOME="/opt/sutazaiapp"
TEST_LOG="/var/log/sutazaiapp_sync_test.log"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$TEST_LOG"
}

# Error handling function
handle_error() {
    log "ERROR: An error occurred on line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Test SSH Connectivity
test_ssh_connectivity() {
    log "Testing SSH Connectivity"
    
    # Test connection from code server to deployment server
    if ssh -q -o BatchMode=yes -o ConnectTimeout=5 \
        "$SUTAZAIAPP_USER@$DEPLOY_SERVER" exit; then
        log "✅ SSH Connection to Deployment Server Successful"
    else
        log "❌ SSH Connection to Deployment Server Failed"
        return 1
    fi
}

# Test Repository Synchronization
test_repository_sync() {
    log "Testing Repository Synchronization"
    
    # Create a test file on code server
    local test_file="sync_test_$(date +%s).txt"
    
    # Create test file
    echo "Synchronization Test $(date)" > "$SUTAZAIAPP_HOME/$test_file"
    
    # Trigger synchronization
    bash "$SUTAZAIAPP_HOME/scripts/trigger_deploy.sh"
    
    # Verify file exists on deployment server
    if ssh "$SUTAZAIAPP_USER@$DEPLOY_SERVER" \
        "test -f $SUTAZAIAPP_HOME/$test_file"; then
        log "✅ Repository Synchronization Successful"
        
        # Clean up test file
        ssh "$SUTAZAIAPP_USER@$DEPLOY_SERVER" \
            "rm $SUTAZAIAPP_HOME/$test_file"
        rm "$SUTAZAIAPP_HOME/$test_file"
    else
        log "❌ Repository Synchronization Failed"
        return 1
    fi
}

# Test Orchestrator Deployment
test_orchestrator_deployment() {
    log "Testing Orchestrator Deployment"
    
    # Verify orchestrator configuration exists
    if ssh "$SUTAZAIAPP_USER@$DEPLOY_SERVER" \
        "test -f $SUTAZAIAPP_HOME/ai_agents/superagi/config.toml"; then
        log "✅ Orchestrator Configuration Deployed"
    else
        log "❌ Orchestrator Configuration Missing"
        return 1
    fi
    
    # Test orchestrator launch script
    if ssh "$SUTAZAIAPP_USER@$DEPLOY_SERVER" \
        "bash $SUTAZAIAPP_HOME/scripts/start_superagi.sh start"; then
        log "✅ Orchestrator Launch Successful"
        
        # Wait a moment
        sleep 10
        
        # Check if PID file exists
        if ssh "$SUTAZAIAPP_USER@$DEPLOY_SERVER" \
            "test -f $SUTAZAIAPP_HOME/logs/superagi.pid"; then
            log "✅ Orchestrator Running"
            
            # Shutdown orchestrator
            ssh "$SUTAZAIAPP_USER@$DEPLOY_SERVER" \
                "bash $SUTAZAIAPP_HOME/scripts/start_superagi.sh stop"
        else
            log "❌ Orchestrator Not Running"
            return 1
        fi
    else
        log "❌ Orchestrator Launch Failed"
        return 1
    fi
}

# Main execution
main() {
    log "Starting Sutazaiapp Synchronization Test Suite"
    
    test_ssh_connectivity
    test_repository_sync
    test_orchestrator_deployment
    
    log "Synchronization Test Suite Complete"
}

main 