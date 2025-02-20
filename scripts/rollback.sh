#!/bin/bash

rollback_deployment() {
    log "WARN" "Initiating rollback"
    
    # Stop services in reverse order
    systemctl stop sutazai-monitor || true
    systemctl stop ai-worker || true
    systemctl stop ai-core || true
    
    # Clean up resources
    if [ -d "venv" ]; then
        rm -rf venv
    fi
    
    log "WARN" "Rollback completed"
} 