#!/bin/bash

# Supreme AI Orchestrator Launch Script
# Manages offline AI agent orchestration

set -euo pipefail

# Configuration
SUTAZAIAPP_HOME="/opt/sutazaiapp"
VENV_PATH="$SUTAZAIAPP_HOME/venv"
CONFIG_PATH="$SUTAZAIAPP_HOME/ai_agents/superagi/config.toml"
LOG_FILE="$SUTAZAIAPP_HOME/logs/superagi_orchestrator.log"
PID_FILE="$SUTAZAIAPP_HOME/logs/superagi.pid"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Error handling function
handle_error() {
    log "ERROR: An error occurred on line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Validate prerequisites
validate_prerequisites() {
    if [[ ! -f "$CONFIG_PATH" ]]; then
        log "ERROR: Configuration file not found at $CONFIG_PATH"
        exit 1
    fi

    if [[ ! -d "$VENV_PATH" ]]; then
        log "ERROR: Virtual environment not found at $VENV_PATH"
        exit 1
    fi
}

# Check if orchestrator is already running
check_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log "Orchestrator already running with PID $pid"
            exit 1
        else
            rm "$PID_FILE"
        fi
    fi
}

# Launch orchestrator
launch_orchestrator() {
    log "Launching Supreme AI Orchestrator"
    
    source "$VENV_PATH/bin/activate"
    
    # Use nohup to keep process running after terminal closes
    nohup python3 -m superagi.orchestrator \
        --config "$CONFIG_PATH" \
        --log-file "$LOG_FILE" \
        > /dev/null 2>&1 & 

    local pid=$!
    echo "$pid" > "$PID_FILE"
    
    log "Orchestrator launched with PID $pid"
}

# Shutdown orchestrator
shutdown_orchestrator() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        log "Shutting down orchestrator (PID: $pid)"
        
        if kill -SIGTERM "$pid" 2>/dev/null; then
            # Wait for graceful shutdown
            sleep 5
            
            # Force kill if not terminated
            kill -SIGKILL "$pid" 2>/dev/null || true
            
            rm "$PID_FILE"
            log "Orchestrator shutdown complete"
        else
            log "Could not terminate orchestrator"
        fi
    else
        log "No running orchestrator found"
    fi
}

# Main execution
main() {
    validate_prerequisites

    case "${1:-start}" in
        "start")
            check_running
            launch_orchestrator
            ;;
        "stop")
            shutdown_orchestrator
            ;;
        "restart")
            shutdown_orchestrator
            launch_orchestrator
            ;;
        *)
            log "Invalid command. Use start, stop, or restart."
            exit 1
            ;;
    esac
}

main "$@" 