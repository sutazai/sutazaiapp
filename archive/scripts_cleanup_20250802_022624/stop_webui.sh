#!/bin/bash
# SutazAI Web UI Stop Script
# This script safely stops the running Web UI server

# Get absolute path to script directory using more reliable method
SCRIPTDIR="$( cd -P "$(dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPTDIR")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
LOG_FILE="${PROJECT_ROOT}/logs/webui.log"
PID_FILE="${PROJECT_ROOT}/.webui.pid"

# Create logs directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null

# Logging function
log() {
    local message="$1"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "$message"
    echo "[$timestamp] $message" >> "$LOG_FILE"
}

log "${BLUE}Stopping SutazAI Web UI server...${NC}"

# Function to check if port 3000 is in use
check_port() {
    netstat -tuln 2>/dev/null | grep -q ":3000 " && return 0 || return 1
}

# Function to kill process using port 3000
free_port() {
    log "${YELLOW}Freeing port 3000${NC}"
    # Find process using the port
    local pid=$(lsof -i:3000 -t 2>/dev/null)
    if [ -n "$pid" ]; then
        log "Found process using port 3000: PID $pid"
        # Try gracefully first
        kill -15 $pid 2>/dev/null || true
        sleep 2
        
        # If still running, force kill
        if kill -0 $pid 2>/dev/null; then
            log "Process still running, forcing termination"
            kill -9 $pid 2>/dev/null || true
            sleep 1
        fi
    fi
    
    # If port still in use, use fuser as last resort
    if check_port; then
        log "Port still in use, using fuser to release"
        fuser -k 3000/tcp 2>/dev/null || true
        sleep 1
    fi
    
    # Final check
    if check_port; then
        log "${RED}Failed to free port 3000${NC}"
        return 1
    else
        log "${GREEN}Port 3000 successfully freed${NC}"
        return 0
    fi
}

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    log "${YELLOW}No PID file found. Web UI may not be running.${NC}"
    
    # Try to find the process by name as a fallback
    PID=$(pgrep -f "node.*next")
    if [ -z "$PID" ]; then
        log "${YELLOW}Could not find any running Web UI process.${NC}"
        
        # Still check if port 3000 is in use
        if check_port; then
            free_port
        fi
        
        exit 0
    else
        log "${YELLOW}Found potential Web UI process with PID: $PID${NC}"
    fi
else
    PID=$(cat "$PID_FILE")
    log "Found PID file with process ID: $PID"
fi

# Check if the process is actually running
if [ -n "$PID" ] && ps -p "$PID" > /dev/null; then
    log "Stopping Web UI process with PID: $PID"
    
    # Send a termination signal
    kill -15 "$PID"
    
    # Wait for the process to terminate gracefully
    TIMEOUT=10
    while [ $TIMEOUT -gt 0 ] && ps -p "$PID" > /dev/null; do
        log "Waiting for Web UI process to terminate... ($TIMEOUT seconds remaining)"
        sleep 1
        TIMEOUT=$((TIMEOUT - 1))
    done
    
    # If the process is still running after the timeout, force kill it
    if ps -p "$PID" > /dev/null; then
        log "${YELLOW}Process did not terminate gracefully. Forcing termination...${NC}"
        kill -9 "$PID"
        sleep 2
    fi
    
    # Verify the process is no longer running
    if ps -p "$PID" > /dev/null; then
        log "${RED}Failed to stop Web UI process.${NC}"
    else
        log "${GREEN}Web UI process stopped successfully.${NC}"
    fi
else
    log "${YELLOW}Process with PID $PID is not running.${NC}"
fi

# Remove the PID file
if [ -f "$PID_FILE" ]; then
    rm "$PID_FILE"
    log "Removed PID file."
fi

# Also check if there's a legacy PID file in the logs directory
if [ -f "${PROJECT_ROOT}/logs/webui.pid" ]; then
    rm "${PROJECT_ROOT}/logs/webui.pid"
    log "Removed legacy PID file from logs directory."
fi

# Check for any orphaned processes
ORPHANED=$(pgrep -f "node.*next")
if [ -n "$ORPHANED" ]; then
    log "${YELLOW}Found orphaned Web UI processes: $ORPHANED. Stopping them automatically.${NC}"
    
    for ORPHAN_PID in $ORPHANED; do
        log "Stopping orphaned process with PID: $ORPHAN_PID"
        kill -15 "$ORPHAN_PID"
        sleep 2
        
        # Force kill if still running
        if ps -p "$ORPHAN_PID" > /dev/null; then
            log "Forcing termination of orphaned process with PID: $ORPHAN_PID"
            kill -9 "$ORPHAN_PID"
        fi
    done
    log "${GREEN}Orphaned processes have been stopped.${NC}"
fi

# Final check to ensure port is free
if check_port; then
    log "${YELLOW}Port 3000 still in use after stopping processes. Freeing port...${NC}"
    free_port
else
    log "${GREEN}Port 3000 is free.${NC}"
fi

log "${GREEN}Web UI server has been stopped.${NC}"
exit 0 