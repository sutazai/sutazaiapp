#!/bin/bash
# SutazAI Backend Stop Script
# This script safely stops the running backend server

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
LOG_FILE="${PROJECT_ROOT}/logs/backend.log"
PID_FILE="${PROJECT_ROOT}/.backend.pid"

# Logging function
log() {
    local message="$1"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "$message"
    echo "[$timestamp] $message" >> "$LOG_FILE"
}

log "${BLUE}Stopping SutazAI backend server...${NC}"

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    log "${YELLOW}No PID file found. Backend may not be running.${NC}"
    
    # Try to find the process by name as a fallback
    PID=$(pgrep -f "python.*main.py")
    if [ -z "$PID" ]; then
        log "${YELLOW}Could not find any running backend process.${NC}"
        exit 0
    else
        log "${YELLOW}Found potential backend process with PID: $PID${NC}"
    fi
else
    PID=$(cat "$PID_FILE")
    log "Found PID file with process ID: $PID"
fi

# Check if the process is actually running
if [ -n "$PID" ] && ps -p "$PID" > /dev/null; then
    log "Stopping backend process with PID: $PID"
    
    # Send a termination signal
    kill -15 "$PID"
    
    # Wait for the process to terminate gracefully
    TIMEOUT=30
    while [ $TIMEOUT -gt 0 ] && ps -p "$PID" > /dev/null; do
        log "Waiting for backend process to terminate... ($TIMEOUT seconds remaining)"
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
        log "${RED}Failed to stop backend process.${NC}"
        exit 1
    else
        log "${GREEN}Backend process stopped successfully.${NC}"
    fi
else
    log "${YELLOW}Process with PID $PID is not running.${NC}"
fi

# Remove the PID file
if [ -f "$PID_FILE" ]; then
    rm "$PID_FILE"
    log "Removed PID file."
fi

# Check for any orphaned processes
ORPHANED=$(pgrep -f "python.*main.py")
if [ -n "$ORPHANED" ]; then
    log "${YELLOW}Warning: Found orphaned backend processes: $ORPHANED${NC}"
    
    read -p "Do you want to stop these processes? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
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
fi

log "${GREEN}Backend server has been stopped.${NC}"
exit 0 