#!/bin/bash
# SutazAI Supreme Orchestrator Stop Script

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Stopping Orchestrator...${NC}"

# Check for PID file
PID_FILE="${PROJECT_ROOT}/logs/orchestrator.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null; then
        echo -e "${YELLOW}Found orchestrator process: $PID${NC}"
        
        # Send SIGTERM first for graceful shutdown
        kill -15 "$PID" 2>/dev/null
        
        # Wait for process to exit
        echo -e "${YELLOW}Waiting for process to exit (5 seconds)...${NC}"
        for i in {1..5}; do
            if ! ps -p "$PID" > /dev/null; then
                echo -e "${GREEN}✓ Orchestrator stopped${NC}"
                rm -f "$PID_FILE"
                exit 0
            fi
            sleep 1
        done
        
        # Force kill if still running
        echo -e "${YELLOW}Process still running, forcing termination...${NC}"
        kill -9 "$PID" 2>/dev/null
        sleep 1
        
        if ! ps -p "$PID" > /dev/null; then
            echo -e "${GREEN}✓ Orchestrator forcefully stopped${NC}"
        else
            echo -e "${RED}✗ Failed to stop orchestrator process${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Process $PID is not running${NC}"
    fi
    
    # Clean up PID file
    rm -f "$PID_FILE"
else
    # Try to find orchestrator processes
    ORCHESTRATOR_PID=$(pgrep -f "python.*supreme_orchestrator.py" || true)
    
    if [ -n "$ORCHESTRATOR_PID" ]; then
        echo -e "${YELLOW}Found orchestrator process: $ORCHESTRATOR_PID${NC}"
        
        # Send SIGTERM first for graceful shutdown
        kill -15 "$ORCHESTRATOR_PID" 2>/dev/null
        
        # Wait for process to exit
        echo -e "${YELLOW}Waiting for process to exit (5 seconds)...${NC}"
        for i in {1..5}; do
            if ! ps -p "$ORCHESTRATOR_PID" > /dev/null; then
                echo -e "${GREEN}✓ Orchestrator stopped${NC}"
                exit 0
            fi
            sleep 1
        done
        
        # Force kill if still running
        echo -e "${YELLOW}Process still running, forcing termination...${NC}"
        kill -9 "$ORCHESTRATOR_PID" 2>/dev/null
        sleep 1
        
        if ! ps -p "$ORCHESTRATOR_PID" > /dev/null; then
            echo -e "${GREEN}✓ Orchestrator forcefully stopped${NC}"
        else
            echo -e "${RED}✗ Failed to stop orchestrator process${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}! No orchestrator process found${NC}"
    fi
fi

exit 0 