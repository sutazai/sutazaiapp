#!/bin/bash

# Supreme AI Orchestrator Service Manager
# This script manages the orchestrator service, including start, stop, and status checks.

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/venv"
PYTHON="$VENV_PATH/bin/python3.11"
ORCHESTRATOR_SCRIPT="$SCRIPT_DIR/start_orchestrator.py"
PID_FILE="/tmp/supreme_ai_orchestrator.pid"
LOG_FILE="$PROJECT_ROOT/logs/orchestrator.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if the service is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to start the service
start() {
    echo -e "${YELLOW}Starting Supreme AI Orchestrator...${NC}"
    
    # Check if already running
    if is_running; then
        echo -e "${RED}Error: Orchestrator is already running with PID $(cat "$PID_FILE")${NC}"
        return 1
    fi

    # Ensure virtual environment exists
    if [ ! -d "$VENV_PATH" ]; then
        echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
        return 1
    fi

    # Create logs directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"

    # Start the orchestrator
    cd "$PROJECT_ROOT" || exit 1
    nohup "$PYTHON" "$ORCHESTRATOR_SCRIPT" >> "$LOG_FILE" 2>&1 &
    pid=$!
    echo $pid > "$PID_FILE"
    
    # Wait briefly and check if process is still running
    sleep 2
    if is_running; then
        echo -e "${GREEN}Supreme AI Orchestrator started successfully with PID $pid${NC}"
        return 0
    else
        echo -e "${RED}Error: Failed to start Supreme AI Orchestrator${NC}"
        return 1
    fi
}

# Function to stop the service
stop() {
    echo -e "${YELLOW}Stopping Supreme AI Orchestrator...${NC}"
    
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            kill "$pid"
            sleep 2
            
            # Check if process is still running
            if ps -p "$pid" > /dev/null 2>&1; then
                echo -e "${RED}Process did not stop gracefully, forcing...${NC}"
                kill -9 "$pid"
            fi
            
            rm -f "$PID_FILE"
            echo -e "${GREEN}Supreme AI Orchestrator stopped${NC}"
        else
            echo -e "${YELLOW}Process not running but PID file exists. Cleaning up...${NC}"
            rm -f "$PID_FILE"
        fi
    else
        echo -e "${RED}PID file not found. Service may not be running.${NC}"
        return 1
    fi
}

# Function to check service status
status() {
    if is_running; then
        pid=$(cat "$PID_FILE")
        echo -e "${GREEN}Supreme AI Orchestrator is running with PID $pid${NC}"
        return 0
    else
        echo -e "${RED}Supreme AI Orchestrator is not running${NC}"
        return 1
    fi
}

# Function to view logs
logs() {
    if [ -f "$LOG_FILE" ]; then
        tail ${1:--f} "$LOG_FILE"
    else
        echo -e "${RED}Log file not found at $LOG_FILE${NC}"
        return 1
    fi
}

# Function to restart the service
restart() {
    echo -e "${YELLOW}Restarting Supreme AI Orchestrator...${NC}"
    stop
    sleep 2
    start
}

# Main script logic
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs "$2"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac

exit $? 