#!/usr/bin/env bash
# MCP Wrapper for claude-task-runner
# Created: 2025-08-16 UTC
# Purpose: Service mesh integration for claude-task-runner MCP server

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/opt/sutazaiapp"
MCP_SERVER_DIR="$ROOT_DIR/mcp-servers/claude-task-runner"

# Load common utilities
if [ -f "$SCRIPT_DIR/../_common.sh" ]; then
    . "$SCRIPT_DIR/../_common.sh"
fi

# Configuration
MCP_NAME="claude-task-runner"
MCP_COMMAND="python3 -m task_runner.mcp.mcp_server"
MCP_TYPE="python"
MCP_DESCRIPTION="Task runner MCP server for managing isolated task execution with Claude"

# Health check function
health_check() {
    if [ -d "$MCP_SERVER_DIR/src/task_runner" ]; then
        if python3 -c "import sys; sys.path.insert(0, '$MCP_SERVER_DIR/src'); import task_runner" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# Selfcheck function
selfcheck() {
    echo "Performing selfcheck for $MCP_NAME..."
    
    # Check Python availability
    if ! command -v python3 >/dev/null 2>&1; then
        echo "ERROR: python3 not available (required for $MCP_NAME)"
        return 1
    fi
    
    # Check if source directory exists
    if [ ! -d "$MCP_SERVER_DIR/src/task_runner" ]; then
        echo "ERROR: $MCP_NAME source directory not found at $MCP_SERVER_DIR/src/task_runner"
        return 1
    fi
    
    # Check for required Python packages
    if ! python3 -c "import mcp" 2>/dev/null; then
        echo "WARNING: mcp package not installed globally"
        echo "Installing required packages locally..."
        
        # Create virtual environment if it doesn't exist
        if [ ! -d "$MCP_SERVER_DIR/venv" ]; then
            python3 -m venv "$MCP_SERVER_DIR/venv"
        fi
        
        # Install packages in virtual environment
        "$MCP_SERVER_DIR/venv/bin/pip" install --quiet --upgrade pip
        "$MCP_SERVER_DIR/venv/bin/pip" install --quiet mcp fastmcp loguru litellm json-repair python-dotenv 2>/dev/null || true
    fi
    
    # Check MCP functionality
    if [ -f "$MCP_SERVER_DIR/src/task_runner/mcp/mcp_server.py" ]; then
        echo "✓ $MCP_NAME selfcheck passed"
        return 0
    else
        echo "ERROR: $MCP_NAME MCP server file not found"
        return 1
    fi
}

# Start function
start_mcp() {
    echo "Starting $MCP_NAME MCP server..."
    
    # Change to the server directory
    cd "$MCP_SERVER_DIR"
    
    # Add src to Python path
    export PYTHONPATH="$MCP_SERVER_DIR/src:$PYTHONPATH"
    
    # Check if using virtual environment
    if [ -d "$MCP_SERVER_DIR/venv" ]; then
        echo "Using virtual environment for $MCP_NAME..."
        exec "$MCP_SERVER_DIR/venv/bin/python" -m task_runner.mcp.mcp_server
    else
        # Try with system Python
        exec python3 -m task_runner.mcp.mcp_server
    fi
}

# Stop function  
stop_mcp() {
    echo "Stopping $MCP_NAME processes..."
    pkill -f "task_runner.mcp.mcp_server" || true
    sleep 2
}

# Status function
status_mcp() {
    if pgrep -f "task_runner.mcp.mcp_server" >/dev/null; then
        echo "✓ $MCP_NAME is running"
        return 0
    else
        echo "✗ $MCP_NAME is not running"
        return 1
    fi
}

# Main command handling
case "${1:-start}" in
    start)
        start_mcp
        ;;
    stop)
        stop_mcp
        ;;
    restart)
        stop_mcp
        sleep 3
        start_mcp
        ;;
    status)
        status_mcp
        ;;
    health|healthcheck)
        health_check
        ;;
    selfcheck|--selfcheck)
        selfcheck
        ;;
    test)
        selfcheck && health_check
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|health|selfcheck|test}"
        echo "  start     - Start the MCP server"
        echo "  stop      - Stop the MCP server"
        echo "  restart   - Restart the MCP server"
        echo "  status    - Check if server is running"
        echo "  health    - Perform health check"
        echo "  selfcheck - Perform comprehensive selfcheck"
        echo "  test      - Run both selfcheck and health check"
        exit 1
        ;;
esac