#!/usr/bin/env bash
# MCP Wrapper for claude-flow
# Created: 2025-08-16 14:30:00 UTC
# Purpose: Service mesh integration for claude-flow MCP server

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/opt/sutazaiapp"

# Load common utilities
if [ -f "$SCRIPT_DIR/../_common.sh" ]; then
    . "$SCRIPT_DIR/../_common.sh"
fi

# Configuration
MCP_NAME="claude-flow"
MCP_COMMAND="npx claude-flow@alpha mcp start"
MCP_TYPE="nodejs"
MCP_DESCRIPTION="Claude Flow MCP server for swarm orchestration"

# Health check function
health_check() {
    if command -v npx >/dev/null 2>&1; then
        if timeout 10 npx claude-flow@alpha --version >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Selfcheck function
selfcheck() {
    echo "Performing selfcheck for $MCP_NAME..."
    
    # Check NPX availability
    if ! command -v npx >/dev/null 2>&1; then
        echo "ERROR: npx not available (required for $MCP_NAME)"
        return 1
    fi
    
    # Check package availability
    if ! timeout 15 npx claude-flow@alpha --version >/dev/null 2>&1; then
        echo "ERROR: claude-flow@alpha package not available or not responding"
        return 1
    fi
    
    # Check MCP functionality
    if timeout 10 npx claude-flow@alpha mcp --help >/dev/null 2>&1; then
        echo "✓ $MCP_NAME selfcheck passed"
        return 0
    else
        echo "ERROR: $MCP_NAME MCP command not responding"
        return 1
    fi
}

# Start function
start_mcp() {
    echo "Starting $MCP_NAME MCP server..."
    exec $MCP_COMMAND
}

# Stop function  
stop_mcp() {
    echo "Stopping $MCP_NAME processes..."
    pkill -f "claude-flow.*mcp.*start" || true
    sleep 2
}

# Status function
status_mcp() {
    if pgrep -f "claude-flow.*mcp.*start" >/dev/null; then
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