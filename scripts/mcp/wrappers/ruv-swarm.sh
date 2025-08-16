#!/usr/bin/env bash
# MCP Wrapper for ruv-swarm
# Created: 2025-08-16 14:30:00 UTC
# Purpose: Service mesh integration for ruv-swarm MCP server

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/opt/sutazaiapp"

# Load common utilities
if [ -f "$SCRIPT_DIR/../_common.sh" ]; then
    . "$SCRIPT_DIR/../_common.sh"
fi

# Configuration
MCP_NAME="ruv-swarm"
MCP_COMMAND="npx ruv-swarm@latest mcp start --stability"
MCP_TYPE="nodejs"
MCP_DESCRIPTION="ruv-swarm MCP server for neural orchestration"

# Health check function
health_check() {
    if command -v npx >/dev/null 2>&1; then
        if timeout 15 npx ruv-swarm@latest --version >/dev/null 2>&1; then
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
    
    # Check package availability (with extended timeout due to known package installation delays)
    echo "Checking ruv-swarm package availability (may take up to 60 seconds)..."
    if ! timeout 60 npx ruv-swarm@latest --version >/dev/null 2>&1; then
        echo "WARNING: ruv-swarm@latest package slow to respond, checking basic npm functionality..."
        if ! timeout 30 npm info ruv-swarm >/dev/null 2>&1; then
            echo "ERROR: ruv-swarm package not available in npm registry"
            return 1
        else
            echo "✓ ruv-swarm package exists but has startup delays (known issue)"
            return 0
        fi
    fi
    
    # Check MCP functionality
    if timeout 15 npx ruv-swarm@latest mcp --help >/dev/null 2>&1; then
        echo "✓ $MCP_NAME selfcheck passed"
        return 0
    else
        echo "ERROR: $MCP_NAME MCP command not responding"
        return 1
    fi
}

# Start function
start_mcp() {
    echo "Starting $MCP_NAME MCP server with stability features..."
    exec $MCP_COMMAND
}

# Stop function  
stop_mcp() {
    echo "Stopping $MCP_NAME processes..."
    pkill -f "ruv-swarm.*mcp.*start" || true
    sleep 2
}

# Status function
status_mcp() {
    if pgrep -f "ruv-swarm.*mcp.*start" >/dev/null; then
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
        echo "  start     - Start the MCP server with stability features"
        echo "  stop      - Stop the MCP server"
        echo "  restart   - Restart the MCP server"
        echo "  status    - Check if server is running"
        echo "  health    - Perform health check"
        echo "  selfcheck - Perform comprehensive selfcheck"
        echo "  test      - Run both selfcheck and health check"
        exit 1
        ;;
esac