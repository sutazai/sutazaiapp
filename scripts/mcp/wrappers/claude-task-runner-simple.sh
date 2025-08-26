#!/usr/bin/env bash
# Simple working wrapper for claude-task-runner

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_MANAGER_DIR="/opt/sutazaiapp/mcp-manager"

# Selfcheck function
selfcheck() {
    echo "Performing selfcheck for claude-task-runner (simple)..."
    if /opt/sutazaiapp/mcp-manager/venv/bin/python "$MCP_MANAGER_DIR/simple_mcp_wrapper.py" health >/dev/null 2>&1; then
        echo "✓ claude-task-runner (simple) selfcheck passed"
        return 0
    fi
    echo "✗ claude-task-runner (simple) selfcheck failed"
    return 1
}

# Start function
start_mcp() {
    exec /opt/sutazaiapp/mcp-manager/venv/bin/python "$MCP_MANAGER_DIR/simple_mcp_wrapper.py" start
}

# Main command handling
case "${1:-start}" in
    start)
        start_mcp
        ;;
    selfcheck|--selfcheck)
        selfcheck
        ;;
    health)
        /opt/sutazaiapp/mcp-manager/venv/bin/python "$MCP_MANAGER_DIR/simple_mcp_wrapper.py" health
        ;;
    *)
        echo "Usage: $0 {start|selfcheck|health}"
        exit 1
        ;;
esac