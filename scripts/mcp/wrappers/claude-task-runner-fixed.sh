#!/usr/bin/env bash
# Fixed wrapper for claude-task-runner using official MCP SDK

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_MANAGER_DIR="/opt/sutazaiapp/mcp-manager"

# Selfcheck function
selfcheck() {
    echo "Performing selfcheck for claude-task-runner (fixed)..."
    
    if [ -f "$MCP_MANAGER_DIR/fixed_claude_task_runner.py" ]; then
        if "$MCP_MANAGER_DIR/venv/bin/python" "$MCP_MANAGER_DIR/fixed_claude_task_runner.py" health >/dev/null 2>&1; then
            echo "✓ claude-task-runner (fixed) selfcheck passed"
            return 0
        fi
    fi
    
    echo "✗ claude-task-runner (fixed) selfcheck failed"
    return 1
}

# Start function
start_mcp() {
    exec "$MCP_MANAGER_DIR/venv/bin/python" "$MCP_MANAGER_DIR/fixed_claude_task_runner.py" start
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
        "$MCP_MANAGER_DIR/venv/bin/python" "$MCP_MANAGER_DIR/fixed_claude_task_runner.py" health
        ;;
    *)
        echo "Usage: $0 {start|selfcheck|health}"
        exit 1
        ;;
esac
