#!/usr/bin/env bash
# Claude Task Runner MCP Server - FastMCP v2 Compatible

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_MANAGER_DIR="/opt/sutazaiapp/mcp-manager"
VENV_DIR="$MCP_MANAGER_DIR/venv"

# Ensure virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..." >&2
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
fi

# Install/upgrade dependencies if needed
check_dependencies() {
    if ! "$VENV_DIR/bin/python" -c "import fastmcp" 2>/dev/null; then
        echo "Installing FastMCP..." >&2
        "$VENV_DIR/bin/pip" install --quiet "fastmcp>=2.3.3"
    fi
}

# Selfcheck function
selfcheck() {
    check_dependencies
    if "$VENV_DIR/bin/python" "$MCP_MANAGER_DIR/claude_task_runner_server.py" health >/dev/null 2>&1; then
        echo "✓ claude-task-runner v2 selfcheck passed"
        return 0
    fi
    echo "✗ claude-task-runner v2 selfcheck failed"
    return 1
}

# Start function
start_mcp() {
    check_dependencies
    exec "$VENV_DIR/bin/python" "$MCP_MANAGER_DIR/claude_task_runner_server.py" start
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
        check_dependencies
        "$VENV_DIR/bin/python" "$MCP_MANAGER_DIR/claude_task_runner_server.py" health
        ;;
    *)
        echo "Usage: $0 {start|selfcheck|health}"
        exit 1
        ;;
esac