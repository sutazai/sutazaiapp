#!/bin/bash

# Task Runner MCP Server Wrapper (simplified)
# Uses the claude-task-runner implementation with fastmcp

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_SERVER_DIR="/opt/sutazaiapp/mcp-servers/claude-task-runner"

# Self-check for health monitoring
if [ "$1" = "--selfcheck" ]; then
    if [ -d "$MCP_SERVER_DIR/venv" ] && [ -f "$MCP_SERVER_DIR/src/task_runner/mcp/mcp_server.py" ]; then
        echo '{"healthy":true,"service":"task-runner"}'
    else
        echo '{"healthy":false,"error":"task-runner not properly installed"}'
        exit 1
    fi
    exit 0
fi

# Ensure virtual environment exists with dependencies
if [ ! -d "$MCP_SERVER_DIR/venv" ]; then
    python3 -m venv "$MCP_SERVER_DIR/venv"
    "$MCP_SERVER_DIR/venv/bin/pip" install --quiet --upgrade pip
    "$MCP_SERVER_DIR/venv/bin/pip" install --quiet "fastmcp>=2.3.3" mcp typer rich loguru litellm json-repair python-dotenv
fi

# Change to server directory and run
cd "$MCP_SERVER_DIR"
export PYTHONPATH="$MCP_SERVER_DIR/src:$PYTHONPATH"

# Run the task runner MCP server
exec "$MCP_SERVER_DIR/venv/bin/python" -m task_runner.mcp.mcp_server