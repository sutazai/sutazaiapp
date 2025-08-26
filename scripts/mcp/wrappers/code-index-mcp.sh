#!/bin/bash

# Code Index MCP Wrapper
# Server for code indexing and analysis

if [ "$1" = "--selfcheck" ]; then
    cd /opt/sutazaiapp/.mcp-servers/code-index-mcp
    if [ -f venv/bin/code-index-mcp ]; then
        echo "✓ code-index-mcp"
        exit 0
    else
        echo "✗ code-index-mcp (not installed)"
        exit 1
    fi
fi

# Change to project directory and activate virtual environment
cd /opt/sutazaiapp/.mcp-servers/code-index-mcp
source venv/bin/activate

# Run the server
exec code-index-mcp "$@"