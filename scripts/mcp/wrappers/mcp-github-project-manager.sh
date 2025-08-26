#!/bin/bash

# MCP GitHub Project Manager Wrapper
# Server for managing GitHub Projects

if [ "$1" = "--selfcheck" ]; then
    cd /opt/sutazaiapp/.mcp-servers/mcp-github-project-manager
    if [ -f build/index.js ]; then
        echo "✓ mcp-github-project-manager"
        exit 0
    else
        echo "✗ mcp-github-project-manager (build not found)"
        exit 1
    fi
fi

# Set required environment variables
export GITHUB_TOKEN="${GITHUB_TOKEN:-}"

# Change to project directory and run the built server
cd /opt/sutazaiapp/.mcp-servers/mcp-github-project-manager
exec node build/index.js "$@"