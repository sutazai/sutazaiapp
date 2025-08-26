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
# Using the provided GitHub token
export GITHUB_TOKEN="${GITHUB_TOKEN:-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}"
# Default owner and repo - users can override these
export GITHUB_OWNER="${GITHUB_OWNER:-octocat}"
export GITHUB_REPO="${GITHUB_REPO:-Hello-World}"

# Change to project directory and run the built server
cd /opt/sutazaiapp/.mcp-servers/mcp-github-project-manager
exec node build/index.js "$@"