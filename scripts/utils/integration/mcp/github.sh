#!/bin/bash

# GitHub MCP Wrapper
# Provides GitHub API access

set -e

# Handle selfcheck
if [[ "$1" == "--selfcheck" ]]; then
    echo "GitHub MCP wrapper operational"
    exit 0
fi

# Set GitHub token
export GITHUB_TOKEN="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"

# Start the MCP server using npx
exec npx -y @modelcontextprotocol/server-github "$@"
