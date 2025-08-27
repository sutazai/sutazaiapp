#!/bin/bash
if [ "$1" == "--selfcheck" ]; then
    echo "gitmcp-anthropic MCP wrapper operational"
    exit 0
fi
# Using git-mcp-server configured for Anthropic repositories
# Note: GitMCP.io (https://gitmcp.io/anthropic/*) is available for HTTP-based clients
export GITHUB_TOKEN="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"
exec npx -y @cyanheads/git-mcp-server@latest
