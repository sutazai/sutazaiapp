#!/bin/bash
if [ "$1" == "--selfcheck" ]; then
    echo "github-project-manager MCP wrapper operational"
    exit 0
fi
export GITHUB_TOKEN="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"
export GITHUB_OWNER="sutazai"
export GITHUB_REPO="sutazaiapp"
exec npx -y mcp-github-project-manager@latest
