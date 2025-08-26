#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-/opt/sutazaiapp}"

# SSH MCP Server wrapper
if [ "${1:-}" = "--selfcheck" ]; then
    echo "SSH MCP Server selfcheck"
    if command -v npx >/dev/null 2>&1; then
        echo "✓ npx available for ssh-mcp"
        exit 0
    else
        echo "✗ npx not found"
        exit 1
    fi
fi

# Run the ssh-mcp server using npx
# This will auto-install if needed
exec npx -y @aiondadotcom/mcp-ssh@latest