#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-/opt/sutazaiapp}"

# SSH MCP Server wrapper for npm-based ssh-mcp package
if [ "${1:-}" = "--selfcheck" ]; then
    echo "SSH MCP Server selfcheck"
    if [ -f "$PROJECT_DIR/node_modules/.bin/ssh-mcp" ]; then
        echo "✓ ssh-mcp package installed"
        exit 0
    else
        echo "✗ ssh-mcp not found"
        exit 1
    fi
fi

# Run the ssh-mcp server
# Note: This server requires SSH host configuration to work
if [ -f "$PROJECT_DIR/node_modules/.bin/ssh-mcp" ]; then
    # Disabled - requires actual SSH credentials
    # To enable, uncomment and configure:
    # exec "$PROJECT_DIR/node_modules/.bin/ssh-mcp" --host your-host --user your-user
    echo "SSH MCP Server disabled - requires SSH host configuration" >&2
    exit 0
else
    echo "Error: ssh-mcp not installed. Run: npm install ssh-mcp" >&2
    exit 1
fi