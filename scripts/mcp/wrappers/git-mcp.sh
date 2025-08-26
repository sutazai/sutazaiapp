#!/usr/bin/env bash
# MCP Wrapper for git-mcp
# Created: 2025-08-26 UTC
# Purpose: Wrapper for GitMCP.io integration

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/opt/sutazaiapp"

# Configuration
MCP_NAME="git-mcp"
MCP_DESCRIPTION="GitMCP.io integration for repository access"
REPO_URL="https://gitmcp.io/sutazai/sutazaiapp"

# Selfcheck function
selfcheck() {
    echo "Performing selfcheck for $MCP_NAME..."
    
    # Check if npx is available
    if ! command -v npx >/dev/null 2>&1; then
        echo "ERROR: npx not available (required for $MCP_NAME)"
        return 1
    fi
    
    # Check if mcp-remote package is available
    if ! npx mcp-remote --version >/dev/null 2>&1; then
        echo "WARNING: mcp-remote not installed, will be downloaded on first run"
    fi
    
    # Test connection to GitMCP.io
    if curl -s -o /dev/null -w "%{http_code}" "$REPO_URL" | grep -q "200"; then
        echo "✓ GitMCP.io endpoint is reachable"
    else
        echo "WARNING: GitMCP.io endpoint might be unreachable"
    fi
    
    echo "✓ $MCP_NAME selfcheck passed"
    return 0
}

# Start function
start_mcp() {
    echo "Starting $MCP_NAME for repository: $REPO_URL"
    
    # Execute mcp-remote with the repository URL
    exec npx -y mcp-remote "$REPO_URL"
}

# Main command handling
case "${1:-start}" in
    start)
        start_mcp
        ;;
    selfcheck|--selfcheck)
        selfcheck
        ;;
    *)
        echo "Usage: $0 {start|selfcheck}"
        echo "  start     - Start the MCP server"
        echo "  selfcheck - Perform comprehensive selfcheck"
        exit 1
        ;;
esac
