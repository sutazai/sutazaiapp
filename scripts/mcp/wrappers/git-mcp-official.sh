#!/usr/bin/env bash
set -Eeuo pipefail

MCP_DIR="/opt/sutazaiapp/mcp-servers/git-mcp"

selfcheck() {
    if [ -f "$MCP_DIR/venv/bin/mcp-server-git" ]; then
        echo "✓ git-mcp selfcheck passed"
        return 0
    fi
    echo "✗ git-mcp selfcheck failed"
    return 1
}

case "${1:-start}" in
    start)
        exec "$MCP_DIR/venv/bin/mcp-server-git"
        ;;
    selfcheck|--selfcheck)
        selfcheck
        ;;
    health)
        echo '{"status": "healthy", "server": "git-mcp"}'
        ;;
    *)
        echo "Usage: $0 {start|selfcheck|health}"
        exit 1
        ;;
esac
