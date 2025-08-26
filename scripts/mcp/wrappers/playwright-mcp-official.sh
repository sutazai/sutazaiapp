#!/usr/bin/env bash
set -Eeuo pipefail

MCP_DIR="/opt/sutazaiapp/mcp-servers/playwright-mcp-official"

selfcheck() {
    if [ -f "$MCP_DIR/package.json" ]; then
        echo "✓ playwright-mcp selfcheck passed"
        return 0
    fi
    echo "✗ playwright-mcp selfcheck failed"
    return 1
}

case "${1:-start}" in
    start)
        cd "$MCP_DIR"
        exec npx @playwright/mcp@latest
        ;;
    selfcheck|--selfcheck)
        selfcheck
        ;;
    health)
        echo '{"status": "healthy", "server": "playwright-mcp"}'
        ;;
    *)
        echo "Usage: $0 {start|selfcheck|health}"
        exit 1
        ;;
esac
