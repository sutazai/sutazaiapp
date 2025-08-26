#!/usr/bin/env bash
set -Eeuo pipefail

selfcheck() {
    if command -v npx >/dev/null 2>&1; then
        echo "✓ nx-mcp selfcheck passed"
        return 0
    fi
    echo "✗ nx-mcp selfcheck failed"
    return 1
}

case "${1:-start}" in
    start)
        exec npx @nx-console/nx-mcp-server@latest
        ;;
    selfcheck|--selfcheck)
        selfcheck
        ;;
    health)
        echo '{"status": "healthy", "server": "nx-mcp"}'
        ;;
    *)
        echo "Usage: $0 {start|selfcheck|health}"
        exit 1
        ;;
esac
