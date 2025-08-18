#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../_common.sh"

if [ "${1:-}" = "--selfcheck" ]; then
  section "GitHub MCP selfcheck $(ts)"
  if has_cmd npx; then ok_line "npx present"; else warn_line "npx not found"; fi
  exit 0
fi

# Use the NPX version as defined in .mcp.json
exec npx -y @modelcontextprotocol/server-github --repositories "sutazai/sutazaiapp"

