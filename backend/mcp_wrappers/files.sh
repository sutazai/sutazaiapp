#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../_common.sh"

PROJECT_DIR="${PROJECT_DIR:-/opt/sutazaiapp}"

if [ "${1:-}" = "--selfcheck" ]; then
  section "Filesystem MCP selfcheck $(ts)"
  if has_cmd npx; then ok_line "npx present"; else err_line "npx not found"; exit 127; fi
  exit 0
fi

if has_cmd npx; then
  exec npx -y @modelcontextprotocol/server-filesystem "$PROJECT_DIR"
fi

err "server-filesystem requires Node+npx. Please install Node or run within an environment with npx available."
exit 127
