#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../_common.sh"

if [ "${1:-}" = "--selfcheck" ]; then
  section "Context7 MCP selfcheck $(ts)"
  if has_cmd npx; then ok_line "npx present"; else err_line "npx not found"; exit 127; fi
  exit 0
fi

require_cmd npx
exec npx -y @upstash/context7-mcp@latest
