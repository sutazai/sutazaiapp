#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../_common.sh"

# Compass MCP (Node)

if [ "${1:-}" = "--selfcheck" ]; then
  section "mcp-compass selfcheck $(ts)"
  if has_cmd npx; then ok_line "npx present"; else err_line "npx not found"; exit 127; fi
  exit 0
fi

# Prefer locally installed binary to avoid network fetches
LOCAL_BIN="${SCRIPT_DIR}/../../../node_modules/.bin/mcp-compass"
if [ -x "$LOCAL_BIN" ]; then
  exec "$LOCAL_BIN"
fi

# Fallback to npx resolution strategies
require_cmd npx

# Apply a short timeout to npx resolution to avoid hangs when offline
NPX_TIMEOUT_SECS="${MCP_NPX_TIMEOUT:-6}"
if command -v timeout >/dev/null 2>&1; then
  TO=(timeout "${NPX_TIMEOUT_SECS}s")
else
  TO=()
fi
CMD=()
if "${TO[@]}" npx -y @liuyoshio/mcp-compass --help >/dev/null 2>&1; then
  CMD=(npx -y @liuyoshio/mcp-compass)
elif "${TO[@]}" npx -y mcp-compass --help >/dev/null 2>&1; then
  CMD=(npx -y mcp-compass)
elif "${TO[@]}" npx -y github:liuyoshio/mcp-compass --help >/dev/null 2>&1; then
  CMD=(npx -y github:liuyoshio/mcp-compass)
else
  err "Unable to resolve mcp-compass (tried local node_modules/.bin, @liuyoshio/mcp-compass, mcp-compass, github:liuyoshio/mcp-compass)."; exit 127
fi

exec "${CMD[@]}"
