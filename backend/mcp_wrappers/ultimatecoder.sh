#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../_common.sh"

PY="/opt/sutazaiapp/.mcp/UltimateCoderMCP/.venv/bin/python"
MAIN="/opt/sutazaiapp/.mcp/UltimateCoderMCP/main.py"

if [ "${1:-}" = "--selfcheck" ]; then
  section "UltimateCoder MCP selfcheck $(ts)"
  if [ -x "$PY" ]; then ok_line "venv python present"; else err_line "venv python missing at $PY"; exit 127; fi
  if [ -f "$MAIN" ]; then ok_line "main.py present"; else err_line "main.py missing at $MAIN"; exit 127; fi
  if "$PY" -c 'import fastmcp' >/dev/null 2>&1; then ok_line "fastmcp import OK"; else err_line "fastmcp not installed in venv"; exit 127; fi
  exit 0
fi

if [ -x "$PY" ] && [ -f "$MAIN" ]; then
  exec "$PY" "$MAIN"
fi

err "UltimateCoder MCP is not prepared at $MAIN with venv $PY. Please initialize it or provide an alternative."
exit 127
