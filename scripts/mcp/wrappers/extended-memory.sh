#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../_common.sh"

VENV_PY="/opt/sutazaiapp/.venvs/extended-memory/bin/python"

if [ "${1:-}" = "--selfcheck" ]; then
  section "extended-memory MCP selfcheck $(ts)"
  if [ -x "$VENV_PY" ]; then
    if "$VENV_PY" -c 'import extended_memory_mcp' >/dev/null 2>&1; then ok_line "venv present + module import OK"; else err_line "venv present but module missing"; exit 127; fi
  else
    warn_line "venv not present at $VENV_PY"
    if has_cmd uv; then ok_line "uv present"; else warn_line "uv not found"; fi
    if has_cmd python3 && python3 -c 'import extended_memory_mcp' >/dev/null 2>&1; then ok_line "system python has module"; else warn_line "system python missing module"; fi
  fi
  exit 0
fi

if [ -x "$VENV_PY" ]; then
  exec "$VENV_PY" -m extended_memory_mcp.server
fi

if has_cmd uv; then
  exec uv run -q -m extended_memory_mcp.server
fi

if has_cmd python3; then
  # Attempt to run if available in current env
  exec python3 -m extended_memory_mcp.server
fi

err "extended-memory MCP requires a prepared venv at $VENV_PY or uv/python with the package installed."
exit 127
