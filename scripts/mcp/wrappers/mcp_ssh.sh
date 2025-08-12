#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../_common.sh"

PROJECT_DIR="${PROJECT_DIR:-/opt/sutazaiapp}"
MCP_SSH_DIR="$PROJECT_DIR/mcp_ssh"

if [ "${1:-}" = "--selfcheck" ]; then
  section "mcp_ssh selfcheck $(ts)"
  if has_cmd uv; then ok_line "uv present"; else warn_line "uv not found"; fi
  if has_cmd python3 && PYTHONPATH="$MCP_SSH_DIR/src:${PYTHONPATH:-}" python3 -c 'import mcp_ssh' >/dev/null 2>&1; then ok_line "python import mcp_ssh OK"; else warn_line "python import mcp_ssh failed"; fi
  exit 0
fi

if has_cmd uv; then
  exec uv --directory "$MCP_SSH_DIR" run mcp_ssh
fi

# Try python directly if uv missing but dependencies already installed
if has_cmd python3; then
  export PYTHONPATH="$MCP_SSH_DIR/src:${PYTHONPATH:-}"
  exec python3 -m mcp_ssh.server
fi

err "mcp_ssh requires 'uv' or python with dependencies installed (see $MCP_SSH_DIR)."
exit 127
