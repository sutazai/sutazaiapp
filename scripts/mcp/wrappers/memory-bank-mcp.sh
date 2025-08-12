#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../_common.sh"

# Memory Bank MCP (Python preferred via uv; fallback to Node GH repo)

if [ "${1:-}" = "--selfcheck" ]; then
  section "memory-bank-mcp selfcheck $(ts)"
  if has_cmd uv; then ok_line "uv present"; else warn_line "uv not found"; fi
  if has_cmd python3 && python3 -c 'import memory_bank_mcp' >/dev/null 2>&1; then ok_line "python import memory_bank_mcp OK"; else warn_line "python module missing"; fi
  if has_cmd npx; then ok_line "npx present"; else warn_line "npx not found"; fi
  if ! has_cmd uv && ! has_cmd python3 && ! has_cmd npx; then err_line "no launcher available"; exit 127; fi
  exit 0
fi

# Resolve a runnable command, then exec it
CMD=()
if has_cmd uv; then
  if uv run -q -- python -c 'import memory_bank_mcp, sys' >/dev/null 2>&1; then
    CMD=(uv run -q -m memory_bank_mcp.server)
  fi
fi

if [ ${#CMD[@]} -eq 0 ] && has_cmd python3; then
  if python3 -c 'import memory_bank_mcp, sys' >/dev/null 2>&1; then
    CMD=(python3 -m memory_bank_mcp.server)
  fi
fi

if [ ${#CMD[@]} -eq 0 ] && has_cmd npx; then
  if npx -y memory-bank-mcp --help >/dev/null 2>&1; then
    CMD=(npx -y memory-bank-mcp)
  elif npx -y github:alioshr/memory-bank-mcp --help >/dev/null 2>&1; then
    CMD=(npx -y github:alioshr/memory-bank-mcp)
  fi
fi

if [ ${#CMD[@]} -eq 0 ]; then
  err "memory-bank-mcp not available (install via 'uv pip install memory-bank-mcp' or 'npm i -g github:alioshr/memory-bank-mcp')."; exit 127
fi

exec "${CMD[@]}"
exit 127
