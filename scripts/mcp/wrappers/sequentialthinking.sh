#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../_common.sh"

if [ "${1:-}" = "--selfcheck" ]; then
  section "SequentialThinking MCP selfcheck $(ts)"
  if has_cmd docker; then ok_line "docker present"; else warn_line "docker not found"; fi
  if has_cmd npx; then ok_line "npx present"; else warn_line "npx not found"; fi
  if ! has_cmd docker && ! has_cmd npx; then err_line "no launcher available"; exit 127; fi
  exit 0
fi

if has_cmd docker; then
  exec docker run --rm -i mcp/sequentialthinking
fi

require_cmd npx
exec npx -y mcp/sequentialthinking
