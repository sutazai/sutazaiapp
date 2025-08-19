#!/usr/bin/env bash
set -Eeuo pipefail
# Back-compat wrapper: puppeteer-mcp now aliases to Playwright MCP
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "--selfcheck" ]; then
  echo "puppeteer-mcp is deprecated; delegating to playwright-mcp" >&2
fi

exec "${SCRIPT_DIR}/playwright-mcp.sh" "$@"

