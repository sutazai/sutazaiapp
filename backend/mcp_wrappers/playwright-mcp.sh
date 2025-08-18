#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../_common.sh"

# Microsoft Playwright MCP Server (Node)

# Prefer local installs to avoid network-dependent npx resolution
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BIN_CANDIDATES=(
  "$ROOT_DIR/node_modules/.bin/mcp-server-playwright"
  "$ROOT_DIR/node_modules/.bin/playwright-mcp"
  "$ROOT_DIR/node_modules/.bin/mcp-playwright"
)

# Check for selfcheck BEFORE trying to execute
if [ "${1:-}" != "--selfcheck" ]; then
  for bin in "${BIN_CANDIDATES[@]}"; do
    if [ -x "$bin" ]; then
      exec "$bin" "$@"
    fi
  done
fi

if [ "${1:-}" = "--selfcheck" ]; then
  section "playwright-mcp selfcheck $(ts)"
  if has_cmd npx; then ok_line "npx present"; else err_line "npx not found"; exit 127; fi
  
  # Check if Playwright MCP is available
  if command -v mcp-server-playwright >/dev/null 2>&1; then
    ok_line "playwright-mcp installed globally"
  elif [ -x "$ROOT_DIR/node_modules/.bin/mcp-server-playwright" ]; then
    ok_line "playwright-mcp installed locally"
  elif npx -y @playwright/mcp --version >/dev/null 2>&1; then
    ok_line "playwright-mcp available via npx"
  else
    err_line "playwright-mcp not found"
    exit 127
  fi
  
  # Check if browser is installed
  if npx playwright show-report >/dev/null 2>&1 || [ -d "$HOME/.cache/ms-playwright" ]; then
    ok_line "playwright browsers installed"
  else
    warn_line "playwright browsers may not be installed"
  fi
  
  exit 0
fi

require_cmd npx
# Resolve a runnable command, then exec it (no broken fallbacks)
CMD=()
if npx -y @playwright/mcp --help >/dev/null 2>&1; then
  CMD=(npx -y @playwright/mcp)
elif npx -y playwright-mcp --help >/dev/null 2>&1; then
  CMD=(npx -y playwright-mcp)
elif npx -y @microsoft/mcp-playwright --help >/dev/null 2>&1; then
  CMD=(npx -y @microsoft/mcp-playwright)
else
  err "Unable to resolve Playwright MCP (checked local node_modules and npx for @playwright/mcp, playwright-mcp, @microsoft/mcp-playwright)."
  err "Tip: install locally with 'npm i -D @playwright/mcp' to avoid npx network fetches."
  exit 127
fi

exec "${CMD[@]}" "$@"
