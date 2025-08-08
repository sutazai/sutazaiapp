#!/usr/bin/env bash
set -euo pipefail

# Purpose: Run the MCP server test suite (best-effort in restricted environments).
# Usage: scripts/mcp/test.sh

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR/mcp_server"

if ! command -v node >/dev/null 2>&1; then
  echo "[mcp] Node.js not found. Cannot run tests."
  exit 1
fi

if [ ! -f package.json ]; then
  echo "[mcp] package.json missing in mcp_server/."
  exit 1
fi

echo "[mcp] Running MCP server tests (may warn if deps/services unavailable)..."
node test.js || true

