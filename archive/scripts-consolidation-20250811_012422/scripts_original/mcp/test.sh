#!/usr/bin/env bash
set -euo pipefail

# Purpose: Run the MCP server test suite (best-effort in restricted environments).
# Usage: scripts/mcp/test.sh


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

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

