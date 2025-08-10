#!/usr/bin/env bash
set -euo pipefail

# Purpose: Build the official MCP SequentialThinking Docker image locally.
# Usage: scripts/mcp/build_sequentialthinking.sh


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

echo "[mcp] Building mcp/sequentialthinking image..."

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

REPO_DIR="${ROOT_DIR}/servers"

if [ ! -d "$REPO_DIR/src/sequentialthinking" ]; then
  echo "[mcp] 'servers' repo missing or incomplete at $REPO_DIR"
  echo "[mcp] Attempt to clone requires network access. If restricted, clone manually:"
  echo "       git clone https://github.com/modelcontextprotocol/servers.git servers"
  exit 1
fi

set -x
# Preferred: build using the sequentialthinking subdir as context
docker build -t mcp/sequentialthinking \
  -f servers/src/sequentialthinking/Dockerfile \
  servers/src/sequentialthinking || {
  echo "[mcp] Fallback: build with repo root as context"
  docker build -t mcp/sequentialthinking \
    -f servers/src/sequentialthinking/Dockerfile \
    servers
}
set +x

echo "[mcp] Built image: mcp/sequentialthinking"

