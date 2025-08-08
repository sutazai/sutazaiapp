#!/usr/bin/env bash
set -euo pipefail

# Purpose: Start the SutazAI MCP server and verify basic health.
# Usage: scripts/mcp/run_server.sh

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

MCP_DIR="${ROOT_DIR}/mcp_server"

if ! command -v node >/dev/null 2>&1; then
  echo "[mcp] Node.js not found. Please install Node 18+ or use Docker Compose profile 'mcp'."
  exit 1
fi

# Check dependencies; if not installed, advise using Docker Compose or npm install (requires network)
if [ ! -f "${MCP_DIR}/node_modules/.package-lock.json" ] && [ ! -d "${MCP_DIR}/node_modules/@modelcontextprotocol" ]; then
  echo "[mcp] Dependencies likely missing in mcp_server/. Use Docker Compose to build, or run 'npm ci' in mcp_server (requires network)."
fi

export BACKEND_API_URL=${BACKEND_API_URL:-"http://localhost:8000"}
export OLLAMA_URL=${OLLAMA_URL:-"http://localhost:11434"}
export SEQUENTIAL_THINKING_IMAGE=${SEQUENTIAL_THINKING_IMAGE:-"mcp/sequentialthinking"}
export MCP_HTTP_PORT=${MCP_HTTP_PORT:-3030}

echo "[mcp] Starting sutazai-mcp-server on port ${MCP_HTTP_PORT}..."
set -x
node "${MCP_DIR}/index.js" 2> >(tee /tmp/sutazai-mcp-server.stderr.log >&2) &
PID=$!
set +x

trap 'echo "[mcp] Stopping (PID ${PID})"; kill ${PID} 2>/dev/null || true' EXIT INT TERM

echo "[mcp] Waiting for health endpoint..."
for i in $(seq 1 20); do
  if curl -fsS "http://localhost:${MCP_HTTP_PORT}/health" >/dev/null 2>&1; then
    echo "[mcp] MCP server is healthy at http://localhost:${MCP_HTTP_PORT}/health"
    break
  fi
  sleep 0.5
done

curl -fsS "http://localhost:${MCP_HTTP_PORT}/info" || true

echo "[mcp] Tail stderr for a few seconds (CTRL+C to exit earlier)..."
timeout 5s tail -f /tmp/sutazai-mcp-server.stderr.log || true

