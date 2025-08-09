#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[MCP] Bootstrapping unified MCP stack (SutazAI + SequentialThinking + Context7)"

# 1) Ensure Sequential Thinking image exists
if ! docker image inspect mcp/sequentialthinking >/dev/null 2>&1; then
  echo "[MCP] Building mcp/sequentialthinking image from modelcontextprotocol/servers"
  if [ ! -d servers ]; then
    git clone https://github.com/modelcontextprotocol/servers.git
  fi
  docker build -t mcp/sequentialthinking \
    -f servers/src/sequentialthinking/Dockerfile \
    servers/src/sequentialthinking
else
  echo "[MCP] Found existing image mcp/sequentialthinking"
fi

# 2) Prepare MCP env file
if [ ! -f mcp_server/.env ]; then
  echo "[MCP] Preparing mcp_server/.env from example"
  cp mcp_server/config.example.env mcp_server/.env
  echo "[MCP] Please review and set Context7 env if needed in mcp_server/.env"
fi

# 3) Bring up stack
echo "[MCP] Starting MCP stack via docker-compose.mcp.yml"
docker compose -f docker-compose.mcp.yml up -d --build

# 4) Health check
MCP_PORT=$(grep -E '^MCP_HTTP_PORT=' mcp_server/.env | cut -d'=' -f2 || true)
MCP_PORT=${MCP_PORT:-3030}
HEALTH_URL="http://localhost:${MCP_PORT}/health"

echo "[MCP] Waiting for MCP health at ${HEALTH_URL}"
for i in {1..30}; do
  if curl -sf "${HEALTH_URL}" >/dev/null; then
    echo "[MCP] MCP health is OK"
    break
  fi
  sleep 1
  if [ "$i" -eq 30 ]; then
    echo "[MCP] MCP health did not respond in time" >&2
    exit 1
  fi
done

# 5) Sequential Thinking smoke test
echo "[MCP] Running Sequential Thinking smoke test"
docker run --rm -i mcp/sequentialthinking \
  --input '{"thought":"Bootstrap test","nextThoughtNeeded":false,"thoughtNumber":1,"totalThoughts":1}' || true

echo "[MCP] Bootstrap complete. Inspector (optional): http://localhost:6274"

