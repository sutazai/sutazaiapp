#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[MCP] Tearing down MCP stack"
docker compose -f docker-compose.mcp.yml down
echo "[MCP] Done"

