#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Unified Memory MCP Wrapper
# Consolidates extended-memory and memory-bank-mcp functionality

PYTHON_SERVICE="/opt/memory/unified-memory-service.py"

if [ "${1:-}" = "--selfcheck" ]; then
  echo "unified-memory MCP selfcheck $(date)"
  if [ -f "$PYTHON_SERVICE" ]; then
    echo "✅ unified-memory service found"
  else
    echo "❌ unified-memory service missing"
    exit 127
  fi
  
  if python3 -c 'import fastapi, uvicorn, sqlite3' >/dev/null 2>&1; then
    echo "✅ required dependencies available"
  else
    echo "❌ missing dependencies"
    exit 127
  fi
  exit 0
fi

if [ "${1:-}" = "health" ]; then
  curl -sf http://localhost:3009/health >/dev/null 2>&1
  exit $?
fi

if [ "${1:-}" = "export-all" ]; then
  curl -sf http://localhost:3009/memory/stats
  exit $?
fi

# Start the unified memory service
exec python3 "$PYTHON_SERVICE"