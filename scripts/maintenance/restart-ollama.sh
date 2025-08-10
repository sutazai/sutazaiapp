#!/bin/bash

# Strict error handling
set -euo pipefail

# Ollama Restart Script


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

echo "Restarting Ollama service..."
docker-compose -f /opt/sutazaiapp/docker-compose.yml restart ollama

echo "Waiting for service to be ready..."
sleep 10

if curl -f -s http://localhost:10104/api/tags >/dev/null; then
    echo "✅ Ollama restarted successfully"
else
    echo "❌ Ollama restart failed"
    exit 1
fi