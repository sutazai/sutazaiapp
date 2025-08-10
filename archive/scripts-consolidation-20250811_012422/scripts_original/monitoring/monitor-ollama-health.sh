#!/bin/bash

# Strict error handling
set -euo pipefail

# Ollama Health Monitoring Script


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

# Timeout mechanism to prevent infinite loops
LOOP_TIMEOUT=${LOOP_TIMEOUT:-300}  # 5 minute default timeout
loop_start=$(date +%s)
while true; do
    echo "=== Ollama Health Check $(date) ==="
    
    # Check service status
    if curl -f -s http://localhost:10104/api/tags >/dev/null; then
        echo "✅ Service: Healthy"
    else
        echo "❌ Service: Unhealthy"
    fi
    
    # Check resource usage
    echo "📊 Resource Usage:"
    docker stats sutazai-ollama --no-stream --format "  CPU: {{.CPUPerc}}  Memory: {{.MemUsage}}"
    
    # Check loaded models
    echo "🧠 Loaded Models:"
    docker exec sutazai-ollama ollama list 2>/dev/null | tail -n +2 | while read line; do
        echo "  $line"
    # Check for timeout
    current_time=$(date +%s)
    if [[ $((current_time - loop_start)) -gt $LOOP_TIMEOUT ]]; then
        echo 'Loop timeout reached after ${LOOP_TIMEOUT}s, exiting...' >&2
        break
    fi

    done
    
    echo ""
    sleep 30
done