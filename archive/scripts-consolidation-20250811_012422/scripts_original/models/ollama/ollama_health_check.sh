#!/bin/bash
# Purpose: Ollama Health Check - Monitor API and container health
# Usage: ./ollama_health_check.sh
# Requires: curl, jq, docker access

set -euo pipefail


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

check_ollama_health() {
    # Check if Ollama is responding
    if curl -f -s http://localhost:10104/api/tags > /dev/null 2>&1; then
        echo "Ollama is healthy"
        
        # Check loaded models
        MODELS=$(curl -s http://localhost:10104/api/tags | jq -r '.models[].name' 2>/dev/null | wc -l)
        echo "Loaded models: $MODELS"
        
        # Check response time
        RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' http://localhost:10104/api/tags)
        echo "API response time: ${RESPONSE_TIME}s"
        
        # Memory usage
        MEMORY=$(docker stats sutazai-ollama --no-stream --format "{{.MemUsage}}" 2>/dev/null)
        echo "Memory usage: $MEMORY"
        
        return 0
    else
        echo "Ollama is not responding"
        return 1
    fi
}

# Run health check
check_ollama_health
