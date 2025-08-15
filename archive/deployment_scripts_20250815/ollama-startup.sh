#!/bin/bash

# Strict error handling
set -euo pipefail

# Ollama startup script with model preloading


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

echo "Starting Ollama with performance optimizations..."

# Wait for Ollama to be ready
until ollama list > /dev/null 2>&1; do
    echo "Waiting for Ollama to start..."
    sleep 5
done

echo "Ollama is ready. Loading models..."

# Pull and load TinyLlama (primary model)
ollama pull tinyllama:latest || true
ollama run tinyllama:latest "test" --verbose || true

echo "Model loaded. Warming up..."

# Warm up the model with a few test queries
for i in {1..3}; do
    echo "Warm-up query $i..."
    timeout 30 ollama run tinyllama:latest "Hello, how are you?" || true
    sleep 2
done

echo "Ollama startup complete and warmed up!"
