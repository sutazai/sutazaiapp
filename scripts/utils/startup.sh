#!/bin/bash

# Strict error handling
set -euo pipefail


# Agent startup script

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

echo "Starting agent..."

# Ensure Ollama is available
if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama not found. Please install Ollama first."
    exit 1
fi

# Start the agent
python app.py
