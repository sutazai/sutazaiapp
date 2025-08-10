#!/bin/bash

# Strict error handling
set -euo pipefail

# Quick navigation helper for SutazAI project


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

case "$1" in
    deploy)
        cd scripts/deployment/system
        ;;
    agents)
        cd scripts/agents
        ;;
    models)
        cd scripts/models
        ;;
    docs)
        cd docs
        ;;
    *)
        echo "Usage: ./navigate.sh [deploy|agents|models|docs]"
        echo "Quick navigation to common directories"
        ;;
esac
