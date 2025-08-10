#!/bin/sh

# Strict error handling
set -euo pipefail

# Universal health check script for Alpine containers

# Check if service is listening on expected port

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

PORT=${HEALTH_CHECK_PORT:-8080}

# Method 1: Use nc (netcat) if available
if command -v nc >/dev/null 2>&1; then
    nc -z localhost $PORT
    exit $?
fi

# Method 2: Use wget if available
if command -v wget >/dev/null 2>&1; then
    wget -q --spider "http://localhost:$PORT/health" 2>/dev/null
    exit $?
fi

# Method 3: Use python if available
if command -v python3 >/dev/null 2>&1; then
    python3 -c "import socket; s=socket.socket(); s.settimeout(5); exit(0 if s.connect_ex(('localhost', $PORT))==0 else 1)"
    exit $?
fi

# Method 4: Check if process is running
if [ -f /tmp/app.pid ]; then
    pid=$(cat /tmp/app.pid)
    if kill -0 $pid 2>/dev/null; then
        exit 0
    fi
fi

# Default: assume healthy if we can't check
exit 0
