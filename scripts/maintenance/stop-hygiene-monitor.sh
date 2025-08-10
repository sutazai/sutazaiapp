#!/bin/bash
# Stop the hygiene monitoring service

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

PID_FILE="/tmp/hygiene-monitor.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Hygiene monitor is not running (no PID file found)"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ps -p "$PID" > /dev/null 2>&1; then
    echo "Stopping hygiene monitor (PID: $PID)..."
    kill "$PID"
    sleep 2
    
    # Force kill if still running
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Force stopping..."
        kill -9 "$PID"
    fi
    
    rm -f "$PID_FILE"
    echo "✅ Hygiene monitor stopped"
else
    echo "Hygiene monitor not running (PID $PID not found)"
    rm -f "$PID_FILE"
fi