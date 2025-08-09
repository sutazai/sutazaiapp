#!/bin/bash
# Stop the hygiene monitoring service

set -euo pipefail

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