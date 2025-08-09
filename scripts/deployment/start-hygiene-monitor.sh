#!/bin/bash
# Start the hygiene monitoring service

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MONITOR_SCRIPT="$SCRIPT_DIR/hygiene-monitor.py"
PID_FILE="/tmp/hygiene-monitor.pid"

# Check if monitor is already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Hygiene monitor is already running (PID: $PID)"
        exit 0
    else
        echo "Removing stale PID file"
        rm -f "$PID_FILE"
    fi
fi

# Install required Python package if not present
if ! python3 -c "import watchdog" 2>/dev/null; then
    echo "Installing watchdog package..."
    pip3 install watchdog
fi

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Start monitor in background
echo "Starting hygiene monitor..."
export SUTAZAI_ROOT="$PROJECT_ROOT"
export HYGIENE_AUTO_FIX="${HYGIENE_AUTO_FIX:-false}"

nohup python3 "$MONITOR_SCRIPT" > "$PROJECT_ROOT/logs/hygiene-monitor.out" 2>&1 &
PID=$!
echo $PID > "$PID_FILE"

echo "✅ Hygiene monitor started (PID: $PID)"
echo "   Auto-fix: $HYGIENE_AUTO_FIX"
echo "   Logs: $PROJECT_ROOT/logs/hygiene-monitor.log"
echo ""
echo "To stop: $SCRIPT_DIR/stop-hygiene-monitor.sh"
echo "To enable auto-fix: export HYGIENE_AUTO_FIX=true"