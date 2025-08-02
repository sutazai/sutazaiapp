#!/bin/bash
#
# start_webui_direct.sh - Direct script to start the Web UI frontend on port 3000
#

set -e

# Configuration
WEBUI_DIR="/opt/sutazaiapp/web_ui"
LOG_DIR="/opt/sutazaiapp/logs"
WEBUI_LOG_DIR="$LOG_DIR/web_ui"

# Create log directory if it doesn't exist
mkdir -p "$WEBUI_LOG_DIR"

echo "Starting Web UI on port 3000..."

# Check if we're in the web_ui directory
if [ ! -d "$WEBUI_DIR" ]; then
    echo "Error: Web UI directory $WEBUI_DIR not found"
    exit 1
fi

# Change to the web_ui directory
cd "$WEBUI_DIR"

# Set up environment variables
export PORT=3000
export NEXT_PUBLIC_API_URL="http://localhost:8000"

# Start Next.js in production mode with explicit port 3000
echo "Starting in production mode on port 3000..."
npx next start -p 3000 2>&1 | tee -a "$WEBUI_LOG_DIR/webui.log" &
WEBUI_PID=$!

# Store PID
echo $WEBUI_PID > "$LOG_DIR/webui.pid"
echo $WEBUI_PID > "/opt/sutazaiapp/.webui.pid"
echo "Web UI started with PID: $WEBUI_PID" 