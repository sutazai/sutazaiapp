#!/bin/bash
#
# start_webui.sh - Script to start the Web UI frontend
#

set -e

# Source environment variables
if [ -f /opt/sutazaiapp/.env ]; then
    source <(grep -v '^#' /opt/sutazaiapp/.env | sed -E 's/(.*)=(.*)/export \1="\2"/')
fi

# Configuration
WEBUI_DIR="/opt/sutazaiapp/web_ui"  # Primary location
ALT_WEBUI_DIR="/opt/sutazaiapp/web" # Alternative location
LOG_DIR="${LOGS_DIR:-/opt/sutazaiapp/logs}"
WEBUI_LOG_DIR="$LOG_DIR/web_ui"
NODE_ENV="${NODE_ENV:-production}"
PORT="3000"  # Explicitly set to 3000
HOST="${WEBUI_HOST:-0.0.0.0}"  # Use 0.0.0.0 to bind to all interfaces
API_URL="${API_URL:-http://localhost:8000}"

# Create log directory if it doesn't exist
mkdir -p "$WEBUI_LOG_DIR"

echo "Starting Web UI on $HOST:$PORT..."

# Check if we're in the web_ui directory or alternative location
if [ -d "$WEBUI_DIR" ]; then
    FINAL_WEBUI_DIR="$WEBUI_DIR"
    echo "Using primary Web UI directory: $FINAL_WEBUI_DIR"
elif [ -d "$ALT_WEBUI_DIR" ]; then
    FINAL_WEBUI_DIR="$ALT_WEBUI_DIR"
    echo "Using alternative Web UI directory: $FINAL_WEBUI_DIR"
else
    echo "Error: Web UI directories not found at $WEBUI_DIR or $ALT_WEBUI_DIR"
    echo "Checking for other possible locations..."
    
    POSSIBLE_DIRS=$(find /opt/sutazaiapp -maxdepth 2 -type d -name "web*" | grep -v "node_modules")
    
    if [ -n "$POSSIBLE_DIRS" ]; then
        echo "Found potential Web UI directories:"
        for dir in $POSSIBLE_DIRS; do
            if [ -f "$dir/package.json" ]; then
                FINAL_WEBUI_DIR="$dir"
                echo "Selected Web UI directory: $FINAL_WEBUI_DIR"
                break
            fi
        done
    fi
    
    if [ -z "$FINAL_WEBUI_DIR" ]; then
        echo "Error: No suitable Web UI directory found"
        exit 1
    fi
fi

# Change to the web_ui directory
cd "$FINAL_WEBUI_DIR"

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Create logs directory if it doesn't exist
LOGS_DIR="logs"
if [ ! -d "$LOGS_DIR" ]; then
    mkdir -p "$LOGS_DIR"
    echo "Created logs directory."
fi
chmod -R 777 "$LOGS_DIR" 2>/dev/null || echo "Warning: Could not set permissions on logs directory"

# Check for .env.local file, create from example if needed
if [ ! -f "app/.env.local" ]; then
    if [ -f "app/.env.local.example" ]; then
        cp app/.env.local.example app/.env.local
        echo "Created app/.env.local from example file."
    else
        echo "Warning: app/.env.local.example not found. Creating default env file."
        echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > app/.env.local
    fi
fi

# --- Improved Process Killing for Streamlit ---
STREAMLIT_PORT=${PORT:-8501} # Use default 8501 if PORT not set
STREAMLIT_APP_PATH="streamlit_app.py" # Adjust if your main app file is different

echo "Attempting to stop any existing Streamlit processes on port $STREAMLIT_PORT or running $STREAMLIT_APP_PATH..."

# Method 1: Kill processes listening on the port
# Use pkill to be more robust than kill $(lsof...)
pkill -f ":$STREAMLIT_PORT"

# Method 2: Kill processes matching the streamlit run command
# This catches processes even if they haven't bound to the port yet
pkill -f "streamlit run $STREAMLIT_APP_PATH"

sleep 2 # Give processes time to terminate

# Method 3: Force kill any remaining stubborn processes (use with caution)
# pkill -9 -f "streamlit run $STREAMLIT_APP_PATH"

echo "Finished attempting to stop existing Streamlit processes."
# --- End Improved Process Killing ---


# --- Set Streamlit Config Environment Variables ---
export STREAMLIT_SERVER_ENABLE_CORS=true
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false # Also disable usage stats
# --- End Config Env Vars ---

# Navigate to the app directory # Assuming streamlit app is in project root or a specific dir?
# Let's assume streamlit_app.py is in the PROJECT_ROOT for now
# cd app || { echo "Error: app directory not found"; exit 1; }
# Check if streamlit_app.py exists
if [ ! -f "$PROJECT_ROOT/$STREAMLIT_APP_PATH" ]; then
    echo "Error: Streamlit app '$STREAMLIT_APP_PATH' not found in project root '$PROJECT_ROOT'."
    # Attempt to find it in common locations
    if [ -f "$PROJECT_ROOT/app/$STREAMLIT_APP_PATH" ]; then
       STREAMLIT_APP_PATH="app/$STREAMLIT_APP_PATH"
       echo "Found Streamlit app in: $PROJECT_ROOT/$STREAMLIT_APP_PATH"
    elif [ -f "$PROJECT_ROOT/ui/$STREAMLIT_APP_PATH" ]; then
        STREAMLIT_APP_PATH="ui/$STREAMLIT_APP_PATH"
        echo "Found Streamlit app in: $PROJECT_ROOT/$STREAMLIT_APP_PATH"
    else
       echo "Could not find Streamlit app. Please ensure STREAMLIT_APP_PATH is set correctly."
       exit 1
    fi
fi

# Start Streamlit directly (assuming streamlit_app.py is the entry point)
# Use specified port and run headless
echo "Starting Streamlit UI..."
# Ensure we are in the project root before running streamlit
cd "$PROJECT_ROOT"
streamlit run "$STREAMLIT_APP_PATH" --server.port "$STREAMLIT_PORT" --server.headless true > "$LOGS_DIR/webui.log" 2>&1 &
# Note: Removed the old npx next start command
WEBUI_PID=$!

# Store PID for future reference
# cd .. # No need to cd back if we start from project root
# Use the specific pids directory
echo $WEBUI_PID > "$PROJECT_ROOT/pids/webui.pid"

echo "Web UI (Streamlit) started with PID: $WEBUI_PID"
echo "Logs are being written to: $LOGS_DIR/webui.log"

# Verify Web UI has started correctly
sleep 5 # Increased sleep time slightly

# Check based on port listening as PID might be the launcher
if ! ss -tulnp | grep -q ":$STREAMLIT_PORT"; then
    echo "Error: Streamlit UI process failed to start or bind to port $STREAMLIT_PORT."
    echo "Check the logs at $LOGS_DIR/webui.log for details."
    if [ -f "$LOGS_DIR/webui.log" ]; then
        echo "Last 10 lines of log:"
        tail -n 10 "$LOGS_DIR/webui.log"
    fi
    exit 1
fi

echo "Web UI successfully started!"
echo "Access the Web UI at: http://localhost:$STREAMLIT_PORT"
echo "To stop the Web UI, run: scripts/stop_webui.sh" # Assuming this script exists 