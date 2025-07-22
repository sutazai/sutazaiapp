#!/bin/bash
#
# start_webui.sh - Script to start the Streamlit Web UI frontend
#

set -e

# Source environment variables
if [ -f /opt/sutazaiapp/.env ]; then
    source <(grep -v '^#' /opt/sutazaiapp/.env | sed -E 's/(.*)=(.*)/export \\1="\\2"/')
fi

# Configuration
LOG_DIR="${LOGS_DIR:-/opt/sutazaiapp/logs}"
WEBUI_LOG_DIR="$LOG_DIR/web_ui" # Specific log dir for streamlit
NODE_ENV="${NODE_ENV:-production}" # Keep NODE_ENV, might be used elsewhere
PORT="${STREAMLIT_PORT:-8501}"  # Use Streamlit specific port variable, default 8501
HOST="${STREAMLIT_HOST:-0.0.0.0}"  # Use Streamlit specific host variable
API_URL="${API_URL:-http://localhost:8000}" # Keep API_URL for Streamlit app to connect

# Create log directory if it doesn't exist
mkdir -p "$WEBUI_LOG_DIR" # Use specific log dir

echo "Starting Streamlit Web UI on $HOST:$PORT..."

# Navigate to the project root directory
# Assuming this script is in /opt/sutazaiapp/bin or similar
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

# Create logs directory within project root if it doesn't exist (redundant but safe)
mkdir -p "$PROJECT_ROOT/logs"
chmod -R 777 "$PROJECT_ROOT/logs" 2>/dev/null || echo "Warning: Could not set permissions on logs directory"

# --- Removed Next.js specific checks and directory logic ---
# --- Removed .env.local creation logic (Streamlit uses different config) ---

# --- Improved Process Killing for Streamlit ---
STREAMLIT_PORT="$PORT" # Use the PORT variable defined above
STREAMLIT_APP_PATH="streamlit_app.py" # Default app path, check existence below

echo "Attempting to stop any existing Streamlit processes on port $STREAMLIT_PORT or running $STREAMLIT_APP_PATH..."

# Method 1: Kill processes listening on the port
pkill -f ":$STREAMLIT_PORT" || true # Allow command to fail if no process found

# Method 2: Kill processes matching the streamlit run command
pkill -f "streamlit run $STREAMLIT_APP_PATH" || true # Allow command to fail

sleep 2 # Give processes time to terminate

# Method 3: Force kill (Optional, uncomment if needed)
# pkill -9 -f "streamlit run $STREAMLIT_APP_PATH" || true

echo "Finished attempting to stop existing Streamlit processes."
# --- End Improved Process Killing ---


# --- Set Streamlit Config Environment Variables ---\nexport STREAMLIT_SERVER_ADDRESS="$HOST" # Set host via env var
export STREAMLIT_SERVER_PORT="$STREAMLIT_PORT" # Set port via env var
export STREAMLIT_SERVER_ENABLE_CORS=true
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false # Also disable usage stats
# --- End Config Env Vars ---

# Check if streamlit_app.py exists
if [ ! -f "$PROJECT_ROOT/$STREAMLIT_APP_PATH" ]; then
    echo "Error: Streamlit app '$STREAMLIT_APP_PATH' not found in project root '$PROJECT_ROOT'."
    # Attempt to find it in common locations
    FOUND_APP_PATH=""
    if [ -f "$PROJECT_ROOT/app/$STREAMLIT_APP_PATH" ]; then
       FOUND_APP_PATH="app/$STREAMLIT_APP_PATH"
    elif [ -f "$PROJECT_ROOT/ui/$STREAMLIT_APP_PATH" ]; then
        FOUND_APP_PATH="ui/$STREAMLIT_APP_PATH"
    elif [ -f "$PROJECT_ROOT/backend/$STREAMLIT_APP_PATH" ]; then # Check backend too?
        FOUND_APP_PATH="backend/$STREAMLIT_APP_PATH"
    fi

    if [ -n "$FOUND_APP_PATH" ]; then
        STREAMLIT_APP_PATH="$FOUND_APP_PATH"
        echo "Found Streamlit app in: $PROJECT_ROOT/$STREAMLIT_APP_PATH"
    else
       echo "Could not find Streamlit app '$STREAMLIT_APP_PATH' in common locations (root, app/, ui/). Please ensure it exists."
       exit 1
    fi
fi

# Activate virtual environment (Important for running streamlit)
VENV_PATH="/opt/venv-sutazaiapp" # Consistent venv path
if [ -d "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
    echo "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Error: streamlit command not found. Ensure it's installed in the virtual environment ($VENV_PATH)."
    exit 1
fi

# Start Streamlit directly
# Use specified port and run headless
echo "Starting Streamlit UI: streamlit run \"$STREAMLIT_APP_PATH\"..."
# Use the specific Web UI log directory
streamlit run "$STREAMLIT_APP_PATH" > "$WEBUI_LOG_DIR/webui.log" 2>&1 &
WEBUI_PID=$!

# Store PID for future reference
mkdir -p "$PROJECT_ROOT/pids" # Ensure pids dir exists
echo "$WEBUI_PID" > "$PROJECT_ROOT/pids/webui.pid"

echo "Web UI (Streamlit) started with PID: $WEBUI_PID"
echo "Logs are being written to: $WEBUI_LOG_DIR/webui.log"

# Verify Web UI has started correctly
echo "Waiting 10 seconds for Streamlit to bind to port $STREAMLIT_PORT..."
sleep 10 # Increased sleep time slightly

# Check based on port listening as PID might be the launcher
if ! ss -tulnp | grep -q ":$STREAMLIT_PORT"; then
    echo "Error: Streamlit UI process failed to start or bind to port $STREAMLIT_PORT."
    echo "Check the logs at $WEBUI_LOG_DIR/webui.log for details."
    if [ -f "$WEBUI_LOG_DIR/webui.log" ]; then
        echo "Last 10 lines of log:"
        tail -n 10 "$WEBUI_LOG_DIR/webui.log"
    fi
    exit 1 # Exit if streamlit fails to start
fi

echo "Web UI successfully started!"
echo "Access the Web UI at: http://localhost:$STREAMLIT_PORT (or http://$HOST:$STREAMLIT_PORT if HOST is not 0.0.0.0)"
echo "To stop the Web UI, use: kill $WEBUI_PID or run the main stop script ($PROJECT_ROOT/bin/stop_all.sh)" # Updated stop instruction

exit 0 # Explicitly exit successfully