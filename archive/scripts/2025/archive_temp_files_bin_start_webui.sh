#!/bin/bash
# SutazAI Web UI Startup Script

APP_ROOT="/opt/sutazaiapp"
# Explicitly set the correct path to the frontend code
WEB_ROOT="$APP_ROOT/ui"
LOGS_DIR="$APP_ROOT/logs"
PIDS_DIR="$APP_ROOT/pids"
PID_FILE="$PIDS_DIR/webui.pid"
LOG_FILE="$LOGS_DIR/webui.log"

# Ensure directories exist
mkdir -p "$LOGS_DIR"
mkdir -p "$PIDS_DIR"

# Kill any existing Web UI process
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null; then
        echo "Stopping existing Web UI process (PID: $OLD_PID)"
        sudo kill "$OLD_PID"
        sleep 2
        # Force kill if still running
        if ps -p "$OLD_PID" > /dev/null; then
            echo "Force killing Web UI process"
            sudo kill -9 "$OLD_PID"
        fi
    fi
fi

# Check if the explicitly set web UI directory exists
if [ ! -d "$WEB_ROOT" ]; then
    echo "Error: Web UI directory '$WEB_ROOT' not found."
    exit 1
fi

# Start Web UI
echo "Starting SutazAI Web UI..."
cd "$WEB_ROOT" || exit 1

# Check for streamlit_app.py FIRST, as it doesn't need a requirements.txt in the UI dir
if [ -f "streamlit_app.py" ]; then
    echo "Starting Streamlit UI..."
    # Activate venv if not already active
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        echo "Activating venv for Streamlit..."
        source "$APP_ROOT/venv-sutazaiapp/bin/activate"
    fi
    # Run streamlit in the background, redirecting output
    streamlit run streamlit_app.py --server.port 8501 --server.headless true --server.enableCORS=false --server.enableWebsocketCompression=false >> "$LOG_FILE" 2>&1 &
    STREAMLIT_INITIAL_PID=$!
    echo "Streamlit launched with initial PID: $STREAMLIT_INITIAL_PID"

    echo "Waiting 10 seconds for Streamlit to bind to port 8501..."
    sleep 10

    WEBUI_PID=""
    # Check if port 8501 is listening
    if ss -tuln | grep ':8501' > /dev/null; then
        echo "Streamlit appears to be listening on port 8501."
        # Try to find the specific streamlit process PID
        FOUND_PID=$(pgrep -f "streamlit run streamlit_app.py.*--server.port 8501" | head -n 1)
        if [ -n "$FOUND_PID" ]; then
            echo "Found specific Streamlit process PID: $FOUND_PID"
            WEBUI_PID="$FOUND_PID"
        else
            echo "Warning: Could not find specific Streamlit process via pgrep. Checking initial PID."
            if ps -p "$STREAMLIT_INITIAL_PID" > /dev/null; then
                echo "Initial PID $STREAMLIT_INITIAL_PID is still running. Using it as fallback."
                WEBUI_PID="$STREAMLIT_INITIAL_PID"
            else
                echo "Error: Cannot find Streamlit process via pgrep, and initial PID $STREAMLIT_INITIAL_PID is gone."
            fi
        fi
    else
        echo "Error: Streamlit did not bind to port 8501 after 10 seconds."
        # Attempt to kill the initial launcher PID if it's still around
        if ps -p "$STREAMLIT_INITIAL_PID" > /dev/null; then
            echo "Attempting to kill initial Streamlit launcher PID $STREAMLIT_INITIAL_PID."
            sudo kill "$STREAMLIT_INITIAL_PID" 2>/dev/null
        fi
    fi

    # Save PID if found
    if [[ "$WEBUI_PID" =~ ^[0-9]+$ ]]; then
        echo "Saving valid PID $WEBUI_PID to $PID_FILE"
        echo "$WEBUI_PID" > "$PID_FILE"
        echo "Web UI (Streamlit) successfully started on port 8501."
        echo "Access the interface at: http://localhost:8501"
    else
        echo "Error: Failed to obtain a valid PID for Streamlit. Check logs at $LOG_FILE"
        rm -f "$PID_FILE" # Clean up potentially empty PID file
        exit 1
    fi
# Check for package.json (Node.js app)
elif [ -f "package.json" ]; then
    # Node.js web UI
    if ! command -v npm &> /dev/null; then
        echo "Error: npm not found. Please install Node.js and npm."
        exit 1
    fi
    
    # Install dependencies if node_modules doesn't exist
    if [ ! -d "node_modules" ]; then
        echo "Installing dependencies..."
        npm install >> "$LOG_FILE" 2>&1
    fi
    
    # Start the application
    echo "Starting Node.js Web UI..."
    # Use dev server for potentially better logging/stability
    npm run dev >> "$LOG_FILE" 2>&1 &
    WEBUI_PID=$!
# Check for requirements.txt (Other Python app)
elif [ -f "requirements.txt" ]; then
    # Python web UI (non-streamlit)
    echo "Starting Python Web UI..."
    # Install dependencies
    python3 -m pip install -r requirements.txt >> "$LOG_FILE" 2>&1
    
    if [ -f "app.py" ]; then
        echo "Starting Python UI with app.py..."
        python3 app.py >> "$LOG_FILE" 2>&1 &
    elif [ -f "main.py" ]; then
        echo "Starting Python UI with main.py..."
        python3 main.py >> "$LOG_FILE" 2>&1 &
    elif [ -f "server.py" ]; then
        echo "Starting Python UI with server.py..."
        python3 server.py >> "$LOG_FILE" 2>&1 &
    else
        echo "Error: No Python entry point found (app.py/main.py/server.py) despite requirements.txt."
        exit 1
    fi
    WEBUI_PID=$!
# Try to find the startup file
else
    if [ -f "index.js" ]; then
        echo "Starting Node.js Web UI with index.js..."
        node index.js >> "$LOG_FILE" 2>&1 &
        WEBUI_PID=$!
    elif [ -f "start.sh" ]; then
        echo "Running existing start.sh script..."
        bash start.sh >> "$LOG_FILE" 2>&1 &
        WEBUI_PID=$!
    else
        echo "Error: Cannot determine how to start the Web UI."
        exit 1
    fi
fi

# Save PID
# Check if WEBUI_PID is a valid number before saving
if ! [[ "$WEBUI_PID" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid or missing PID captured for Web UI. Check $LOG_FILE for errors."
    # Attempt to find it again just in case
    WEBUI_PID=$(pgrep -f "streamlit run streamlit_app.py")
    if ! [[ "$WEBUI_PID" =~ ^[0-9]+$ ]]; then
        echo "Error: Still cannot find valid Web UI PID."
        exit 1
    fi
    echo "Found valid PID $WEBUI_PID after re-check."
fi
echo "$WEBUI_PID" > "$PID_FILE"
echo "Web UI potentially started with PID: $WEBUI_PID"
echo "Logs are being written to: $LOG_FILE"

# Wait a moment and check if the specific streamlit process is running using pgrep
sleep 3
if pgrep -f "streamlit run streamlit_app.py" > /dev/null; then
    echo "Web UI (Streamlit) successfully started on port 8501."
    echo "Access the interface at: http://localhost:8501"
# Fallback check using the saved PID if pgrep fails (less reliable)
elif ps -p "$WEBUI_PID" > /dev/null; then
    echo "Warning: Streamlit process check failed, but PID $WEBUI_PID exists. Check logs."
    echo "Access the interface at: http://localhost:8501"
else
    echo "Error: Web UI failed to start. Check logs at $LOG_FILE"
    exit 1
fi 