#!/bin/bash
# SutazAI Backend Startup Script
# Optimized for Dell PowerEdge R720 with E5-2640 processors

APP_ROOT="/opt/sutazaiapp"
LOGS_DIR="$APP_ROOT/logs"
PIDS_DIR="$APP_ROOT/pids"
PID_FILE="$PIDS_DIR/backend.pid"
LOG_FILE="$LOGS_DIR/backend.log"

# Ensure directories exist
mkdir -p "$LOGS_DIR"
mkdir -p "$PIDS_DIR"

# Kill any existing backend process
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null; then
        echo "Stopping existing backend process (PID: $OLD_PID)"
        # Use sudo to ensure permission
        sudo kill "$OLD_PID"
        sleep 2
        # Force kill if still running
        if ps -p "$OLD_PID" > /dev/null; then
            echo "Force killing backend process"
            # Use sudo to ensure permission
            sudo kill -9 "$OLD_PID"
        fi
    fi
    # Remove PID file after attempting kill
    rm -f "$PID_FILE"
fi

# Set thread count for optimal performance on E5-2640 (typically 12 cores)
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export NUMEXPR_MAX_THREADS=12

# Start backend
echo "Starting SutazAI backend..."
cd "$APP_ROOT" || exit 1

# --- Activate Virtual Environment ---
VENV_PATH="$APP_ROOT/venv-sutazaiapp"
if [ -d "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
    echo "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi
# ----------------------------------

# Check if uvicorn is installed (should be in venv)
if ! command -v uvicorn &> /dev/null; then
    echo "Error: uvicorn not found in the current environment (should be in venv)."
    exit 1
fi

# Start the backend with optimized settings for E5-2640
# Run in background, redirect output to log file
# Note: $! will capture the PID of the initial python process, which might exit.
# The check below verifies if the *service* is listening, which is more reliable.
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level debug --timeout-keep-alive 600 >> "$LOG_FILE" 2>&1 &
BACKEND_PID=$!

# Save PID (useful for stopping later, even if the check uses port)
echo "$BACKEND_PID" > "$PID_FILE"
echo "Backend launching with PID: $BACKEND_PID (Note: This might be the launcher PID)"
echo "Logs are being written to: $LOG_FILE"

# Wait a bit longer and check if the port is listening
echo "Waiting 20 seconds for backend to bind to port 8000..."
sleep 20
if ss -tuln | grep ':8000' > /dev/null; then
    echo "Backend successfully started and listening on port 8000."
    # Find the actual worker PID and update the PID file
    WORKER_PID=$(pgrep -f "uvicorn.*backend.main:app.*worker.*port 8000" | head -n 1)
    if [ -n "$WORKER_PID" ]; then
        echo "Found worker PID: $WORKER_PID. Updating PID file."
        echo "$WORKER_PID" > "$PID_FILE"
    else
        echo "Warning: Could not find specific worker PID, keeping initial PID $BACKEND_PID."
        # Check if initial PID is still valid as a fallback
        if ! ps -p "$BACKEND_PID" > /dev/null; then
             echo "Error: Initial PID $BACKEND_PID is also gone. Backend likely failed."
             rm -f "$PID_FILE"
             exit 1
        fi
    fi
else
    echo "Error: Backend failed to start or is not listening on port 8000 after 20 seconds."
    echo "Check logs at $LOG_FILE"
    # Optional: Attempt to kill the launcher PID if it's still around somehow
    if ps -p "$BACKEND_PID" > /dev/null; then
        sudo kill "$BACKEND_PID" 2>/dev/null
    fi
    rm -f "$PID_FILE" # Clean up PID file on failure
    exit 1
fi 