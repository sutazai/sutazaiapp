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

# Check if node_modules exists, install dependencies if not
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Check if we need to build the app
if [ ! -d ".next" ] || [ "$NODE_ENV" = "production" ]; then
    echo "Building the app..."
    npm run build
fi

# Set up environment for the Next.js app
export NEXT_PUBLIC_API_URL="$API_URL"
export PORT="$PORT"  # Set PORT explicitly to 3000
export HOSTNAME="$HOST"  # Set the hostname environment variable

# Start the app based on environment
if [ "$NODE_ENV" = "production" ]; then
    echo "Starting in production mode..."
    # Start Next.js in production mode with specific host and port
    npm run start -- -H "$HOST" -p "$PORT" 2>&1 | tee -a "$WEBUI_LOG_DIR/webui.log" &
    WEBUI_PID=$!
    # Store PID in both locations for backward compatibility
    echo $WEBUI_PID > "$LOG_DIR/webui.pid"
    echo $WEBUI_PID > "/opt/sutazaiapp/.webui.pid"
    echo "Web UI started with PID: $WEBUI_PID"
else
    echo "Starting in development mode..."
    # Use development server with specific host and port
    npm run dev -- -H "$HOST" -p "$PORT" 2>&1 | tee -a "$WEBUI_LOG_DIR/webui.log" &
    WEBUI_PID=$!
    # Store PID in both locations for backward compatibility
    echo $WEBUI_PID > "$LOG_DIR/webui.pid"
    echo $WEBUI_PID > "/opt/sutazaiapp/.webui.pid"
    echo "Web UI started with PID: $WEBUI_PID"
fi

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

# Kill existing webui process if running
if [ -f ".webui.pid" ]; then
    OLD_PID=$(cat .webui.pid)
    if ps -p "$OLD_PID" > /dev/null; then
        echo "Stopping existing Web UI process (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
        # Force kill if still running
        kill -9 "$OLD_PID" 2>/dev/null || true
        echo "Previous Web UI process stopped."
    else
        echo "Stale Web UI PID file found. Cleaning up..."
    fi
    rm -f ".webui.pid"
fi

# Check if port 3000 is already in use by a different process
if netstat -tuln | grep -q ":3000 "; then
    PID=$(lsof -i :3000 -t 2>/dev/null || echo "unknown")
    if [ "$PID" != "unknown" ]; then
        # Check if it's our own process
        OUR_PROCESS=$(ps -p "$PID" -o cmd= 2>/dev/null | grep -q "node.*next" && echo "yes" || echo "no")
        if [ "$OUR_PROCESS" = "yes" ]; then
            echo "Web UI is already running with PID: $PID"
            # Update PID file
            echo $PID > "$LOGS_DIR/webui.pid"
            # Use sudo if needed to ensure we can write the PID file
            if [ -w "." ]; then
                echo $PID > ".webui.pid"
            else
                echo "Using sudo to write PID file (may prompt for password)..."
                sudo sh -c "echo $PID > \"$PROJECT_ROOT/.webui.pid\""
            fi
            echo "Web UI PID files updated. Access at: http://localhost:3000"
            exit 0
        else
            echo "Error: Port 3000 is already in use by another process (PID: $PID)."
            echo "Please stop that process first and try again."
            exit 1
        fi
    else
        echo "Error: Port 3000 is already in use, but cannot determine the process."
        echo "Please free up port 3000 and try again."
        exit 1
    fi
fi

# Navigate to the app directory
cd app || { echo "Error: app directory not found"; exit 1; }

# Check if package.json exists in app directory
if [ ! -f "package.json" ]; then
    echo "Error: package.json not found in app directory. Cannot start the Web UI."
    exit 1
fi

# Install Node.js dependencies if node_modules doesn't exist or if npm-shrinkwrap.json is newer
if [ ! -d "node_modules" ] || [ "package.json" -nt "node_modules" ] || [ -f "npm-shrinkwrap.json" -a "npm-shrinkwrap.json" -nt "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
    # Check if installation was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install Node.js dependencies."
        exit 1
    fi
fi

# Build the Next.js application if .next doesn't exist or if source files are newer
if [ ! -d ".next" ] || [ -n "$(find src -newer .next -type f -name "*.js" -o -name "*.jsx" -o -name "*.ts" -o -name "*.tsx" 2>/dev/null)" ]; then
    echo "Building Next.js application..."
    npx next build
    # Check if build was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build Next.js application."
        exit 1
    fi
fi

# Start the Next.js server
echo "Starting SutazAI Web UI..."
npx next start > "../$LOGS_DIR/webui.log" 2>&1 &
WEBUI_PID=$!

# Store PID for future reference
cd ..
echo $WEBUI_PID > "$LOGS_DIR/webui.pid"
# Use sudo if needed to ensure we can write the PID file
if [ -w "." ]; then
    echo $WEBUI_PID > ".webui.pid"
else
    echo "Using sudo to write PID file (may prompt for password)..."
    sudo sh -c "echo $WEBUI_PID > \"$PROJECT_ROOT/.webui.pid\""
fi

echo "Web UI started with PID: $WEBUI_PID"
echo "Logs are being written to: $LOGS_DIR/webui.log"

# Verify Web UI has started correctly
sleep 3
if ! ps -p $WEBUI_PID > /dev/null; then
    echo "Error: Web UI process failed to start or terminated immediately."
    echo "Check the logs at $LOGS_DIR/webui.log for details."
    if [ -f "$LOGS_DIR/webui.log" ]; then
        echo "Last 10 lines of log:"
        tail -n 10 "$LOGS_DIR/webui.log"
    fi
    exit 1
fi

echo "Web UI successfully started!"
echo "Access the Web UI at: http://localhost:3000"
echo "To stop the Web UI, run: scripts/stop_webui.sh" 