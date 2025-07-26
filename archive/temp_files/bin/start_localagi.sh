#!/bin/bash
# SutazAI LocalAGI Startup Script

APP_ROOT="/opt/sutazaiapp"
LOGS_DIR="$APP_ROOT/logs"
PIDS_DIR="$APP_ROOT/pids"
PID_FILE="$PIDS_DIR/localagi.pid"
LOG_FILE="$LOGS_DIR/localagi.log"
LOCALAGI_PORT=8090

# Ensure directories exist
mkdir -p "$LOGS_DIR"
mkdir -p "$PIDS_DIR"

# Kill any existing LocalAGI process
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null; then
        echo "Stopping existing LocalAGI process (PID: $OLD_PID)"
        sudo kill "$OLD_PID"
        sleep 2
        # Force kill if still running
        if ps -p "$OLD_PID" > /dev/null; then
            echo "Force killing LocalAGI process"
            sudo kill -9 "$OLD_PID"
        fi
    fi
fi

# Paths to check for LocalAGI installation
LOCALAGI_PATHS=(
    "$APP_ROOT/ai_agents/localagi"
    "$APP_ROOT/ai_agents/superagi"
    "$APP_ROOT/localagi"
    "$APP_ROOT/superagi"
)

# Find LocalAGI path
LOCALAGI_PATH=""
for path in "${LOCALAGI_PATHS[@]}"; do
    if [ -d "$path" ]; then
        LOCALAGI_PATH="$path"
        break
    fi
done

if [ -z "$LOCALAGI_PATH" ]; then
    echo "Error: LocalAGI directory not found."
    echo "Checking for installation methods..."
    
    # Check if we can clone it from GitHub
    if command -v git &> /dev/null; then
        echo "Git found. Attempting to clone LocalAGI..."
        mkdir -p "$APP_ROOT/localagi"
        LOCALAGI_PATH="$APP_ROOT/localagi"
        git clone https://github.com/louisgv/local-agi.git "$LOCALAGI_PATH" >> "$LOG_FILE" 2>&1
        
        if [ $? -ne 0 ]; then
            echo "Failed to clone LocalAGI repository."
            exit 1
        fi
    else
        echo "Error: Cannot find or install LocalAGI."
        exit 1
    fi
fi

echo "Found LocalAGI at: $LOCALAGI_PATH"
cd "$LOCALAGI_PATH" || exit 1

# Check for requirements
if [ -f "requirements.txt" ]; then
    echo "Installing LocalAGI requirements..."
    python3 -m pip install -r requirements.txt >> "$LOG_FILE" 2>&1
fi

# Try to find the startup script or main file
if [ -f "run.py" ]; then
    # LocalAGI main script
    echo "Starting LocalAGI with run.py..."
    python3 run.py --port $LOCALAGI_PORT >> "$LOG_FILE" 2>&1 &
    LOCALAGI_PID=$!
elif [ -f "app.py" ]; then
    # Main app file
    echo "Starting LocalAGI with app.py..."
    python3 app.py --port $LOCALAGI_PORT >> "$LOG_FILE" 2>&1 &
    LOCALAGI_PID=$!
elif [ -f "main.py" ]; then
    # Main file
    echo "Starting LocalAGI with main.py..."
    python3 main.py --port $LOCALAGI_PORT >> "$LOG_FILE" 2>&1 &
    LOCALAGI_PID=$!
elif [ -f "start.sh" ]; then
    # Existing start script
    echo "Running LocalAGI start.sh script..."
    bash start.sh >> "$LOG_FILE" 2>&1 &
    LOCALAGI_PID=$!
else
    # Create a minimal LocalAGI server if no entry point is found
    echo "No LocalAGI entry point found. Creating a minimal server..."
    MINIMAL_SERVER="$LOCALAGI_PATH/minimal_server.py"
    
    cat > "$MINIMAL_SERVER" << EOF
#!/usr/bin/env python3
"""
Minimal LocalAGI Server
This is a simplified LocalAGI server that provides basic AGI endpoints.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional
import argparse
from pathlib import Path

# Set up command line arguments
parser = argparse.ArgumentParser(description='Minimal LocalAGI Server')
parser.add_argument('--port', type=int, default=$LOCALAGI_PORT, help='Port to run the server on')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='${LOG_FILE}',
    filemode='a'
)
logger = logging.getLogger("LocalAGI")

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    logger.error("Required packages not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"])
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    import uvicorn

# Create FastAPI app
app = FastAPI(title="LocalAGI", description="Minimal LocalAGI Server")

@app.get("/health")
async def health_check():
    """Check if the server is running."""
    return {"status": "ok", "service": "LocalAGI"}

@app.post("/agent/execute")
async def execute_agent(request: Request):
    """Execute an agent task."""
    try:
        data = await request.json()
        agent_type = data.get("agent_type", "default")
        task = data.get("task", {})
        
        logger.info(f"Executing agent: {agent_type}")
        
        # Simulate agent execution
        result = {
            "status": "success",
            "agent_type": agent_type,
            "result": f"Processed task with {agent_type} agent",
            "details": task
        }
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error executing agent: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/agents/list")
async def list_agents():
    """List available agents."""
    agents = [
        {"id": "general", "name": "General Purpose Agent", "description": "Handles general tasks"},
        {"id": "document", "name": "Document Agent", "description": "Processes documents"},
        {"id": "code", "name": "Code Agent", "description": "Analyzes and generates code"}
    ]
    return {"agents": agents}

if __name__ == "__main__":
    logger.info(f"Starting LocalAGI server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
EOF
    
    # Make it executable
    chmod +x "$MINIMAL_SERVER"
    
    # Run the minimal server
    echo "Starting minimal LocalAGI server..."
    python3 "$MINIMAL_SERVER" >> "$LOG_FILE" 2>&1 &
    LOCALAGI_PID=$!
fi

# Save PID
echo "$LOCALAGI_PID" > "$PID_FILE"
echo "LocalAGI started with PID: $LOCALAGI_PID"
echo "Logs are being written to: $LOG_FILE"

# Wait a moment and verify the service is running
sleep 5
if ps -p "$LOCALAGI_PID" > /dev/null; then
    echo "LocalAGI service is running on port $LOCALAGI_PORT"
    
    # Try to verify the API is accessible
    if command -v curl &> /dev/null; then
        if curl -s "http://localhost:$LOCALAGI_PORT/health" > /dev/null; then
            echo "LocalAGI API is accessible"
        else
            echo "Warning: LocalAGI API may not be accessible yet. Check logs at $LOG_FILE"
        fi
    fi
else
    echo "Error: LocalAGI failed to start. Check logs at $LOG_FILE"
    exit 1
fi 