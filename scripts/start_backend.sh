#!/bin/bash

# ==============================================================================
# SutazAI AGI - Backend Startup Script
# ==============================================================================
# Description: Starts the FastAPI backend server using Uvicorn.
#              Provides configuration details and access URLs.
# Usage:       bash scripts/start_backend.sh [--workers N]
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- SutazAI Backend Startup Initializing ---"

# --- Argument Parsing ---
# Initialize variables
WORKERS_ARG=""

# Process command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --workers)
        WORKERS_ARG="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        echo "[WARN] Unknown option: $1"
        shift # past argument
        ;;
    esac
done

# --- Environment Setup ---
echo "[INFO] Setting up script environment..."
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT" || { echo "[ERROR] Failed to change directory to project root: $PROJECT_ROOT"; exit 1; }
echo "[INFO] Current directory set to project root: $PROJECT_ROOT"

# Activate virtual environment if it exists
VENV_ACTIVATED=false
if [ -d "venv" ]; then
    echo "[INFO] Activating virtual environment from 'venv'..."
    source venv/bin/activate
    VENV_ACTIVATED=true
elif [ -d ".venv" ]; then
    echo "[INFO] Activating virtual environment from '.venv'..."
    source .venv/bin/activate
    VENV_ACTIVATED=true
else
    echo "[WARN] No virtual environment ('venv' or '.venv') found or activated."
fi

# Check for essential command: uvicorn
if ! command -v uvicorn &> /dev/null; then
    echo "[ERROR] 'uvicorn' command not found. Please ensure Uvicorn is installed in the environment."
    exit 1
fi

# Check if main backend file exists
BACKEND_ENTRY="sutazai_agi/backend/main.py"
if [ ! -f "$BACKEND_ENTRY" ]; then
    echo "[ERROR] Backend entry point not found at $BACKEND_ENTRY"
    exit 1
fi
echo "[INFO] Backend entry point found: $BACKEND_ENTRY"

# --- Configuration ---
echo "[INFO] Determining backend configuration..."
# Set default host and port, allow overrides via environment variables
HOST=${SUTAZAI_BACKEND_HOST:-"0.0.0.0"}
PORT=${SUTAZAI_BACKEND_PORT:-8000}
# Default workers from ENV or 4, overridden by argument if provided
WORKERS=${SUTAZAI_BACKEND_WORKERS:-4}
if [[ -n "$WORKERS_ARG" ]]; then
    # Basic validation: check if it's a positive integer
    if [[ "$WORKERS_ARG" =~ ^[1-9][0-9]*$ ]]; then
        echo "[INFO] Overriding worker count with command-line argument: $WORKERS_ARG"
        WORKERS=$WORKERS_ARG
    else
        echo "[WARN] Invalid value for --workers argument: '$WORKERS_ARG'. Using default/env value: $WORKERS"
    fi
fi
RELOAD_FLAG=${SUTAZAI_DEV_MODE:-false} # Set SUTAZAI_DEV_MODE=true for development reload

echo "--------------------------------------------------"
echo " Configuration:"
echo "--------------------------------------------------"
echo "  Backend Host (SUTAZAI_BACKEND_HOST): $HOST"
echo "  Backend Port (SUTAZAI_BACKEND_PORT): $PORT"
echo "  Worker Processes (SUTAZAI_BACKEND_WORKERS): $WORKERS"

UVICORN_CMD="uvicorn sutazai_agi.backend.main:app --host $HOST --port $PORT --workers $WORKERS"

if [ "$RELOAD_FLAG" = true ] ; then
    echo "  Mode (SUTAZAI_DEV_MODE): Development (Reload Enabled)"
    echo "[WARN] Running in development mode with reload. Not recommended for production."
    # Note: reload flag often conflicts with multi-worker, typically used with 1 worker
    UVICORN_CMD="uvicorn sutazai_agi.backend.main:app --host $HOST --port $PORT --reload"
else
    echo "  Mode (SUTAZAI_DEV_MODE): Production"
fi
echo "--------------------------------------------------"

# --- Execution ---
echo "[INFO] Preparing to launch Uvicorn server..."
echo "[CMD] $UVICORN_CMD"
echo "--------------------------------------------------"
echo " Access Information:"
echo "--------------------------------------------------"
echo "  Backend API accessible at: http://$HOST:$PORT"
echo "  API Documentation (Swagger): http://$HOST:$PORT/docs"
# Determine likely Web UI URL - This is an assumption!
# Use hostname if HOST is 0.0.0.0 or 127.0.0.1, otherwise use HOST
if [[ "$HOST" == "0.0.0.0" || "$HOST" == "127.0.0.1" ]]; then
  DISPLAY_HOST=$(hostname -I | awk '{print $1}' || echo "localhost") # Try to get primary IP
else
  DISPLAY_HOST=$HOST
fi
# Assume standard OpenWebUI port 8080 (update if different)
WEBUI_PORT=${OPEN_WEBUI_PORT:-8080}
echo "  Likely OpenWebUI Frontend: http://$DISPLAY_HOST:$WEBUI_PORT"
echo "  (Note: Web UI URL depends on how OpenWebUI is deployed and configured.)"
echo "--------------------------------------------------"

echo "[INFO] Starting Uvicorn server... (Press CTRL+C to quit)"

# Execute Uvicorn - Use exec to replace the shell process with uvicorn
# This ensures signals like CTRL+C are handled correctly by uvicorn
exec $UVICORN_CMD

# --- Cleanup (This part might not be reached if exec is used) ---
echo "[INFO] Backend server stopped."
if [ "$VENV_ACTIVATED" = true ] && command -v deactivate &> /dev/null; then
    echo "[INFO] Deactivating virtual environment."
    deactivate
fi

echo "--- SutazAI Backend Startup Script Finished ---"
exit 0 