#!/bin/bash

# Script to start the SutazAI FastAPI backend server

# Ensure script is run from the project root directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT" || exit 1

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check if main backend file exists
BACKEND_ENTRY="sutazai_agi/backend/main.py"
if [ ! -f "$BACKEND_ENTRY" ]; then
    echo "Error: Backend entry point not found at $BACKEND_ENTRY"
    exit 1
fi

# Set default host and port, allow overrides via environment variables
HOST=${SUTAZAI_BACKEND_HOST:-"0.0.0.0"}
PORT=${SUTAZAI_BACKEND_PORT:-8000}
WORKERS=${SUTAZAI_BACKEND_WORKERS:-4} # Number of Uvicorn workers
RELOAD_FLAG=${SUTAZAI_DEV_MODE:-false} # Set SUTAZAI_DEV_MODE=true for development reload

UVICORN_CMD="uvicorn sutazai_agi.backend.main:app --host $HOST --port $PORT --workers $WORKERS"

if [ "$RELOAD_FLAG" = true ] ; then
    echo "Starting backend in development mode (with reload)..."
    # Note: reload flag often conflicts with multi-worker, typically used with 1 worker
    UVICORN_CMD="uvicorn sutazai_agi.backend.main:app --host $HOST --port $PORT --reload"
else
    echo "Starting backend in production mode..."
fi

echo "Running command: $UVICORN_CMD"
echo "Access the API at http://$HOST:$PORT (e.g., http://$HOST:$PORT/docs for Swagger UI)"

# Execute Uvicorn
$UVICORN_CMD

# Deactivate environment if applicable
if command -v deactivate &> /dev/null; then
    if [[ "$VIRTUAL_ENV" == "$PROJECT_ROOT/venv" || "$VIRTUAL_ENV" == "$PROJECT_ROOT/.venv" ]]; then
      deactivate
    fi
fi

exit 0 