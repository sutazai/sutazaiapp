#!/bin/bash

# Script to start the SutazAI Streamlit UI

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

# Check if main UI file exists
UI_ENTRY="sutazai_agi/ui/SutazAI_UI.py"
if [ ! -f "$UI_ENTRY" ]; then
    echo "Error: UI entry point not found at $UI_ENTRY"
    exit 1
fi

# Set default port, allow override via environment variable
PORT=${SUTAZAI_UI_PORT:-8501}

STREAMLIT_CMD="streamlit run $UI_ENTRY --server.port $PORT --server.address 0.0.0.0"

echo "Starting Streamlit UI..."
echo "Running command: $STREAMLIT_CMD"
echo "Access the UI in your browser (usually http://localhost:$PORT or http://<server_ip>:$PORT)"

# Execute Streamlit
$STREAMLIT_CMD

# Deactivate environment if applicable
if command -v deactivate &> /dev/null; then
    if [[ "$VIRTUAL_ENV" == "$PROJECT_ROOT/venv" || "$VIRTUAL_ENV" == "$PROJECT_ROOT/.venv" ]]; then
      deactivate
    fi
fi

exit 0 