#!/bin/bash
# SutazAI SuperAGI Startup Script
# This script starts the SuperAGI agent for SutazAI

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
SUPERAGI_LOG="${PROJECT_ROOT}/logs/superagi.log"
mkdir -p "$(dirname "$SUPERAGI_LOG")"

# Check if the virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found. Please run setup script first.${NC}"
    exit 1
fi

# Source the virtual environment
source /opt/venv-sutazaiapp/bin/activate

# Check if config.toml exists
if [ ! -f "${PROJECT_ROOT}/ai_agents/superagi/config.toml" ]; then
    echo -e "${RED}Error: SuperAGI config file not found.${NC}"
    echo "Expected location: ${PROJECT_ROOT}/ai_agents/superagi/config.toml"
    exit 1
fi

# Check if necessary model files exist
MODEL_PATH=$(grep "model_path" "${PROJECT_ROOT}/ai_agents/superagi/config.toml" | cut -d '"' -f 2 | sed 's/\.\.\/\.\.\///')
MODEL_FILE=$(grep "model_file" "${PROJECT_ROOT}/ai_agents/superagi/config.toml" | cut -d '"' -f 2)

if [ ! -d "${PROJECT_ROOT}/${MODEL_PATH}" ]; then
    echo -e "${YELLOW}Warning: Model directory not found: ${PROJECT_ROOT}/${MODEL_PATH}${NC}"
    echo "Creating model directory..."
    mkdir -p "${PROJECT_ROOT}/${MODEL_PATH}"
fi

if [ ! -f "${PROJECT_ROOT}/${MODEL_PATH}/${MODEL_FILE}" ]; then
    echo -e "${YELLOW}Warning: Model file not found: ${MODEL_PATH}/${MODEL_FILE}${NC}"
    echo "Please download the required model file and place it in the correct location."
    echo "You can continue, but the agent may not function correctly without the model."
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Please set up the model files first."
        exit 1
    fi
fi

# Create necessary directories if they don't exist
mkdir -p "${PROJECT_ROOT}/logs/agent"
mkdir -p "${PROJECT_ROOT}/workspace"
mkdir -p "${PROJECT_ROOT}/outputs"
mkdir -p "${PROJECT_ROOT}/storage"

# Check if supreme_agent.py exists
if [ ! -f "${PROJECT_ROOT}/ai_agents/superagi/supreme_agent.py" ]; then
    echo -e "${RED}Error: SuperAGI agent file not found.${NC}"
    echo "Expected location: ${PROJECT_ROOT}/ai_agents/superagi/supreme_agent.py"
    exit 1
fi

# Check if we need to run as a service or in the foreground
if [ "$1" = "--service" ]; then
    RUN_AS_SERVICE=true
else
    RUN_AS_SERVICE=false
fi

# Set environment variables
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}

echo -e "${BLUE}Starting SutazAI SuperAGI agent...${NC}"

# Change to the SuperAGI directory
cd "${PROJECT_ROOT}/ai_agents/superagi"

# Start the SuperAGI agent
if [ "$RUN_AS_SERVICE" = true ]; then
    # Run as a background service
    python -c "
import sys
sys.path.append('${PROJECT_ROOT}')
from ai_agents.superagi.supreme_agent import SupremeAgent
import time

agent = SupremeAgent('${PROJECT_ROOT}/ai_agents/superagi/config.toml')
agent.initialize()
print('SuperAGI agent initialized and running in service mode')
# Keep the script running
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print('Stopping SuperAGI agent')
    agent.stop()
" > "${SUPERAGI_LOG}" 2>&1 &

    SUPERAGI_PID=$!
    # Store PID in both locations for backward compatibility
    echo $SUPERAGI_PID > "${PROJECT_ROOT}/logs/superagi.pid"
    echo $SUPERAGI_PID > "${PROJECT_ROOT}/.superagi.pid"
    echo -e "${GREEN}SuperAGI agent started in service mode with PID: $SUPERAGI_PID${NC}"
    echo "Logs are being written to: $SUPERAGI_LOG"
    echo "To stop the agent, run: kill $SUPERAGI_PID"
else
    # Run in the foreground
    echo "Starting SuperAGI agent in foreground mode. Press Ctrl+C to stop."
    python -c "
import sys
sys.path.append('${PROJECT_ROOT}')
from ai_agents.superagi.supreme_agent import SupremeAgent

agent = SupremeAgent('${PROJECT_ROOT}/ai_agents/superagi/config.toml')
agent.initialize()
print('SuperAGI agent initialized and ready to process tasks')
print('Example: agent.run(\"Generate a Python function to calculate Fibonacci numbers\")')
print('Press Ctrl+D to exit the interactive session')
" || {
        echo -e "${RED}Failed to start SuperAGI agent.${NC}"
        exit 1
    }
fi

# Add an entry to the agent log
echo "[$(date)] - SuperAGI agent started" >> "${PROJECT_ROOT}/logs/agent.log"

# If we're running in the foreground, deactivate the virtual environment when done
if [ "$RUN_AS_SERVICE" = false ]; then
    deactivate
fi

exit 0
