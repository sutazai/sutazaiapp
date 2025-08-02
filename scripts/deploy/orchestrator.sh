#!/bin/bash
# SutazAI Supreme Orchestrator CLI
# A simple CLI wrapper for the Supreme AI Orchestrator

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Check if the virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found. Please run setup script first.${NC}"
    exit 1
fi

# Source the virtual environment
source /opt/venv-sutazaiapp/bin/activate

# Check if the orchestrator script exists
ORCHESTRATOR_SCRIPT="${PROJECT_ROOT}/ai_agents/supreme_orchestrator.py"
if [ ! -f "$ORCHESTRATOR_SCRIPT" ]; then
    echo -e "${RED}Error: Supreme Orchestrator script not found at ${ORCHESTRATOR_SCRIPT}${NC}"
    exit 1
fi

# Make sure the orchestrator script is executable
chmod +x "$ORCHESTRATOR_SCRIPT"

# Function to display help
show_help() {
    echo -e "${BOLD}SutazAI Supreme Orchestrator CLI${NC}"
    echo -e "Usage: orchestrator.sh [OPTIONS] [ACTION]"
    echo
    echo -e "${BOLD}Actions:${NC}"
    echo "  monitor   - Start monitoring of servers and services (default)"
    echo "  sync      - Synchronize code to deployment server"
    echo "  deploy    - Deploy application to production"
    echo "  restart   - Restart services on a server"
    echo "  status    - Show status of all servers and services"
    echo
    echo -e "${BOLD}Options:${NC}"
    echo "  --help, -h       - Show this help message"
    echo "  --config=PATH    - Specify custom config file path"
    echo "  --interval=SECS  - Set monitoring interval in seconds (default: 300)"
    echo "  --server=TYPE    - Specify server type (code or deployment, default: deployment)"
    echo "  --sync-mode=MODE - Specify sync mode (normal, fast, full, default: normal)"
    echo
    echo -e "${BOLD}Examples:${NC}"
    echo "  orchestrator.sh monitor --interval=60"
    echo "  orchestrator.sh sync --sync-mode=fast"
    echo "  orchestrator.sh deploy"
    echo "  orchestrator.sh restart --server=code"
    echo "  orchestrator.sh status"
}

# Parse command line arguments
ACTION="monitor"
ARGS=()

# Default values
CONFIG=""
INTERVAL=300
SERVER="deployment"
SYNC_MODE="normal"

# Parse positional and named arguments
for arg in "$@"; do
    case "$arg" in
        monitor|sync|deploy|restart|status)
            ACTION="$arg"
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        --config=*)
            CONFIG="${arg#*=}"
            ARGS+=("--config" "$CONFIG")
            ;;
        --interval=*)
            INTERVAL="${arg#*=}"
            ARGS+=("--interval" "$INTERVAL")
            ;;
        --server=*)
            SERVER="${arg#*=}"
            ARGS+=("--server" "$SERVER")
            ;;
        --sync-mode=*)
            SYNC_MODE="${arg#*=}"
            ARGS+=("--sync-mode" "$SYNC_MODE")
            ;;
        *)
            echo -e "${YELLOW}Warning: Unknown argument '$arg'${NC}"
            ;;
    esac
done

# Add action to args
ARGS+=("--action" "$ACTION")

# Add --local flag for local-only mode (no SSH connections to remote servers)
ARGS+=("--local")

echo -e "${BLUE}Running Supreme Orchestrator with action: ${CYAN}$ACTION${NC} (local-only mode)"

# Check for config directory
CONFIG_DIR="${PROJECT_ROOT}/config"
if [ ! -d "$CONFIG_DIR" ]; then
    echo -e "${YELLOW}Creating config directory...${NC}"
    mkdir -p "$CONFIG_DIR"
fi

# Execute the orchestrator script with proper signal handling
if [ "$ACTION" = "monitor" ]; then
    # For monitoring, we run it in the background
    echo "Starting monitoring. Press Ctrl+C to stop."
    
    # Setup PID file
    PID_FILE="${PROJECT_ROOT}/logs/orchestrator.pid"
    
    # Run in background with nohup
    nohup python "$ORCHESTRATOR_SCRIPT" "${ARGS[@]}" > "${PROJECT_ROOT}/logs/orchestrator.log" 2>&1 &
    
    # Save PID
    echo $! > "$PID_FILE"
    echo -e "${GREEN}Orchestrator started in background with PID $(cat $PID_FILE)${NC}"
    echo -e "To stop: ${CYAN}scripts/stop_orchestrator.sh${NC} or ${CYAN}kill -TERM $(cat $PID_FILE)${NC}"
else
    # For other actions, run in foreground
    python "$ORCHESTRATOR_SCRIPT" "${ARGS[@]}"
    RESULT=$?
fi

# Deactivate virtual environment
deactivate

exit $RESULT 