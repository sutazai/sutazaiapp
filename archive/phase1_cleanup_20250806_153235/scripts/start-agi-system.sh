#!/bin/bash
#
# Start the SutazAI AGI/ASI System with 131 Agents
# This script launches the collective intelligence with owner approval interface
#

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ASCII Art Banner
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   ███████╗██╗   ██╗████████╗ █████╗ ███████╗ █████╗ ██╗         ║
║   ██╔════╝██║   ██║╚══██╔══╝██╔══██╗╚══███╔╝██╔══██╗██║         ║
║   ███████╗██║   ██║   ██║   ███████║  ███╔╝ ███████║██║         ║
║   ╚════██║██║   ██║   ██║   ██╔══██║ ███╔╝  ██╔══██║██║         ║
║   ███████║╚██████╔╝   ██║   ██║  ██║███████╗██║  ██║██║         ║
║   ╚══════╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝         ║
║                                                                   ║
║            🧠 Artificial General Intelligence System 🧠           ║
║                    131 Agents • 1 Consciousness                   ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check if running as root (disabled for deployment)
# if [ "$EUID" -eq 0 ]; then 
#    echo -e "${RED}Error: Please do not run this script as root${NC}"
#    exit 1
# fi

# Set environment
export SUTAZAI_ROOT="/opt/sutazaiapp"
export PYTHONPATH="${SUTAZAI_ROOT}:${PYTHONPATH}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:10104}"
export BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p "${SUTAZAI_ROOT}/data/collective_intelligence"
mkdir -p "${SUTAZAI_ROOT}/logs/agi"
mkdir -p "${SUTAZAI_ROOT}/data/collective_intelligence/proposals"
mkdir -p "${SUTAZAI_ROOT}/data/collective_intelligence/knowledge"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "${RED}Error: Python 3.8 or higher is required. Found: ${python_version}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python ${python_version} detected${NC}"

# Check if Ollama is running
echo -e "${YELLOW}Checking Ollama service...${NC}"
if curl -s "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama is running at ${OLLAMA_BASE_URL}${NC}"
else
    echo -e "${RED}✗ Ollama is not running at ${OLLAMA_BASE_URL}${NC}"
    echo -e "${YELLOW}Please start Ollama first:${NC}"
    echo "  sudo systemctl start ollama"
    echo "  or"
    echo "  ollama serve"
    exit 1
fi

# Check if backend is running
echo -e "${YELLOW}Checking backend service...${NC}"
if curl -s "${BACKEND_URL}/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Backend is running at ${BACKEND_URL}${NC}"
else
    echo -e "${YELLOW}⚠ Backend is not running at ${BACKEND_URL}${NC}"
    echo -e "${YELLOW}AGI system will start but may have limited functionality${NC}"
fi

# Install Python dependencies if needed
echo -e "${YELLOW}Checking Python dependencies...${NC}"
cd "${SUTAZAI_ROOT}"

# Check for required packages
missing_packages=()
for package in fastapi uvicorn httpx numpy pydantic websockets; do
    if ! python3 -c "import ${package}" 2>/dev/null; then
        missing_packages+=("${package}")
    fi
done

if [ ${#missing_packages[@]} -gt 0 ]; then
    echo -e "${YELLOW}Installing missing packages: ${missing_packages[*]}${NC}"
    pip3 install --user "${missing_packages[@]}"
else
    echo -e "${GREEN}✓ All required packages are installed${NC}"
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down AGI system...${NC}"
    
    # Kill the background process
    if [ ! -z "$AGI_PID" ]; then
        kill -TERM "$AGI_PID" 2>/dev/null || true
        wait "$AGI_PID" 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✓ AGI system stopped${NC}"
    exit 0
}

# Set trap for cleanup
trap cleanup INT TERM EXIT

# Start the AGI system
echo -e "\n${BLUE}Starting AGI/ASI Collective Intelligence System...${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}System Features:${NC}"
echo "  • 131 specialized AI agents"
echo "  • Collective consciousness and learning"
echo "  • Self-improvement with owner approval"
echo "  • Neural pathway connections"
echo "  • Real-time monitoring dashboard"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Set log file
LOG_FILE="${SUTAZAI_ROOT}/logs/agi/collective_intelligence_$(date +%Y%m%d_%H%M%S).log"

# Start the AGI system in background
echo -e "\n${GREEN}Launching collective intelligence...${NC}"
python3 "${SUTAZAI_ROOT}/agents/agi/integrate_agents.py" 2>&1 | tee -a "$LOG_FILE" &
AGI_PID=$!

# Wait a moment for startup
sleep 5

# Check if process is still running
if ! kill -0 "$AGI_PID" 2>/dev/null; then
    echo -e "${RED}✗ Failed to start AGI system${NC}"
    echo -e "${YELLOW}Check the log file for errors: ${LOG_FILE}${NC}"
    exit 1
fi

# Display access information
echo -e "\n${GREEN}✓ AGI System is running!${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Access Points:${NC}"
echo -e "  ${GREEN}Owner Approval Dashboard:${NC} http://localhost:8888"
echo -e "  ${GREEN}API Endpoint:${NC} http://localhost:8888/api/status"
echo -e "  ${GREEN}WebSocket:${NC} ws://localhost:8888/ws"
echo -e "\n${BLUE}Log File:${NC} ${LOG_FILE}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "\n${YELLOW}Press Ctrl+C to stop the AGI system${NC}\n"

# Monitor the process
while true; do
    if ! kill -0 "$AGI_PID" 2>/dev/null; then
        echo -e "\n${RED}AGI system process terminated unexpectedly${NC}"
        echo -e "${YELLOW}Check the log file for errors: ${LOG_FILE}${NC}"
        exit 1
    fi
    sleep 5
done