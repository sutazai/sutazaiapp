#!/bin/bash
# Comprehensive deployment script that utilizes existing scripts
# This script combines the best of all existing scripts

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}     SutazaiApp Deployment Script        ${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# Step 1: Check environment
echo -e "${YELLOW}Step 1: Checking environment...${NC}"
./scripts/check_environment.sh
if [ $? -ne 0 ]; then
    echo -e "${RED}Environment check failed!${NC}"
    echo -e "${YELLOW}Continuing anyway, but there might be issues.${NC}"
fi
echo ""

# Step 2: Stop any running services
echo -e "${YELLOW}Step 2: Stopping any running services...${NC}"
# First try systemd services
if command -v systemctl &> /dev/null; then
    sudo systemctl stop sutazaiapp sutazai-orchestrator 2>/dev/null || true
fi

# Then try to kill any running Python processes for the application
echo "Stopping any running Python processes..."
pkill -f "python.*backend.main" 2>/dev/null || true
pkill -f "python.*orchestrator.orchestrator" 2>/dev/null || true
pkill -f "python.*uvicorn.*backend.main" 2>/dev/null || true
echo ""

# Step 3: Ensure directories exist with proper permissions
echo -e "${YELLOW}Step 3: Setting up directories...${NC}"
mkdir -p logs data/documents data/models data/vectors run tmp
chmod -R 755 logs data run tmp
echo ""

# Step 4: Set up virtual environment if needed
echo -e "${YELLOW}Step 4: Checking virtual environment...${NC}"
if [ ! -L "venv" ] || [ ! -d "venv" ]; then
    echo "Virtual environment link not found, creating..."
    ln -sf /opt/venv-sutazaiapp /opt/sutazaiapp/venv
fi
echo ""

# Step 5: Start the application using the start.sh script
echo -e "${YELLOW}Step 5: Starting application components...${NC}"
./scripts/start.sh
echo ""

# Step 6: Install systemd services if available
echo -e "${YELLOW}Step 6: Setting up systemd services...${NC}"
if command -v systemctl &> /dev/null; then
    echo "Installing systemd services..."
    if [ -f "systemd/sutazaiapp.service" ]; then
        sudo cp systemd/sutazaiapp.service /etc/systemd/system/
        echo "Installed sutazaiapp.service"
    fi
    
    if [ -f "systemd/sutazai-orchestrator.service" ]; then
        sudo cp systemd/sutazai-orchestrator.service /etc/systemd/system/
        echo "Installed sutazai-orchestrator.service"
    fi
    
    sudo systemctl daemon-reload
    sudo systemctl enable sutazaiapp sutazai-orchestrator
    echo "Systemd services installed and enabled"
fi
echo ""

# Step 7: Attempt to start web UI if Node.js is available
echo -e "${YELLOW}Step 7: Starting Web UI (if available)...${NC}"
if command -v npm &> /dev/null; then
    echo "Node.js found, attempting to start web UI..."
    if [ -f "scripts/start_webui.sh" ]; then
        ./scripts/start_webui.sh &
        echo "Web UI started"
    else
        echo "Web UI start script not found"
    fi
else
    echo "Node.js not found, skipping web UI"
fi
echo ""

# Step 8: Verify deployment with check_status.sh
echo -e "${YELLOW}Step 8: Verifying deployment...${NC}"
./scripts/check_status.sh
echo ""

# Step 9: Final health check
echo -e "${YELLOW}Step 9: Running health check...${NC}"
if [ -f "scripts/health_check.sh" ]; then
    ./scripts/health_check.sh
fi
echo ""

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}     SutazaiApp Deployment Complete      ${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo -e "Access the API at: ${GREEN}http://localhost:8000${NC}"
echo -e "Access the Web UI at: ${GREEN}http://localhost:3000${NC} (if available)"
echo ""
echo -e "Use ${YELLOW}./scripts/check_status.sh${NC} to check service status"
echo -e "Use ${YELLOW}sudo systemctl start/stop sutazaiapp${NC} to control the API service"
echo -e "Use ${YELLOW}sudo systemctl start/stop sutazai-orchestrator${NC} to control the orchestrator"
echo "" 