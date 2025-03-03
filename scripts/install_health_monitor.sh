#!/bin/bash

# Supreme AI Health Monitor Installation Script
# This script installs and configures the health monitor service.

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="supreme-ai-health-monitor"
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
SYSTEMD_DIR="/etc/systemd/system"
VENV_PATH="$PROJECT_ROOT/venv"
LOG_DIR="$PROJECT_ROOT/logs"

# Function to check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}Error: This script must be run as root${NC}"
        exit 1
    fi
}

# Function to check dependencies
check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    # Check Python version
    if ! command -v python3.11 &> /dev/null; then
        echo -e "${RED}Error: Python 3.11 is required${NC}"
        exit 1
    fi
    
    # Check virtual environment
    if [ ! -d "$VENV_PATH" ]; then
        echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
        exit 1
    fi
}

# Function to create required directories
create_directories() {
    echo -e "${YELLOW}Creating required directories...${NC}"
    
    mkdir -p "$LOG_DIR"
    
    # Set correct permissions
    chown -R sutazaiapp_dev:sutazaiapp_dev "$LOG_DIR"
    chmod -R 750 "$LOG_DIR"
}

# Function to install Python dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    
    sudo -u sutazaiapp_dev bash -c "
        source $VENV_PATH/bin/activate
        pip install prometheus_client psutil aiohttp
    "
}

# Function to install systemd service
install_service() {
    echo -e "${YELLOW}Installing systemd service...${NC}"
    
    # Copy service file
    cp "$SERVICE_FILE" "$SYSTEMD_DIR/$SERVICE_NAME.service"
    chmod 644 "$SYSTEMD_DIR/$SERVICE_NAME.service"
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable and start service
    systemctl enable "$SERVICE_NAME"
    systemctl start "$SERVICE_NAME"
    
    # Check service status
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${GREEN}Service installed and started successfully${NC}"
    else
        echo -e "${RED}Service installation failed${NC}"
        exit 1
    fi
}

# Function to verify installation
verify_installation() {
    echo -e "${YELLOW}Verifying installation...${NC}"
    
    # Check service status
    if ! systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${RED}Service is not running${NC}"
        exit 1
    fi
    
    # Check log file creation
    if [ ! -f "$LOG_DIR/health_monitor.log" ]; then
        echo -e "${RED}Log file not created${NC}"
        exit 1
    fi
    
    # Check metrics endpoint
    if ! curl -s http://localhost:9090/metrics > /dev/null; then
        echo -e "${RED}Metrics endpoint not accessible${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Installation verified successfully${NC}"
}

# Main installation process
main() {
    echo -e "${YELLOW}Starting Supreme AI Health Monitor installation...${NC}"
    
    # Check if running as root
    check_root
    
    # Check dependencies
    check_dependencies
    
    # Create required directories
    create_directories
    
    # Install Python dependencies
    install_dependencies
    
    # Install systemd service
    install_service
    
    # Verify installation
    verify_installation
    
    echo -e "${GREEN}Supreme AI Health Monitor installation completed successfully${NC}"
    echo -e "Metrics available at: http://localhost:9090/metrics"
    echo -e "Logs available at: $LOG_DIR/health_monitor.log"
}

# Run main installation
main 