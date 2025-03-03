#!/bin/bash

# Supreme AI Orchestrator Installation Script
# This script installs and configures the Supreme AI Orchestrator service.

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="supreme-ai-orchestrator"
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
SYSTEMD_DIR="/etc/systemd/system"
VENV_PATH="$PROJECT_ROOT/venv"
CONFIG_DIR="$PROJECT_ROOT/config"
CERT_DIR="$CONFIG_DIR/certs"

# Function to check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}Error: This script must be run as root${NC}"
        exit 1
    fi
}

# Function to create required directories
create_directories() {
    echo -e "${YELLOW}Creating required directories...${NC}"
    
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$CERT_DIR"
    mkdir -p "$PROJECT_ROOT/data"
    
    # Set correct permissions
    chown -R sutazaiapp_dev:sutazaiapp_dev "$PROJECT_ROOT"
    chmod -R 750 "$PROJECT_ROOT"
}

# Function to setup Python virtual environment
setup_venv() {
    echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
    
    if [ ! -d "$VENV_PATH" ]; then
        python3.11 -m venv "$VENV_PATH"
    fi
    
    # Upgrade pip and install requirements
    sudo -u sutazaiapp_dev bash -c "
        source $VENV_PATH/bin/activate
        pip install --upgrade pip setuptools wheel
        pip install -r $PROJECT_ROOT/requirements.txt
    "
}

# Function to generate SSL certificates
generate_certificates() {
    echo -e "${YELLOW}Generating SSL certificates...${NC}"
    
    if [ ! -f "$CERT_DIR/server.key" ] || [ ! -f "$CERT_DIR/server.crt" ]; then
        openssl req -x509 -newkey rsa:4096 -nodes \
            -keyout "$CERT_DIR/server.key" \
            -out "$CERT_DIR/server.crt" \
            -days 365 \
            -subj "/C=JP/ST=Tokyo/L=Tokyo/O=SutazAI/OU=Development/CN=localhost"
        
        # Set correct permissions
        chown sutazaiapp_dev:sutazaiapp_dev "$CERT_DIR/server.key" "$CERT_DIR/server.crt"
        chmod 600 "$CERT_DIR/server.key"
        chmod 644 "$CERT_DIR/server.crt"
    else
        echo -e "${YELLOW}Certificates already exist, skipping generation${NC}"
    fi
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
    if [ ! -f "$PROJECT_ROOT/logs/orchestrator.log" ]; then
        echo -e "${RED}Log file not created${NC}"
        exit 1
    fi
    
    # Check process
    if ! pgrep -f "supreme_ai_orchestrator" > /dev/null; then
        echo -e "${RED}Orchestrator process not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Installation verified successfully${NC}"
}

# Main installation process
main() {
    echo -e "${YELLOW}Starting Supreme AI Orchestrator installation...${NC}"
    
    # Check if running as root
    check_root
    
    # Create required directories
    create_directories
    
    # Setup virtual environment
    setup_venv
    
    # Generate SSL certificates
    generate_certificates
    
    # Install systemd service
    install_service
    
    # Verify installation
    verify_installation
    
    echo -e "${GREEN}Supreme AI Orchestrator installation completed successfully${NC}"
}

# Run main installation
main 