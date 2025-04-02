#!/bin/bash
# title        :install_services.sh
# description  :This script installs the systemd service and sets up cron jobs
# author       :SutazAI Team
# version      :1.0
# usage        :sudo bash scripts/install_services.sh
# notes        :Requires bash 4.0+ and standard Linux utilities

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Check if script is run with sudo
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Please run this script with sudo or as root"
    exit 1
fi

# Validate system Python installation
if ! python3 -c "import sys; assert sys.version_info >= (3,11)"; then
    echo -e "${RED}[ERROR]${NC} Python 3.11+ required"
    exit 1
fi

# Validate system Python installation
if ! python3 -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11+ required'"; then
    echo -e "${RED}[ERROR]${NC} Python 3.11+ required"
    exit 1
fi

# Step 0: Install and verify Python environment
echo -e "${BLUE}[INFO]${NC} Installing Python environment dependencies..."
apt-get update
apt-get install -y \
    python3.11=3.11.8-1~22.04 \
    python3.11-venv=3.11.8-1~22.04 \
    python3.11-dev=3.11.8-1~22.04 \
    virtualenv=20.16.7+ds-1

# Add virtualenv to requirements with version pinning
if ! grep -q "virtualenv==" "${PROJECT_ROOT}/requirements.txt"; then
    echo "virtualenv==20.16.7" >> "${PROJECT_ROOT}/requirements.txt"
fi

echo -e "${GREEN}[SUCCESS]${NC} Python dependencies installed"

echo -e "${BLUE}[INFO]${NC} Installing SutazAI services and cron jobs"

# Step 1: Install all systemd services
echo -e "${BLUE}[INFO]${NC} Installing systemd services..."

# Make sure systemd directory exists
mkdir -p /etc/systemd/system

# Copy all service files
cp "$PROJECT_ROOT/systemd/"*.service /etc/systemd/system/
cp "$PROJECT_ROOT/systemd/"*.timer /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

echo -e "${GREEN}[SUCCESS]${NC} Systemd services installed"

# Step 2: Enable services to start on boot
echo -e "${BLUE}[INFO]${NC} Enabling services to start on boot..."
systemctl enable sutazai.service
systemctl enable sutazai-service-monitor.timer
systemctl enable sutazai-memory-optimizer.timer
systemctl enable sutazai-cpu-monitor.service
systemctl enable sutazai-service-monitor.service

# Step 3: Set up cron jobs
echo -e "${BLUE}[INFO]${NC} Setting up cron jobs..."
bash "$PROJECT_ROOT/scripts/setup_cron_jobs.sh"

# Step 4: Verify installations
echo -e "${BLUE}[INFO]${NC} Verifying installations..."

if systemctl is-enabled sutazai.service >/dev/null 2>&1; then
    echo -e "${GREEN}[SUCCESS]${NC} SutazAI service is enabled"
else
    echo -e "${RED}[ERROR]${NC} Failed to enable SutazAI service"
fi

# Step 5: Set up environment for the application
echo -e "${BLUE}[INFO]${NC} Setting up environment..."

# Create required directories
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/tmp"
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/storage"
mkdir -p "$PROJECT_ROOT/uploads"
mkdir -p "$PROJECT_ROOT/workspace"
mkdir -p "$PROJECT_ROOT/pids"

# Set proper permissions
chown -R sutazaidev:sutazaidev "$PROJECT_ROOT"
chmod -R 755 "$PROJECT_ROOT/scripts"
chmod 644 "$PROJECT_ROOT/.env"

# Step 6: Optimize database
echo -e "${BLUE}[INFO]${NC} Optimizing database..."
bash "$PROJECT_ROOT/scripts/cleanup_redundancies.sh"

# Step 7: Start all services if requested
if [ "$1" = "--start" ]; then
    echo -e "${BLUE}[INFO]${NC} Starting all services..."
    systemctl start sutazai.service
    systemctl start sutazai-service-monitor.timer
    systemctl start sutazai-memory-optimizer.timer
    systemctl start sutazai-cpu-monitor.service
    
    # Check if main service started successfully
    if systemctl is-active --quiet sutazai.service; then
        echo -e "${GREEN}[SUCCESS]${NC} SutazAI service started successfully"
    else
        echo -e "${RED}[ERROR]${NC} Failed to start SutazAI service"
        systemctl status sutazai.service
    fi
fi

# Final summary
echo -e "\n${GREEN}[SUCCESS]${NC} SutazAI installation completed successfully!"
echo -e "\n${BOLD}Services installed:${NC}"
echo -e "  - SutazAI main service (sutazai.service)"
echo -e "  - SutazAI service monitor (sutazai-service-monitor.service)"
echo -e "  - SutazAI CPU monitor (sutazai-cpu-monitor.service)"
echo -e "  - SutazAI memory optimizer (sutazai-memory-optimizer.service)"
echo -e "  - Automated maintenance cron jobs"
echo -e "\n${BOLD}Next steps:${NC}"
echo -e "  1. Start the service: ${BLUE}sudo systemctl start sutazai.service${NC}"
echo -e "  2. Check service status: ${BLUE}sudo systemctl status sutazai.service${NC}"
echo -e "  3. View logs: ${BLUE}sudo journalctl -u sutazai.service${NC}"
echo -e "\nThe application will also start automatically on system boot."

exit 0 