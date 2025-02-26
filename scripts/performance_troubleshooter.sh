#!/usr/bin/env bash
# SutazAI Performance Troubleshooter

set -euo pipefail

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Logging
LOG_DIR="/opt/sutazaiapp/logs/troubleshooting"
TROUBLESHOOT_LOG="$LOG_DIR/performance_troubleshoot_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$TROUBLESHOOT_LOG"
}

# Check and install required system tools
install_system_tools() {
    log "${YELLOW}Checking and Installing System Tools${NC}"
    
    # List of required tools
    REQUIRED_TOOLS=(
        "sysstat"     # For mpstat, iostat
        "net-tools"   # For netstat
        "htop"        # System monitoring
        "lsof"        # Process and port information
        "build-essential"  # Compilation tools
        "software-properties-common"  # For adding repositories
    )

    # Update package lists
    sudo apt-get update

    # Install required tools
    for tool in "${REQUIRED_TOOLS[@]}"; do
        if ! dpkg -s "$tool" >/dev/null 2>&1; then
            log "${YELLOW}Installing $tool${NC}"
            sudo apt-get install -y "$tool"
        else
            log "${GREEN}$tool is already installed${NC}"
        fi
    done
}

# Diagnose Python environment issues
diagnose_python_environment() {
    log "${YELLOW}Diagnosing Python Environment${NC}"

    # Check if Python 3.11 is installed
    if ! command -v python3.11 &> /dev/null; then
        log "${RED}Python 3.11 not found. Installing...${NC}"
        sudo add-apt-repository ppa:deadsnakes/ppa -y
        sudo apt-get update
        sudo apt-get install -y python3.11 python3.11-dev python3.11-venv
    fi

    # Check Python version
    log "${GREEN}Python Version:${NC}"
    python3.11 --version | tee -a "$TROUBLESHOOT_LOG"

    # Check pip version
    log "${GREEN}Pip Version:${NC}"
    python3.11 -m pip --version | tee -a "$TROUBLESHOOT_LOG"

    # Check for virtual environment
    if [ ! -d "/opt/sutazaiapp/venv" ]; then
        log "${RED}Virtual environment not found. Creating...${NC}"
        python3.11 -m venv /opt/sutazaiapp/venv
    fi

    # Activate virtual environment
    source /opt/sutazaiapp/venv/bin/activate

    # Upgrade pip and setuptools
    python3.11 -m pip install --upgrade pip setuptools wheel

    # Verify pip installation
    python3.11 -m pip list | tee -a "$TROUBLESHOOT_LOG"

    deactivate
}

# Check system resource constraints
check_resource_constraints() {
    log "${YELLOW}Checking System Resource Constraints${NC}"

    # Memory information
    log "${GREEN}Memory Information:${NC}"
    free -h | tee -a "$TROUBLESHOOT_LOG"

    # Disk space
    log "${GREEN}Disk Space:${NC}"
    df -h | tee -a "$TROUBLESHOOT_LOG"

    # CPU information
    log "${GREEN}CPU Information:${NC}"
    lscpu | tee -a "$TROUBLESHOOT_LOG"

    # Check for memory or swap issues
    if [ "$(free | awk '/Swap:/ {print $2}')" -eq 0 ]; then
        log "${YELLOW}No swap space detected. Creating swap file...${NC}"
        sudo fallocate -l 4G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
    fi
}

# Optimize system performance
optimize_performance() {
    log "${YELLOW}Optimizing System Performance${NC}"

    # Kernel parameter optimization
    sudo tee /etc/sysctl.d/99-sutazai-performance.conf > /dev/null << EOF
# SutazAI Performance Optimization

# Improve file system performance
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288

# Memory management
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.overcommit_memory = 1

# Network optimization
net.core.netdev_max_backlog = 65536
net.core.somaxconn = 65536
net.ipv4.tcp_max_syn_backlog = 65536
net.ipv4.tcp_slow_start_after_idle = 0
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_timestamps = 1

# CPU performance
kernel.sched_migration_cost_ns = 5000000
kernel.sched_autogroup_enabled = 1
EOF

    # Apply kernel parameters
    sudo sysctl --system
}

# Main troubleshooting function
main() {
    log "${GREEN}Starting SutazAI Performance Troubleshooter${NC}"

    # Run diagnostic and optimization steps
    install_system_tools
    diagnose_python_environment
    check_resource_constraints
    optimize_performance

    log "${GREEN}Performance Troubleshooting Complete${NC}"
    log "${YELLOW}Detailed log available at: $TROUBLESHOOT_LOG${NC}"
}

# Execute main function
main 