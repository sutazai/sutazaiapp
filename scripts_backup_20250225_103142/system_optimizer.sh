#!/usr/bin/env bash
# Comprehensive System Optimization Script for SutazAI

set -euo pipefail

# Logging
LOG_FILE="/var/log/sutazai_system_optimization.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to log messages
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Check and set CPU governor to performance
optimize_cpu() {
    log "${YELLOW}Optimizing CPU Performance${NC}"
    
    # Check if cpufreq-set is available
    if command -v cpufreq-set &> /dev/null; then
        for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
            echo performance | sudo tee "$cpu"/cpufreq/scaling_governor
        done
        log "${GREEN}CPU governor set to performance${NC}"
    else
        log "${RED}cpufreq-tools not installed. Skipping CPU optimization.${NC}"
    fi
}

# Optimize system memory
optimize_memory() {
    log "${YELLOW}Optimizing Memory Settings${NC}"
    
    # Adjust swappiness
    sudo sysctl vm.swappiness=10
    
    # Enable memory overcommit
    sudo sysctl vm.overcommit_memory=1
    
    log "${GREEN}Memory optimization completed${NC}"
}

# Clean up system caches and temporary files
system_cleanup() {
    log "${YELLOW}Performing System Cleanup${NC}"
    
    # Clear package manager cache
    sudo apt-get clean
    sudo apt-get autoremove -y
    
    # Clear system journal logs
    sudo journalctl --vacuum-time=3d
    
    # Clear temporary files
    sudo find /tmp -type f -atime +7 -delete
    
    log "${GREEN}System cleanup completed${NC}"
}

# Optimize disk I/O
optimize_disk_io() {
    log "${YELLOW}Optimizing Disk I/O${NC}"
    
    # Use deadline scheduler for better SSD performance
    for disk in /sys/block/sd*; do
        echo deadline | sudo tee "$disk"/queue/scheduler
    done
    
    log "${GREEN}Disk I/O optimization completed${NC}"
}

# Install and optimize Python environment
python_environment_setup() {
    log "${YELLOW}Setting Up Python Environment${NC}"
    
    # Upgrade pip and setuptools
    python3 -m pip install --upgrade pip setuptools wheel
    
    # Install performance-related packages
    python3 -m pip install \
        cython \
        numpy \
        numba \
        psutil \
        py-spy
    
    log "${GREEN}Python environment optimized${NC}"
}

# Main optimization function
main() {
    log "${GREEN}Starting SutazAI System Optimization${NC}"
    
    optimize_cpu
    optimize_memory
    optimize_disk_io
    system_cleanup
    python_environment_setup
    
    log "${GREEN}System Optimization Complete!${NC}"
}

# Run the main function
main 