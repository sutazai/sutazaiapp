#!/usr/bin/env bash
# Comprehensive System Performance Diagnostics and Optimization Script

set -euo pipefail

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Logging
LOG_DIR="/var/log/sutazai_performance"
DIAGNOSTIC_LOG="$LOG_DIR/system_diagnostics_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$DIAGNOSTIC_LOG"
}

# Comprehensive system diagnostics
run_system_diagnostics() {
    log "${YELLOW}Starting Comprehensive System Diagnostics${NC}"

    # System Information
    log "${GREEN}System Information:${NC}"
    uname -a | tee -a "$DIAGNOSTIC_LOG"
    cat /proc/cpuinfo | grep "model name" | uniq | tee -a "$DIAGNOSTIC_LOG"
    free -h | tee -a "$DIAGNOSTIC_LOG"
    df -h | tee -a "$DIAGNOSTIC_LOG"

    # Top resource consumers
    log "${GREEN}Top Resource Consumers:${NC}"
    ps aux --sort=-%cpu,-%mem | head -n 15 | tee -a "$DIAGNOSTIC_LOG"

    # I/O Wait and CPU Steal Time
    log "${GREEN}I/O Wait and CPU Steal Time:${NC}"
    mpstat 1 5 | tee -a "$DIAGNOSTIC_LOG"

    # Disk I/O Performance
    log "${GREEN}Disk I/O Performance:${NC}"
    iostat -x 1 5 | tee -a "$DIAGNOSTIC_LOG"

    # Network Statistics
    log "${GREEN}Network Statistics:${NC}"
    netstat -tuln | tee -a "$DIAGNOSTIC_LOG"
}

# Advanced system optimization
optimize_system() {
    log "${YELLOW}Applying Advanced System Optimizations${NC}"

    # Kernel Parameters Optimization
    log "${GREEN}Optimizing Kernel Parameters${NC}"
    sudo tee /etc/sysctl.d/99-sutazai-optimization.conf << EOF
# SutazAI Performance Optimization

# Increase maximum number of open file descriptors
fs.file-max = 2097152

# Improve memory management
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.overcommit_memory = 1

# Network stack optimization
net.core.netdev_max_backlog = 65536
net.core.somaxconn = 65536
net.ipv4.tcp_max_syn_backlog = 65536
net.ipv4.tcp_slow_start_after_idle = 0
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_timestamps = 1

# CPU and Performance
kernel.sched_migration_cost_ns = 5000000
kernel.sched_autogroup_enabled = 1
EOF

    # Apply kernel parameters
    sudo sysctl --system

    # CPU Governor Performance
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        echo performance | sudo tee "$cpu"/cpufreq/scaling_governor
    done

    # Disable unnecessary services
    services_to_disable=(
        "bluetooth.service"
        "cups.service"
        "avahi-daemon.service"
        "ModemManager.service"
    )

    for service in "${services_to_disable[@]}"; do
        sudo systemctl disable "$service" || true
        sudo systemctl stop "$service" || true
    done

    log "${GREEN}System Optimization Complete${NC}"
}

# Comprehensive cleanup
system_cleanup() {
    log "${YELLOW}Performing System Cleanup${NC}"

    # Package management cleanup
    sudo apt-get update
    sudo apt-get autoremove -y
    sudo apt-get autoclean

    # Journal log management
    sudo journalctl --vacuum-time=3d
    sudo journalctl --vacuum-size=100M

    # Clear temporary files
    sudo find /tmp -type f -atime +7 -delete
    sudo find /var/tmp -type f -atime +7 -delete

    log "${GREEN}System Cleanup Complete${NC}"
}

# Python environment optimization
python_environment_optimization() {
    log "${YELLOW}Optimizing Python Environment${NC}"

    # Upgrade pip and setuptools
    python3 -m pip install --upgrade pip setuptools wheel

    # Install performance-related packages
    python3 -m pip install \
        cython \
        numpy \
        numba \
        psutil \
        py-spy \
        memory_profiler

    # Compile Python bytecode
    python3 -m compileall /opt/sutazai_project/SutazAI

    log "${GREEN}Python Environment Optimization Complete${NC}"
}

# Main execution
main() {
    log "${GREEN}Starting SutazAI Performance Optimization${NC}"

    # Run diagnostics first
    run_system_diagnostics

    # Apply optimizations
    optimize_system
    system_cleanup
    python_environment_optimization

    log "${GREEN}Performance Optimization Process Complete${NC}"
    log "${YELLOW}Diagnostic Log: $DIAGNOSTIC_LOG${NC}"
}

# Execute main function
main 