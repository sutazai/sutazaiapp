#!/usr/bin/env bash

# SutazAI Master Control Script
# This script consolidates functionality from multiple shell scripts into a single control interface

# Set strict error handling
set -euo pipefail
IFS=$'\n\t'

# Configuration
SUTAZAI_ROOT="/opt/sutazaiapp"
LOG_DIR="${SUTAZAI_ROOT}/logs"
SCRIPTS_DIR="${SUTAZAI_ROOT}/scripts"
BACKUP_DIR="${SCRIPTS_DIR}/_script_backups"

# Ensure required directories exist
mkdir -p "${LOG_DIR}" "${BACKUP_DIR}"

# Logging setup
LOG_FILE="${LOG_DIR}/master_control.log"
exec 1> >(tee -a "${LOG_FILE}")
exec 2> >(tee -a "${LOG_FILE}" >&2)

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Verify Python version
verify_python_version() {
    local python_version
    python_version=$(python3.11 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
    
    if [[ "$python_version" != "3.11" ]]; then
        log_error "Python 3.11 is required but found version $python_version"
        return 1
    fi
    
    log_info "Python version verified: $python_version"
    return 0
}

# System health check
check_system_health() {
    log_info "Running system health check..."
    
    # Check disk space
    local disk_usage
    disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    if (( disk_usage > 90 )); then
        log_warn "High disk usage: ${disk_usage}%"
    fi
    
    # Check memory usage
    local memory_usage
    memory_usage=$(free | awk '/Mem:/ {printf("%.0f", $3/$2 * 100)}')
    if (( memory_usage > 90 )); then
        log_warn "High memory usage: ${memory_usage}%"
    fi
    
    # Check CPU load
    local cpu_load
    cpu_load=$(uptime | awk -F'load average:' '{ print $2 }' | cut -d, -f1)
    if (( $(echo "$cpu_load > 5" | bc -l) )); then
        log_warn "High CPU load: ${cpu_load}"
    fi
    
    log_info "System health check completed"
}

# Deploy application
deploy_application() {
    log_info "Starting application deployment..."
    
    # Verify Python version first
    verify_python_version || {
        log_error "Python version check failed"
        return 1
    }
    
    # Create required directories
    mkdir -p "${SUTAZAI_ROOT}"/{logs,data,config,temp}
    
    # Install dependencies
    log_info "Installing dependencies..."
    python3.11 -m pip install -r "${SUTAZAI_ROOT}/requirements.txt"
    
    # Run deployment script
    log_info "Running deployment script..."
    python3.11 "${SCRIPTS_DIR}/sutazai_unified_manager.py"
    
    log_info "Deployment completed successfully"
}

# Monitor system resources
monitor_resources() {
    log_info "Starting resource monitoring..."
    
    while true; do
        # Get system metrics
        local cpu_usage
        local memory_usage
        local disk_usage
        
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}')
        memory_usage=$(free | awk '/Mem:/ {printf("%.2f", $3/$2 * 100)}')
        disk_usage=$(df -h / | awk 'NR==2 {print $5}')
        
        # Log metrics
        log_info "System Metrics:"
        log_info "  CPU Usage: ${cpu_usage}%"
        log_info "  Memory Usage: ${memory_usage}%"
        log_info "  Disk Usage: ${disk_usage}"
        
        sleep 60
    done
}

# Rotate logs
rotate_logs() {
    log_info "Starting log rotation..."
    
    find "${LOG_DIR}" -type f -name "*.log" -mtime +7 -exec gzip {} \;
    find "${LOG_DIR}" -type f -name "*.log.gz" -mtime +30 -delete
    
    log_info "Log rotation completed"
}

# Main menu
show_menu() {
    echo -e "\nSutazAI Master Control"
    echo "===================="
    echo "1) Deploy Application"
    echo "2) Check System Health"
    echo "3) Monitor Resources"
    echo "4) Rotate Logs"
    echo "5) Run Unified Manager"
    echo "q) Quit"
    echo
    read -rp "Select an option: " choice
    
    case "$choice" in
        1) deploy_application ;;
        2) check_system_health ;;
        3) monitor_resources ;;
        4) rotate_logs ;;
        5) python3.11 "${SCRIPTS_DIR}/sutazai_unified_manager.py" ;;
        q) exit 0 ;;
        *) log_error "Invalid option" ;;
    esac
}

# Main execution
main() {
    log_info "Starting SutazAI Master Control"
    
    # Verify Python version
    verify_python_version || {
        log_error "Python version check failed"
        exit 1
    }
    
    # Show menu in a loop
    while true; do
        show_menu
    done
}

# Run main function
main 