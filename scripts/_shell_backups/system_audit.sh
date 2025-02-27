#!/bin/bash

# System Audit Script - Comprehensive System Check
# Version 1.0
# Author: Your Name

# Add at the beginning of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"
source "${SCRIPT_DIR}/password_manager.sh"
source "${SCRIPT_DIR}/voice_verification.sh"

# Load configuration
CONFIG_FILE="${SCRIPT_DIR}/system_audit.conf"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    log_message "ERROR: Configuration file not found"
    exit 1
fi

# Define output file
AUDIT_LOG="/var/log/system_audit_$(date +%Y%m%d_%H%M%S).log"

# Add at the beginning of the script
MAX_LOG_SIZE=10485760 # 10MB
if [ -f "$AUDIT_LOG" ] && [ $(stat -c%s "$AUDIT_LOG") -gt $MAX_LOG_SIZE ]; then
    mv "$AUDIT_LOG" "${AUDIT_LOG}.old"
fi

# Add at the beginning of the script
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root"
    exit 1
fi

# Check for suspicious environment variables
if [ -n "$LD_PRELOAD" ] || [ -n "$LD_LIBRARY_PATH" ]; then
    log_message "WARNING: Suspicious environment variables detected"
    exit 1
fi

if ! verify_password; then
    log_message "ERROR: Incorrect master password"
    exit 1
fi

if ! voice_verification; then
    log_message "ERROR: Voice verification failed"
    exit 1
fi

# Function to log messages
log_message() {
    local level="INFO"
    if [[ "$1" == "WARNING:"* ]] || [[ "$1" == "ERROR:"* ]]; then
        level=$(echo "$1" | cut -d: -f1)
    fi
    printf "[%s] [%-7s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$level" "$1" | tee -a $AUDIT_LOG
}

# Add after log_message function
check_dependencies() {
    local scripts=(
        "system_cleanup.sh"
        "system_health.sh"
        "system_backup.sh"
        "system_update.sh"
        "log_analysis.sh"
        "system_monitor.sh"
        "log_rotation.sh"
        "resource_report.sh"
        "uptime_report.sh"
    )
    
    local missing=0
    for script in "${scripts[@]}"; do
        if [ ! -f "$script" ]; then
            log_message "ERROR: Required script $script not found"
            missing=$((missing + 1))
        fi
    done
    
    if [ $missing -gt 0 ]; then
        log_message "FATAL: Missing $missing required scripts"
        exit 1
    fi
}

# Call it after header
log_message "Starting Comprehensive System Audit"
check_dependencies

# Modify the source command blocks to include error checking
run_script() {
    local script_name=$1
    local retries=3
    local attempt=0
    local result=1
    
    while [ $attempt -lt $retries ]; do
        if [ -f "$script_name" ]; then
            if source "$script_name" >> $AUDIT_LOG 2>&1; then
                result=0
                break
            else
                log_message "WARNING: Attempt $((attempt + 1)) failed for $script_name"
                sleep 5
            fi
        else
            log_message "ERROR: Script $script_name not found"
            return 2
        fi
        attempt=$((attempt + 1))
    done
    
    if [ $result -ne 0 ]; then
        log_message "ERROR: Failed to execute $script_name after $retries attempts"
        return 1
    fi
    return 0
}

# Add timing function
time_script() {
    local start_time=$(date +%s)
    "$@"
    local end_time=$(date +%s)
    echo $((end_time - start_time))
}

# Add resource monitoring function
monitor_resources() {
    local interval=5
    local duration=$1
    local end_time=$((SECONDS + duration))
    
    while [ $SECONDS -lt $end_time ]; do
        log_message "Resource Usage - CPU: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')%, Memory: $(free -m | awk '/Mem:/ { printf "%.1f%%", $3/$2*100 }')"
        sleep $interval
    done
}

# Add performance tracking
track_performance() {
    local start_time=$(date +%s)
    "$@"
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $duration -gt 300 ]; then
        log_message "WARNING: Long execution time detected ($duration seconds)"
        log_message "Consider optimizing the following:"
        log_message "1. Parallel execution of independent tasks"
        log_message "2. Reducing log verbosity"
        log_message "3. Implementing caching mechanisms"
    fi
}

# Add before other checks
log_message "=== Collecting System Information ==="
run_script "system_info.sh"
log_message "System Information Collection Completed"

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulnerability_scan.sh"
log_message "Vulnerability Scan Completed"


# Final Summary
log_message "=== System Audit Summary ==="
TOTAL_CHECKS=10
SUCCESS_CHECKS=$(grep -c "Completed$" $AUDIT_LOG)
FAILED_CHECKS=$((TOTAL_CHECKS - SUCCESS_CHECKS))

log_message "Total Checks Run: $TOTAL_CHECKS"
log_message "Successful Checks: $SUCCESS_CHECKS"
log_message "Failed Checks: $FAILED_CHECKS"

# Check for errors
ERROR_COUNT=$(grep -i "error" $AUDIT_LOG | wc -l)
WARNING_COUNT=$(grep -i "warning" $AUDIT_LOG | wc -l)

log_message "Total Errors Found: $ERROR_COUNT"
log_message "Total Warnings Found: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ] || [ $WARNING_COUNT -gt 0 ]; then
    log_message "WARNING: Issues detected in system audit. Please review the log file."
    exit 1
else
    log_message "SUCCESS: No critical issues found in system audit"
    exit 0
fi

# Add at the end of the script
send_notification() {
    local message="System audit found $ERROR_COUNT errors and $WARNING_COUNT warnings"
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "System Audit Report" "$ADMIN_EMAIL"
    fi
    
    # Slack notification
    if command -v curl &> /dev/null && [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" "$SLACK_WEBHOOK" >/dev/null 2>&1
    fi
    
    # Log notification
    log_message "Notification sent: $message"
}

send_notification

validate_environment() {
    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        log_message "ERROR: Script must be run as root"
        exit 1
    fi
    
    # Check for suspicious environment variables
    local suspicious_vars=("LD_PRELOAD" "LD_LIBRARY_PATH")
    for var in "${suspicious_vars[@]}"; do
        if [ -n "${!var}" ]; then
            log_message "WARNING: Suspicious environment variable $var detected"
            exit 1
        fi
    done
    
    # Check script integrity
    local script_hash=$(sha256sum "$0" | awk '{print $1}')
    if [ "$script_hash" != "$(cat /etc/system_audit.hash 2>/dev/null)" ]; then
        log_message "WARNING: Script integrity check failed"
        exit 1
    fi
}

# Call at the beginning of the script
validate_environment

# Example usage
track_performance run_script "system_cleanup.sh"

# Add help function
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -q, --quiet       Suppress non-essential output"
    echo "  -c, --config      Specify alternate config file"
    echo
    echo "Comprehensive system audit script that checks and resolves system issues."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            log_message "ERROR: Unknown option $1"
            show_help
            ;;
    esac
done

# Check system requirements
# - Disk space
# - Memory
# - CPU
# - OS version
# - Required packages 

# Add these checks at the beginning of the script
check_system_requirements() {
    echo "🔍 Checking system requirements..." | tee -a $AUDIT_LOG
    
    # Disk space
    local min_disk=20 # GB
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ $disk_space -lt $min_disk ]; then
        handle_error "Insufficient disk space: $disk_space GB (minimum $min_disk GB required)"
    fi
    
    # Memory
    local min_memory=4 # GB
    local memory=$(free -g | awk '/Mem:/ {print $2}')
    if [ $memory -lt $min_memory ]; then
        handle_error "Insufficient memory: $memory GB (minimum $min_memory GB required)"
    fi
    
    # CPU
    local min_cpu=2
    local cpu=$(nproc)
    if [ $cpu -lt $min_cpu ]; then
        handle_error "Insufficient CPU cores: $cpu (minimum $min_cpu required)"
    fi
    
    # OS version
    local min_os="18.04"
    local os_version=$(lsb_release -rs)
    if [[ "$os_version" < "$min_os" ]]; then
        handle_error "Unsupported OS version: $os_version (minimum $min_os required)"
    fi
    
    echo "✅ System requirements met" | tee -a $AUDIT_LOG
}

# Call this function early in the script
check_system_requirements

# Add after system info collection
log_message "=== Running System Performance Benchmark ==="
run_script "system_benchmark.sh"
log_message "System Performance Benchmark Completed"

# Add after system info collection
log_message "=== Running Configuration Drift Detection ==="
run_script "config_drift.sh"
log_message "Configuration Drift Detection Completed"

# Add after system performance benchmark
log_message "=== Running System Performance Tuning ==="
run_script "performance_tuning.sh"
log_message "System Performance Tuning Completed"

# 1. System Cleanup Check
log_message "=== Running System Cleanup Check ==="
( monitor_resources 60 & )
execution_time=$(time_script run_script "system_cleanup.sh")
log_message "System Cleanup Check Completed in ${execution_time} seconds"

# 2. System Health Check
log_message "=== Running System Health Check ==="
run_script "system_health.sh"
log_message "System Health Check Completed"

# Add after system health check
log_message "=== Running File System Check ==="
run_script "filesystem_check.sh"
log_message "File System Check Completed"

# Add after system health check
log_message "=== Running Database Health Check ==="
run_script "database_check.sh"
log_message "Database Health Check Completed"

# Add after system health check
log_message "=== Running Web Server Check ==="
run_script "webserver_check.sh"
log_message "Web Server Check Completed"

# Add after system health check
log_message "=== Running Hardware Health Check ==="
run_script "hardware_health.sh"
log_message "Hardware Health Check Completed"

# Add after system health check
log_message "=== Running Service Dependency Check ==="
run_script "service_dependency.sh"
log_message "Service Dependency Check Completed"

# Add after system health check
log_message "=== Running Container and Virtualization Check ==="
run_script "container_virtualization.sh"
log_message "Container and Virtualization Check Completed"

# Add after system health check
log_message "=== Running System Time Synchronization Check ==="
run_script "time_sync_check.sh"
log_message "System Time Synchronization Check Completed"

# 3. Backup Verification
log_message "=== Running Backup Verification ==="
run_script "backup_verification.sh"
log_message "Backup Verification Completed"

# 4. Update Status Check
log_message "=== Running System Update Check ==="
run_script "system_update.sh"
log_message "System Update Check Completed"

# Add after system update check
log_message "=== Running System Package Update Check ==="
run_script "package_update_check.sh"
log_message "System Package Update Check Completed"

# 5. Log Analysis
log_message "=== Running Log Analysis ==="
run_script "log_analysis.sh"
log_message "Log Analysis Completed"

# Add after log analysis
log_message "=== Running System Log Analysis ==="
run_script "log_analysis.sh"
log_message "System Log Analysis Completed"

# 6. System Monitoring
log_message "=== Running System Monitoring ==="
run_script "system_monitor.sh"
log_message "System Monitoring Completed"

# 7. Log Rotation Verification
log_message "=== Running Log Rotation Check ==="
run_script "log_rotation.sh"
log_message "Log Rotation Check Completed"

# Add after log rotation verification
log_message "=== Running System Log Rotation Check ==="
run_script "log_rotation_check.sh"
log_message "System Log Rotation Check Completed"

# 8. Resource Reporting
log_message "=== Running Resource Report ==="
run_script "resource_report.sh"
log_message "Resource Report Completed"

# Add after resource reporting
log_message "=== Running System Resource Limits Check ==="
run_script "resource_limits.sh"
log_message "System Resource Limits Check Completed"

# 9. Uptime Reporting
log_message "=== Running Uptime Report ==="
run_script "uptime_report.sh"
log_message "Uptime Report Completed"


log_message "=== Running Package Integrity Check ==="
run_script "package_integrity.sh"
log_message "Package Integrity Check Completed"

log_message "=== Running User Account Audit ==="
run_script "user_audit.sh"
log_message "User Account Audit Completed"


log_message "=== Running Network Performance Check ==="
run_script "network_performance.sh"
log_message "Network Performance Check Completed"

log_message "=== Running Cron Job Audit ==="
run_script "cron_audit.sh"
log_message "Cron Job Audit Completed"


log_message "=== Running Kernel Parameters Check ==="
run_script "kernel_parameters.sh"
log_message "Kernel Parameters Check Completed"

log_message "=== Running SSL/TLS Configuration Check ==="
run_script "ssl_tls_check.sh"
log_message "SSL/TLS Configuration Check Completed"

# Add after log analysis
log_message "=== Running Log File Integrity Check ==="
run_script "log_integrity.sh"
log_message "Log File Integrity Check Completed"

log_message "=== Running Vulnerability Scan ==="
run_script "vulner