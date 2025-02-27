#!/bin/bash

# Full System Health Check Script
# This script will execute all system health checks in sequence

# Set error handling
set -e

# Define timestamp
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOGFILE="/tmp/system_health_$TIMESTAMP.log"

# Function to run a health check
run_check() {
    local script_name=$1
    echo "Running $script_name..." | tee -a $LOGFILE
    if [ -f "./$script_name" ]; then
        if bash -n "./$script_name"; then
            ./$script_name >> $LOGFILE 2>&1
            echo "$script_name completed successfully." | tee -a $LOGFILE
        else
            echo "Error: $script_name contains syntax errors" | tee -a $LOGFILE
        fi
    else
        echo "Error: $script_name not found" | tee -a $LOGFILE
    fi
    echo "----------------------------------------" | tee -a $LOGFILE
}

# Start health checks
echo "Starting full system health check at $(date)" | tee $LOGFILE

# List of checks to perform
declare -a health_checks=(
    "hardware_health.sh"
    "system_audit.sh"
    "service_dependency.sh"
    "resource_limits.sh"
    "container_virtualization.sh"
    "performance_tuning.sh"
    "log_integrity.sh"
    "log_analysis.sh"
)

# Run all checks
for check in "${health_checks[@]}"; do
    run_check "$check"
done

# Final status
echo "Full system health check completed at $(date)" | tee -a $LOGFILE
echo "Log file saved to $LOGFILE" 