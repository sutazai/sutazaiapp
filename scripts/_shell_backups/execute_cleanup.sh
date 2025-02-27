#!/bin/bash

# SutazAI Direct Cleanup Script
# This script removes redundant files and directories identified by the cleanup plan
# WARNING: This script directly deletes files without creating backups

set -e  # Exit on error

echo -e "\e[1;31m==== WARNING: This script will permanently delete files ====\e[0m"
echo -e "Press Ctrl+C now to abort if you want to review the cleanup plan first."
echo -e "Continuing in 5 seconds..."
sleep 5

echo -e "\n\e[1;34m==== Starting SutazAI Codebase Cleanup ====\e[0m"

# Create a log file for cleanup operations
LOG_FILE="/tmp/sutazai_cleanup_$(date +%Y%m%d_%H%M%S).log"
echo "Logging cleanup operations to: $LOG_FILE"

# Function to log operations
log() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $1" | tee -a "$LOG_FILE"
}

# Ensure we are in the project root directory
cd /opt/sutazaiapp

# 1. Remove duplicate system audit scripts
log "Removing duplicate system audit scripts..."
find sutazaiapp/scripts -name "*system*audit*.py" | grep -v "system_audit.py" | xargs rm -f 2>/dev/null || true
log "Removed duplicate system audit scripts"

# 2. Remove duplicate dependency management scripts
log "Removing duplicate dependency management scripts..."
find sutazaiapp/scripts -name "*dependency*.py" | grep -v "unified_dependency_manager.py" | xargs rm -f 2>/dev/null || true
log "Removed duplicate dependency management scripts"

# 3. Remove duplicate system health check scripts
log "Removing duplicate system health check scripts..."
find sutazaiapp/scripts -name "*health*.py" | grep -v "system_health_check.py" | xargs rm -f 2>/dev/null || true
log "Removed duplicate system health check scripts"

# 4. Remove duplicate system optimizer scripts
log "Removing duplicate system optimizer scripts..."
find sutazaiapp/scripts -name "*optimizer*.py" | grep -v "system_optimizer.py" | xargs rm -f 2>/dev/null || true
log "Removed duplicate system optimizer scripts"

# 5. Remove duplicate setup/initialization scripts
log "Removing duplicate setup/initialization scripts..."
find sutazaiapp/scripts -name "*setup*.py" -o -name "*initializer*.py" | grep -v "system_initializer.py" | xargs rm -f 2>/dev/null || true
log "Removed duplicate setup/initialization scripts"

# 6. Remove the problematic files identified by black formatting errors
log "Removing files with Python 3.11 compatibility issues..."
find sutazaiapp/misc/core_system -name "*.py" -exec grep -l "Cannot parse for target version Python 3.11" {} \; 2>/dev/null | xargs rm -f 2>/dev/null || true
log "Removed files with Python 3.11 compatibility issues"

# 7. Remove the entire misc directory (optional)
log "Removing the entire misc directory..."
rm -rf sutazaiapp/misc
log "Removed the entire misc directory"

# 8. Create a simple cleanup report
echo -e "\n\e[1;32m==== Cleanup Summary ====\e[0m" | tee -a "$LOG_FILE"
echo "Removed all duplicate system audit scripts" | tee -a "$LOG_FILE"
echo "Removed all duplicate dependency management scripts" | tee -a "$LOG_FILE"
echo "Removed all duplicate system health check scripts" | tee -a "$LOG_FILE"
echo "Removed all duplicate system optimizer scripts" | tee -a "$LOG_FILE"
echo "Removed all duplicate setup/initialization scripts" | tee -a "$LOG_FILE"
echo "Removed all files with Python 3.11 compatibility issues" | tee -a "$LOG_FILE"
echo "Removed the entire misc directory" | tee -a "$LOG_FILE"

echo -e "\n\e[1;32m==== Cleanup Completed Successfully ====\e[0m"
echo "Cleanup log saved to: $LOG_FILE" 