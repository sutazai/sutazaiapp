#!/bin/bash

# System Backup Verification Script

log_message "=== Starting Backup Verification ==="

# Check backup integrity
if [ -f "/backup/latest_backup.tar.gz" ]; then
    log_message "Verifying backup integrity..."
    tar -tzf /backup/latest_backup.tar.gz >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        log_message "Backup integrity check passed"
    else
        log_message "ERROR: Backup integrity check failed"
    fi
else
    log_message "WARNING: Backup file not found, skipping verification"
fi

log_message "=== Backup Verification Completed ===" 