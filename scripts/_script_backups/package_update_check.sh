#!/bin/bash

# System Package Update Check Script

log_message "=== Starting System Package Update Check ==="

# Check for available updates
if command -v apt &> /dev/null; then
    log_message "Available Package Updates:"
    apt list --upgradable 2>/dev/null | while read -r line; do
        log_message "$line"
    done
elif command -v yum &> /dev/null; then
    log_message "Available Package Updates:"
    yum check-update | while read -r line; do
        log_message "$line"
    done
else
    log_message "WARNING: Package manager not found, skipping update check"
fi

log_message "=== System Package Update Check Completed ===" 