#!/bin/bash

# System Dependency Check Script

log_message "=== Starting System Dependency Check ==="

# Check for missing dependencies
log_message "Missing Dependencies:"
if command -v apt &> /dev/null; then
    apt-get check | while read -r line; do
        log_message "$line"
    done
elif command -v yum &> /dev/null; then
    yum check | while read -r line; do
        log_message "$line"
    done
else
    log_message "WARNING: Package manager not found, skipping dependency check"
fi

log_message "=== System Dependency Check Completed ==="