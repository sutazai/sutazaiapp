#!/bin/bash

# System Configuration Drift Detection Script

log_message "=== Starting Configuration Drift Detection ==="

# Define baseline configuration file
BASELINE="/etc/baseline_config.txt"

# Check if baseline exists
if [ ! -f "$BASELINE" ]; then
    log_message "WARNING: Baseline configuration not found, creating new baseline"
    find /etc -type f -exec md5sum {} \; > "$BASELINE"
fi

# Compare current configuration with baseline
log_message "Configuration Drift:"
find /etc -type f -exec md5sum {} \; | diff "$BASELINE" - | while read -r line; do
    log_message "$line"
done

log_message "=== Configuration Drift Detection Completed ===" 