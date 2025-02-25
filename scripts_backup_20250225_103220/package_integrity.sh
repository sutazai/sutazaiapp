#!/bin/bash

# Package Integrity Check Script

log_message "=== Starting Package Integrity Check ==="

if command -v debsums &> /dev/null; then
    log_message "Checking installed package integrity..."
    debsums -c 2>/dev/null | while read -r line; do
        log_message "$line"
    done
else
    log_message "WARNING: debsums not installed, skipping package integrity check"
fi

log_message "=== Package Integrity Check Completed ===" 