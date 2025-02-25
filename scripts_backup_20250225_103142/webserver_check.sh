#!/bin/bash

# Web Server Check Script

log_message "=== Starting Web Server Check ==="

# Check Apache
if command -v apachectl &> /dev/null; then
    log_message "Apache status:"
    apachectl status | while read -r line; do
        log_message "$line"
    done
else
    log_message "Apache not installed, skipping check"
fi

# Check Nginx
if command -v nginx &> /dev/null; then
    log_message "Nginx status:"
    nginx -t 2>&1 | while read -r line; do
        log_message "$line"
    done
else
    log_message "Nginx not installed, skipping check"
fi

log_message "=== Web Server Check Completed ===" 