#!/bin/bash

# Network Security Check Script

log_message "=== Starting Network Security Check ==="

# Check open ports
log_message "Open ports:"
netstat -tuln | while read -r line; do
    log_message "$line"
done

# Check firewall status
if command -v ufw &> /dev/null; then
    log_message "Firewall status:"
    ufw status | while read -r line; do
        log_message "$line"
    done
else
    log_message "WARNING: ufw not installed, skipping firewall check"
fi

log_message "=== Network Security Check Completed ===" 