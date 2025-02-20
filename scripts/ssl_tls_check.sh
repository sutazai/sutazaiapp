#!/bin/bash

# SSL/TLS Configuration Check Script

log_message "=== Starting SSL/TLS Configuration Check ==="

# Check SSL/TLS versions
if command -v openssl &> /dev/null; then
    log_message "SSL/TLS Versions:"
    openssl ciphers -v | while read -r line; do
        log_message "$line"
    done
else
    log_message "WARNING: openssl not found, skipping SSL/TLS check"
fi

log_message "=== SSL/TLS Configuration Check Completed ===" 