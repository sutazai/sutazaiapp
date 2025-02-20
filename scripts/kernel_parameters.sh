#!/bin/bash

# Kernel Parameters Check Script

log_message "=== Starting Kernel Parameters Check ==="

# Check important kernel parameters
log_message "Kernel Parameters:"
sysctl -a | grep -E "net.ipv4.conf.all.rp_filter|net.ipv4.tcp_syncookies|kernel.randomize_va_space" | while read -r line; do
    log_message "$line"
done

log_message "=== Kernel Parameters Check Completed ===" 