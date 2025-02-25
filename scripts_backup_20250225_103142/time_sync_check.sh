#!/bin/bash

# System Time Synchronization Check Script

log_message "=== Starting System Time Synchronization Check ==="

# Check NTP status
if command -v ntpq &> /dev/null; then
    log_message "NTP Status:"
    ntpq -p | while read -r line; do
        log_message "$line"
    done
else
    log_message "WARNING: ntpq not found, skipping NTP check"
fi

# Check chrony status
if command -v chronyc &> /dev/null; then
    log_message "Chrony Status:"
    chronyc tracking | while read -r line; do
        log_message "$line"
    done
else
    log_message "WARNING: chronyc not found, skipping chrony check"
fi

log_message "=== System Time Synchronization Check Completed ===" 