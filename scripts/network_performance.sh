#!/bin/bash

# Network Performance Check Script

log_message "=== Starting Network Performance Check ==="

# Check network interface statistics
log_message "Network Interface Statistics:"
ip -s link | while read -r line; do
    log_message "$line"
done

# Check network latency
if command -v ping &> /dev/null; then
    log_message "Network Latency:"
    ping -c 4 google.com | while read -r line; do
        log_message "$line"
    done
else
    log_message "WARNING: ping command not found, skipping latency check"
fi

# Check bandwidth
if command -v iperf3 &> /dev/null; then
    log_message "Running bandwidth test..."
    iperf3 -c iperf.he.net -p 5201 -P 8 -t 10 | while read -r line; do
        log_message "$line"
    done
else
    log_message "WARNING: iperf3 not found, skipping bandwidth test"
fi

log_message "=== Network Performance Check Completed ===" 