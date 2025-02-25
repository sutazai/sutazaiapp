#!/bin/bash

# System Information Collection Script

log_message "=== Collecting System Information ==="

# Basic system info
log_message "Hostname: $(hostname)"
log_message "OS: $(lsb_release -d | cut -f2)"
log_message "Kernel: $(uname -r)"
log_message "Uptime: $(uptime -p)"

# CPU information
log_message "CPU: $(grep -m 1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs)"
log_message "CPU Cores: $(grep -c ^processor /proc/cpuinfo)"

# Memory information
log_message "Total Memory: $(free -h | awk '/Mem:/ {print $2}')"
log_message "Used Memory: $(free -h | awk '/Mem:/ {print $3}')"

# Disk information
log_message "Disk Usage:"
df -h | grep -v tmpfs | while read -r line; do
    log_message "$line"
done

# Network information
log_message "Network Interfaces:"
ip -br addr show | while read -r line; do
    log_message "$line"
done

log_message "=== System Information Collection Completed ===" 