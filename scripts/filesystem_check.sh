#!/bin/bash

# File System Check Script

log_message "=== Starting File System Check ==="

# Check for read-only file systems
log_message "Read-only file systems:"
mount | grep "\sro[\s,]" | while read -r line; do
    log_message "$line"
done

# Check for file system errors
log_message "File system errors:"
dmesg | grep -i "error" | grep -i "filesystem" | while read -r line; do
    log_message "$line"
done

log_message "=== File System Check Completed ===" 