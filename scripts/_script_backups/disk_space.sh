#!/bin/bash
# Monitor disk space and alert if low

MIN_DISK_SPACE=20 # in GB

log "DEBUG" "Checking available disk space"
available_space=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')

if [ "$available_space" -lt "$MIN_DISK_SPACE" ]; then
    handle_error "Insufficient disk space: ${available_space}GB available, ${MIN_DISK_SPACE}GB required"
fi

log "INFO" "Disk space check passed: ${available_space}GB available"

# Add function to check inode usage
check_inode_usage() {
    local inode_usage=$(df -i / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ "$inode_usage" -gt 90 ]; then
        echo "Warning: High inode usage: ${inode_usage}%"
        return 1
    fi
    
    return 0
}

DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | tr -d '%')
if (( $DISK_USAGE > 90 )); then
    ./alert.sh "Low disk space: $DISK_USAGE%"
fi 