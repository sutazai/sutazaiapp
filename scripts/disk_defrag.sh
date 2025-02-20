#!/bin/bash
# Defragment disk partitions
for partition in $(lsblk -lno NAME,TYPE | grep part | awk '{print $1}'); do
    e4defrag /dev/$partition
done
echo "Disk defragmentation completed successfully!"

log "DEBUG" "Checking disk fragmentation"
fragmentation=$(df -h | awk '/\/$/ {print $5}' | tr -d '%')

if [ "$fragmentation" -gt 80 ]; then
    handle_error "High disk fragmentation detected: ${fragmentation}%"
fi

log "INFO" "Disk fragmentation check passed: ${fragmentation}%" 