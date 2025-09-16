#!/bin/bash
# MCP Memory Cleanup Service
# Runs periodically to prevent memory accumulation

MEMORY_LIMIT_MB=100
MEMORY_DIRS=("/opt/sutazaiapp/backend/memory-bank" "/tmp/memory" "$HOME/.memory")

for dir in "${MEMORY_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        # Get directory size in MB
        SIZE_MB=$(du -sm "$dir" 2>/dev/null | awk '{print $1}' || echo "0")
        
        if [ "$SIZE_MB" -gt "$MEMORY_LIMIT_MB" ]; then
            echo "[$(date)] Cleaning $dir (${SIZE_MB}MB > ${MEMORY_LIMIT_MB}MB limit)"
            
            # Remove old files (older than 1 hour)
            find "$dir" -type f -mmin +60 -delete 2>/dev/null || true
            
            # If still too large, remove all but recent files
            if [ "$SIZE_MB" -gt "$MEMORY_LIMIT_MB" ]; then
                find "$dir" -type f -mmin +30 -delete 2>/dev/null || true
            fi
        fi
    fi
done
