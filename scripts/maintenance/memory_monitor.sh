#!/bin/bash
# Memory Bank Monitoring Script with Auto-Cleanup
# Runs periodically to prevent bloat

set -euo pipefail

# Configuration
MEMORY_BANK_DIR="/opt/sutazaiapp/memory-bank"
ACTIVE_CONTEXT_FILE="${MEMORY_BANK_DIR}/activeContext.md"
MAX_SIZE_MB=1
LOG_FILE="/var/log/memory_monitor.log"
CLEANUP_SCRIPT="/opt/sutazaiapp/scripts/maintenance/memory_cleanup.py"
DEDUP_SCRIPT="/opt/sutazaiapp/scripts/maintenance/memory_deduplicator.py"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if file exists
if [ ! -f "$ACTIVE_CONTEXT_FILE" ]; then
    log "WARNING: activeContext.md not found"
    exit 0
fi

# Get file size in MB
FILE_SIZE_BYTES=$(stat -c%s "$ACTIVE_CONTEXT_FILE")
FILE_SIZE_MB=$((FILE_SIZE_BYTES / 1024 / 1024))

log "INFO: Current file size: ${FILE_SIZE_MB}MB (threshold: ${MAX_SIZE_MB}MB)"

# Check if cleanup is needed
if [ "$FILE_SIZE_MB" -gt "$MAX_SIZE_MB" ]; then
    log "WARNING: File size exceeded threshold, initiating cleanup..."
    
    # First try regular cleanup
    if python3 "$CLEANUP_SCRIPT" --cleanup --max-age 3; then
        log "INFO: Regular cleanup completed"
    else
        log "ERROR: Regular cleanup failed"
    fi
    
    # Check size again
    FILE_SIZE_BYTES=$(stat -c%s "$ACTIVE_CONTEXT_FILE")
    FILE_SIZE_MB=$((FILE_SIZE_BYTES / 1024 / 1024))
    
    # If still too large, run deduplication
    if [ "$FILE_SIZE_MB" -gt "$MAX_SIZE_MB" ]; then
        log "WARNING: File still too large, running deduplication..."
        if python3 "$DEDUP_SCRIPT"; then
            log "INFO: Deduplication completed"
        else
            log "ERROR: Deduplication failed"
        fi
    fi
    
    # Final size check
    FILE_SIZE_BYTES=$(stat -c%s "$ACTIVE_CONTEXT_FILE")
    FILE_SIZE_MB=$((FILE_SIZE_BYTES / 1024 / 1024))
    log "INFO: Final file size: ${FILE_SIZE_MB}MB"
else
    log "INFO: File size within limits, no action needed"
fi

# Check for archive files older than 30 days
if [ -d "${MEMORY_BANK_DIR}/archives" ]; then
    OLD_ARCHIVES=$(find "${MEMORY_BANK_DIR}/archives" -name "*.gz" -mtime +30 2>/dev/null | wc -l)
    if [ "$OLD_ARCHIVES" -gt 0 ]; then
        log "INFO: Found $OLD_ARCHIVES archive files older than 30 days"
        # Optionally clean up old archives
        # find "${MEMORY_BANK_DIR}/archives" -name "*.gz" -mtime +30 -delete
    fi
fi

log "INFO: Memory monitoring complete"