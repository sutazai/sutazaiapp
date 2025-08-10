#!/bin/bash
#
# memory_optimizer.sh - Script to optimize memory usage and clean up resources
#

set -e

# Source environment variables

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

if [ -f /opt/sutazaiapp/.env ]; then
    source <(grep -v '^#' /opt/sutazaiapp/.env | sed -E 's/(.*)=(.*)/export \1="\2"/')
fi

# Configuration
LOG_DIR="${LOGS_DIR:-/opt/sutazaiapp/logs}"
CACHE_DIR="${CACHE_DIR:-/opt/sutazaiapp/data/cache}"
TEMP_DIR="${TEMP_DIR:-/tmp/sutazai}"
MAX_CACHE_AGE="${MAX_CACHE_AGE:-7}"  # days
MAX_LOG_AGE="${MAX_LOG_AGE:-14}"     # days
MAX_TEMP_AGE="${MAX_TEMP_AGE:-1}"    # days
MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-85}"  # percentage

# Create log directory for this script
SCRIPT_LOG_DIR="$LOG_DIR/maintenance"
mkdir -p "$SCRIPT_LOG_DIR"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$SCRIPT_LOG_DIR/memory_optimizer_$TIMESTAMP.log"

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

log "Starting memory optimization process..."

# Check memory usage
MEMORY_USAGE=$(free | grep Mem | awk '{print int($3/$2 * 100.0)}')
log "Current memory usage: ${MEMORY_USAGE}%"

# Clean up old cache files
if [ -d "$CACHE_DIR" ]; then
    log "Cleaning up cache files older than $MAX_CACHE_AGE days..."
    find "$CACHE_DIR" -type f -mtime +$MAX_CACHE_AGE -delete 2>/dev/null || log "  No old cache files found"
fi

# Clean up old log files
log "Cleaning up log files older than $MAX_LOG_AGE days..."
find "$LOG_DIR" -type f -name "*.log" -mtime +$MAX_LOG_AGE -delete 2>/dev/null || log "  No old log files found"
find "$LOG_DIR" -type f -name "*.log.*" -mtime +$MAX_LOG_AGE -delete 2>/dev/null || log "  No old rotated log files found"

# Clean up temporary files
if [ -d "$TEMP_DIR" ]; then
    log "Cleaning up temporary files older than $MAX_TEMP_AGE days..."
    find "$TEMP_DIR" -type f -mtime +$MAX_TEMP_AGE -delete 2>/dev/null || log "  No old temporary files found"
fi

# Run additional cleanup if memory usage is above threshold
if [ "$MEMORY_USAGE" -gt "$MEMORY_THRESHOLD" ]; then
    log "Memory usage is above threshold (${MEMORY_USAGE}% > ${MEMORY_THRESHOLD}%), performing additional cleanup..."
    
    # Drop page cache if running as root
    if [ "$(id -u)" = "0" ]; then
        log "Dropping page cache..."
        echo 1 > /proc/sys/vm/drop_caches
    else
        log "Not running as root, skipping page cache drop"
    fi
    
    # Find and restart services with high memory usage
    log "Checking for high memory services..."
    HIGH_MEM_SERVICES=$(systemctl list-units --type=service --state=running | grep sutazai | awk '{print $1}' | xargs -I{} bash -c "systemctl status {} | grep -q 'Memory: [5-9][0-9]%' && echo {}" || true)
    
    if [ -n "$HIGH_MEM_SERVICES" ]; then
        log "Found services with high memory usage:"
        for service in $HIGH_MEM_SERVICES; do
            log "  Restarting $service..."
            systemctl restart "$service" || log "    Failed to restart $service"
        done
    else
        log "No services with high memory usage found"
    fi
fi

# Clean up Python bytecode
log "Cleaning up Python bytecode files..."
find /opt/sutazaiapp -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find /opt/sutazaiapp -name "*.pyc" -delete 2>/dev/null || true

# Final memory usage
MEMORY_USAGE_AFTER=$(free | grep Mem | awk '{print int($3/$2 * 100.0)}')
log "Memory usage after optimization: ${MEMORY_USAGE_AFTER}%"
log "Memory optimization completed"

# Create summary
SUMMARY="Memory Optimization Summary:
- Before: ${MEMORY_USAGE}%
- After:  ${MEMORY_USAGE_AFTER}%
- Change: $(($MEMORY_USAGE - $MEMORY_USAGE_AFTER))%
"
log "$SUMMARY"

exit 0 