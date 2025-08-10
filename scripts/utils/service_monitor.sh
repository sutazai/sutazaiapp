#!/bin/bash
#
# service_monitor.sh - Monitors SutazAI services and restarts them if memory usage is too high
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
MEMORY_THRESHOLD="${SERVICE_MEMORY_THRESHOLD:-90}"  # percentage
SERVICES=(
    "sutazai-api"
    "sutazai-vector-db"
    "sutazai-webui"
)

# Create log directory for this script
SCRIPT_LOG_DIR="$LOG_DIR/monitoring"
mkdir -p "$SCRIPT_LOG_DIR"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$SCRIPT_LOG_DIR/service_monitor_$TIMESTAMP.log"

# Function to log messages
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

log "Starting service monitoring..."

# Check service memory usage
for service in "${SERVICES[@]}"; do
    if ! systemctl is-active --quiet "$service"; then
        log "$service is not running, skipping"
        continue
    fi
    
    # Get memory usage for service
    SERVICE_MEMORY=$(systemctl status "$service" | grep -E 'Memory:' | awk '{print $2}' | sed 's/%//')
    
    if [ -z "$SERVICE_MEMORY" ]; then
        log "Could not determine memory usage for $service, skipping"
        continue
    fi
    
    log "$service memory usage: ${SERVICE_MEMORY}%"
    
    # If memory usage is above threshold, restart service
    if [ "$SERVICE_MEMORY" -gt "$MEMORY_THRESHOLD" ]; then
        log "Memory usage for $service is above threshold (${SERVICE_MEMORY}% > ${MEMORY_THRESHOLD}%), restarting service"
        systemctl restart "$service"
        log "$service restarted"
    fi
done

# Check overall system memory
MEMORY_USAGE=$(free | grep Mem | awk '{print int($3/$2 * 100.0)}')
log "Overall system memory usage: ${MEMORY_USAGE}%"

# If overall memory is too high, run memory optimizer
if [ "$MEMORY_USAGE" -gt "$MEMORY_THRESHOLD" ]; then
    log "Overall memory usage is above threshold (${MEMORY_USAGE}% > ${MEMORY_THRESHOLD}%), running memory optimizer"
    /opt/sutazaiapp/scripts/memory_optimizer.sh
fi

log "Service monitoring completed"

exit 0 