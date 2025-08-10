#!/bin/bash
# Purpose: Enhanced sync monitoring system with performance metrics and alerts
# Usage: ./sync_monitor.sh
# Requires: docker, system monitoring tools, mail (optional)

set -euo pipefail

# Source configuration

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

source /opt/sutazaiapp/scripts/config/sync_config.sh

# Set up logging
LOG_FILE="$PROJECT_ROOT/logs/sync_monitor.log"
METRICS_FILE="$PROJECT_ROOT/logs/metrics/system_metrics.json"

# Ensure log and metrics directories exist
mkdir -p "$(dirname $LOG_FILE)"
mkdir -p "$(dirname $METRICS_FILE)"

# Function for logging
log() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $1" | tee -a "$LOG_FILE"
}

# Function to collect and store metrics
collect_metrics() {
    # System metrics
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')
    MEM_USAGE=$(free | grep Mem | awk '{print $3/$2 * 100.0}')
    DISK_USAGE=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $5}' | tr -d '%')
    
    # Create metrics JSON
    cat > "$METRICS_FILE" << METRICS
{
    "timestamp": "$(date -Iseconds)",
    "system": {
        "cpu_usage": $CPU_USAGE,
        "memory_usage": $MEM_USAGE,
        "disk_usage": $DISK_USAGE
    },
    "sync": {
        "last_successful_sync": "$(find "$PROJECT_ROOT/logs/sync/" -name "sync_*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)",
        "error_count_24h": $(grep -l "ERROR" $(find "$PROJECT_ROOT/logs/sync/" -name "sync_*.log" -type f -mtime -1) 2>/dev/null | wc -l)
    }
}
METRICS

    log "Metrics collected and stored in $METRICS_FILE"
    
    # Check for alert conditions
    if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
        send_alert "High CPU Usage" "CPU usage is at ${CPU_USAGE}%"
    fi
    
    if (( $(echo "$MEM_USAGE > 80" | bc -l) )); then
        send_alert "High Memory Usage" "Memory usage is at ${MEM_USAGE}%"
    fi
    
    if [ "$DISK_USAGE" -gt 80 ]; then
        send_alert "High Disk Usage" "Disk usage is at ${DISK_USAGE}%"
    fi
}

# Function to send alerts
send_alert() {
    SUBJECT="$1"
    MESSAGE="$2"
    
    # Log the alert
    ALERT_LOG="$PROJECT_ROOT/logs/alerts/alert_$(date +%Y%m%d%H%M%S).log"
    mkdir -p "$(dirname "$ALERT_LOG")"
    
    cat > "$ALERT_LOG" << ALERT
Timestamp: $(date -Iseconds)
Subject: $SUBJECT
Message: $MESSAGE
ALERT

    log "ALERT: $SUBJECT - $MESSAGE"
    
    # Send email notification if enabled
    if [ "$ENABLE_EMAIL_NOTIFICATIONS" = true ]; then
        echo "$MESSAGE" | mail -s "SutazAI Sync Alert: $SUBJECT" $ADMIN_EMAIL
    fi
}

# Determine which server we're on
CURRENT_IP=$(hostname -I | awk '{print $1}')
if [[ "$CURRENT_IP" == "$CODE_SERVER" ]]; then
    SERVER_TYPE="code"
    REMOTE_SERVER="$DEPLOY_SERVER"
    SYNC_DIRECTION="--to-deploy"
elif [[ "$CURRENT_IP" == "$DEPLOY_SERVER" ]]; then
    SERVER_TYPE="deploy"
    REMOTE_SERVER="$CODE_SERVER"
    SYNC_DIRECTION="--to-code"
else
    log "ERROR: Current server IP ($CURRENT_IP) doesn't match either Code or Deployment server."
    exit 1
fi

log "Running on $SERVER_TYPE server, will sync $SYNC_DIRECTION"

# Function to check if sync is needed
check_for_changes() {
    if [ "$SERVER_TYPE" == "code" ]; then
        # On code server, check for local Git changes
        cd "$PROJECT_ROOT"
        if [ -d ".git" ]; then
            CHANGES=$(git status --porcelain | wc -l)
            if [ "$CHANGES" -gt 0 ]; then
                log "Detected $CHANGES Git changes"
                # We won't sync here - let the git hooks handle this when committed
                return 0
            fi
        fi
    elif [ "$SERVER_TYPE" == "deploy" ]; then
        # On deploy server, check for modified files to sync back
        cd "$PROJECT_ROOT"
        CHANGES=$(find . -type f -not -path "*/\.*" -not -path "*/venv/*" -not -path "*/logs/*" -mmin -$((SYNC_INTERVAL/60)) | wc -l)
        if [ "$CHANGES" -gt 0 ]; then
            log "Detected $CHANGES modified files in the last $(($SYNC_INTERVAL/60)) minutes"
            log "Executing sync back to code server"
            $PROJECT_ROOT/scripts/two_way_sync.sh $SYNC_DIRECTION
            return $?
        fi
    fi
    
    return 0
}

# Function to check remote server health
check_remote_health() {
    if ping -c 1 $REMOTE_SERVER &> /dev/null; then
        log "Remote server $REMOTE_SERVER is reachable"
        return 0
    else
        log "WARNING: Remote server $REMOTE_SERVER is not reachable"
        send_alert "Remote Server Unreachable" "Cannot ping remote server $REMOTE_SERVER"
        return 1
    fi
}

# Main monitoring loop
log "Starting SutazAI sync monitor"

# Timeout mechanism to prevent infinite loops
LOOP_TIMEOUT=${LOOP_TIMEOUT:-300}  # 5 minute default timeout
loop_start=$(date +%s)
while true; do
    # Collect system metrics
    collect_metrics
    
    # Check remote server health
    check_remote_health
    
    # Check for changes and sync if needed
    check_for_changes
    
    # Sleep until next check
    log "Sleeping for $MONITORING_INTERVAL seconds until next check"
    sleep $MONITORING_INTERVAL
    # Check for timeout
    current_time=$(date +%s)
    if [[ $((current_time - loop_start)) -gt $LOOP_TIMEOUT ]]; then
        echo 'Loop timeout reached after ${LOOP_TIMEOUT}s, exiting...' >&2
        break
    fi

done 