#!/bin/bash

# Source configuration
source /opt/sutazaiapp/scripts/config/sync_config.sh

# Set up logging
LOG_FILE="$PROJECT_ROOT/logs/sync_monitor.log"

# Ensure log directory exists
mkdir -p "$(dirname $LOG_FILE)"

log() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $1" >> "$LOG_FILE"
}

log "Starting SutazAI sync monitor"

# Set process niceness to reduce CPU priority
renice 19 -p $$ > /dev/null 2>&1
ionice -c 3 -p $$ > /dev/null 2>&1

# Ensure we're only running one instance of the sync monitor
SCRIPT_NAME=$(basename "$0")
if pgrep -f "$SCRIPT_NAME" | grep -v $$ > /dev/null; then
    log "Another instance of sync monitor is already running. Exiting."
    exit 0
fi

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

# Check resource usage function
check_resources() {
    CPU_USAGE=$(awk '{u=$2+$4; t=$2+$4+$5; if (NR==1){u1=u; t1=t;} else print ($2+$4)*100/(t-t1);}' \
        <(grep 'cpu ' /proc/stat) <(sleep 0.1 && grep 'cpu ' /proc/stat)) 2>/dev/null
    
    if (( $(echo "$CPU_USAGE > 80" | bc -l 2>/dev/null) )); then
        log "WARNING: High CPU usage detected ($CPU_USAGE%). Extending sync interval."
        # Double the sync interval when CPU is high
        CURRENT_SYNC_INTERVAL=$((SYNC_INTERVAL * 2))
        return 1
    else
        CURRENT_SYNC_INTERVAL=$SYNC_INTERVAL
        return 0
    fi
}

while true; do
    # Check system resources before performing operations
    check_resources
    HIGH_LOAD=$?
    
    # Skip intensive operations if load is high
    if [ $HIGH_LOAD -eq 1 ]; then
        log "System load is high. Skipping file scan operations."
        sleep $CURRENT_SYNC_INTERVAL
        continue
    fi
    
    # Check if changes exist that need to be synced
    if [ "$SERVER_TYPE" == "code" ]; then
        # On code server, check for local Git changes
        cd "$PROJECT_ROOT"
        if [ -d ".git" ]; then
            # Use a more optimized approach to check for changes
            CHANGES=$(timeout 10s git status --porcelain 2>/dev/null | wc -l)
            if [ "$?" -ne 0 ] || [ -z "$CHANGES" ]; then
                log "Git status check failed or timed out. Skipping sync."
                CHANGES=0
            fi
            
            if [ "$CHANGES" -gt 0 ]; then
                log "Detected $CHANGES Git changes, preparing to sync"
                # Let the git hooks handle this when committed
            fi
        fi
    elif [ "$SERVER_TYPE" == "deploy" ]; then
        # On deploy server, use a more efficient way to check for modified files
        cd "$PROJECT_ROOT"
        # Use find with -newer flag instead of -mmin for better performance
        TIMESTAMP_FILE="/tmp/sync_timestamp"
        
        if [ ! -f "$TIMESTAMP_FILE" ]; then
            touch "$TIMESTAMP_FILE"
        else
            # Use a more efficient find command
            CHANGES=$(timeout 30s find . -type f -not -path "*/\.*" -not -path "*/venv/*" -not -path "*/logs/*" -newer "$TIMESTAMP_FILE" -print -quit | wc -l)
            
            if [ "$?" -ne 0 ]; then
                log "Find command timed out. Skipping sync check."
                CHANGES=0
            fi
            
            if [ "$CHANGES" -gt 0 ]; then
                log "Detected recent file modifications"
                log "Executing sync back to code server"
                nice -n 19 $PROJECT_ROOT/scripts/two_way_sync.sh $SYNC_DIRECTION
            fi
            
            # Update timestamp file
            touch "$TIMESTAMP_FILE"
        fi
    fi
    
    # Sleep with a more efficient approach - use sleep directly
    sleep $CURRENT_SYNC_INTERVAL
done 