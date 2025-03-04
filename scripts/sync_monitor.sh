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

while true; do
    # Check if changes exist that need to be synced
    if [ "$SERVER_TYPE" == "code" ]; then
        # On code server, check for local Git changes
        cd "$PROJECT_ROOT"
        if [ -d ".git" ]; then
            CHANGES=$(git status --porcelain | wc -l)
            if [ "$CHANGES" -gt 0 ]; then
                log "Detected $CHANGES Git changes, preparing to sync"
                # Let the git hooks handle this when committed
            fi
        fi
    elif [ "$SERVER_TYPE" == "deploy" ]; then
        # On deploy server, check for modified files to sync back
        cd "$PROJECT_ROOT"
        CHANGES=$(find . -type f -not -path "*/\.*" -not -path "*/venv/*" -not -path "*/logs/*" -mmin -$SYNC_INTERVAL | wc -l)
        if [ "$CHANGES" -gt 0 ]; then
            log "Detected $CHANGES modified files in the last $(($SYNC_INTERVAL/60)) minutes"
            log "Executing sync back to code server"
            $PROJECT_ROOT/scripts/two_way_sync.sh $SYNC_DIRECTION
        fi
    fi
    
    sleep $SYNC_INTERVAL
done 