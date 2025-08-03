#!/bin/bash
# Purpose: Enhanced two-way sync script with optimized performance and conflict resolution
# Usage: ./two_way_sync.sh [--dry-run] [--to-code|--to-deploy] [--force]
# Requires: rsync, ssh, configured sync servers

set -euo pipefail

# Source configuration
source /opt/sutazaiapp/scripts/config/sync_config.sh

# Define log file and variables
LOG_DIR="$PROJECT_ROOT/logs/sync"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
LOG_FILE="$LOG_DIR/sync_$TIMESTAMP.log"
EXCLUDE_FILE="$PROJECT_ROOT/scripts/sync_exclude.txt"
CURRENT_DATE=$(date +%Y-%m-%d\ %H:%M:%S)

# Command line arguments
DRY_RUN=false
DIRECTION=""
FORCE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --to-code)
            DIRECTION="to-code"
            shift
            ;;
        --to-deploy)
            DIRECTION="to-deploy"
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--to-code|--to-deploy] [--force]"
            exit 1
            ;;
    esac
done

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Configure logging
exec > >(tee -a "$LOG_FILE") 2>&1

# Function for error handling
handle_error() {
    echo "[$CURRENT_DATE] ERROR: $1"
    
    # Send notification if enabled
    if [ "$ENABLE_EMAIL_NOTIFICATIONS" = true ]; then
        echo "Sync Error: $1" | mail -s "SutazAI Sync Error" $ADMIN_EMAIL
    fi
    
    exit 1
}

# Function to determine server type
determine_server() {
    CURRENT_IP=$(hostname -I | awk '{print $1}')
    if [[ "$CURRENT_IP" == "$CODE_SERVER" ]]; then
        echo "code"
    elif [[ "$CURRENT_IP" == "$DEPLOY_SERVER" ]]; then
        echo "deploy"
    else
        handle_error "Current server IP ($CURRENT_IP) doesn't match either Code or Deployment server."
    fi
}

# Function to handle conflicts based on configuration
handle_conflicts() {
    echo "[$CURRENT_DATE] Handling potential conflicts with strategy: $CONFLICT_RESOLUTION"
    
    case "$CONFLICT_RESOLUTION" in
        "newer")
            # Prefer newer files based on modification time
            RSYNC_CMD="$RSYNC_CMD --update"
            ;;
        "code-server")
            # Always prefer code server version in conflicts
            if [[ "$CURRENT_SERVER" == "code" || "$SOURCE" == "$CODE_SERVER" ]]; then
                RSYNC_CMD="$RSYNC_CMD --ignore-existing"
            fi
            ;;
        "deploy-server")
            # Always prefer deploy server version in conflicts
            if [[ "$CURRENT_SERVER" == "deploy" || "$SOURCE" == "$DEPLOY_SERVER" ]]; then
                RSYNC_CMD="$RSYNC_CMD --ignore-existing"
            fi
            ;;
        "interactive")
            # For manual conflict resolution
            RSYNC_CMD="$RSYNC_CMD --dry-run"
            # Store conflicts for manual resolution
            CONFLICT_LOG="$LOG_DIR/conflicts_$TIMESTAMP.log"
            eval $RSYNC_CMD $SOURCE_PATH root@$DESTINATION:$DEST_PATH > "$CONFLICT_LOG"
            # Send notification if conflicts detected
            if grep -q "^>" "$CONFLICT_LOG"; then
                if [ "$ENABLE_EMAIL_NOTIFICATIONS" = true ]; then
                    echo "Conflicts detected during sync. Please resolve manually." | mail -s "SutazAI Sync Conflict" $ADMIN_EMAIL
                fi
                echo "[$CURRENT_DATE] Conflicts detected. Manual resolution required."
                echo "[$CURRENT_DATE] See $CONFLICT_LOG for details"
                exit 1
            fi
            # Proceed with actual sync if no conflicts
            RSYNC_CMD=${RSYNC_CMD/--dry-run/}
            ;;
    esac
}

# Function to optimize rsync performance
optimize_rsync() {
    # Adapt to system load
    CPU_LOAD=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')
    if (( $(echo "$CPU_LOAD > 70" | bc -l) )); then
        # System is under heavy load, reduce rsync priority
        RSYNC_CMD="nice -n 19 $RSYNC_CMD"
        
        # Limit bandwidth if needed
        if [ "$MAX_BANDWIDTH" != "0" ]; then
            RSYNC_CMD="$RSYNC_CMD --bwlimit=$MAX_BANDWIDTH"
        fi
    fi
    
    # Optimize for large files if enabled
    if [ "$LARGE_FILES_OPTIMIZATION" = true ]; then
        RSYNC_CMD="$RSYNC_CMD --inplace --no-whole-file"
    fi
}

# Function to send notification
send_notification() {
    SUBJECT="$1"
    MESSAGE="$2"
    
    echo "[$CURRENT_DATE] $SUBJECT: $MESSAGE"
    
    if [ "$ENABLE_EMAIL_NOTIFICATIONS" = true ]; then
        echo "$MESSAGE" | mail -s "SutazAI Sync: $SUBJECT" $ADMIN_EMAIL
    fi
}

# Function to sync from source to destination
sync_servers() {
    local SOURCE=$1
    local DESTINATION=$2
    local SOURCE_PATH=$3
    local DEST_PATH=$4

    echo "[$CURRENT_DATE] Syncing from $SOURCE to $DESTINATION"
    
    # Build rsync command with optimized parameters
    RSYNC_CMD="rsync -avz --delete --compress-level=$COMPRESSION_LEVEL"
    
    # Add exclude file if it exists
    if [ -f "$EXCLUDE_FILE" ]; then
        RSYNC_CMD="$RSYNC_CMD --exclude-from=$EXCLUDE_FILE"
    fi
    
    # Add dry-run flag if specified
    if [ "$DRY_RUN" = true ]; then
        RSYNC_CMD="$RSYNC_CMD --dry-run"
        echo "[$CURRENT_DATE] DRY RUN - No actual changes will be made"
    fi
    
    # Handle conflicts based on configuration
    handle_conflicts
    
    # Optimize rsync based on system conditions
    optimize_rsync
    
    # Construct the complete command
    FULL_CMD="$RSYNC_CMD $SOURCE_PATH root@$DESTINATION:$DEST_PATH"
    
    echo "[$CURRENT_DATE] Executing: $FULL_CMD"
    
    # Execute with retry logic
    RETRY_COUNT=0
    while [ $RETRY_COUNT -lt $MAX_SYNC_RETRIES ]; do
        if eval $FULL_CMD; then
            echo "[$CURRENT_DATE] Sync from $SOURCE to $DESTINATION completed successfully"
            send_notification "Sync Completed" "Sync from $SOURCE to $DESTINATION completed successfully"
            return 0
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            if [ $RETRY_COUNT -lt $MAX_SYNC_RETRIES ]; then
                echo "[$CURRENT_DATE] Sync attempt $RETRY_COUNT failed. Retrying in 10 seconds..."
                sleep 10
            else
                handle_error "Sync failed from $SOURCE to $DESTINATION after $MAX_SYNC_RETRIES attempts"
            fi
        fi
    done
    
    # If we get here, all retries failed
    handle_error "Sync failed from $SOURCE to $DESTINATION after $MAX_SYNC_RETRIES attempts"
}

# Main logic
CURRENT_SERVER=$(determine_server)
echo "[$CURRENT_DATE] Running on $CURRENT_SERVER server"

if [[ -z "$DIRECTION" ]]; then
    echo "[$CURRENT_DATE] No direction specified, determining based on current server"
    if [[ "$CURRENT_SERVER" == "code" ]]; then
        DIRECTION="to-deploy"
    else
        DIRECTION="to-code"
    fi
fi

if [[ "$DIRECTION" == "to-deploy" ]]; then
    if [[ "$CURRENT_SERVER" == "code" ]]; then
        SOURCE="$CODE_SERVER"
        sync_servers "$CODE_SERVER" "$DEPLOY_SERVER" "$PROJECT_ROOT/" "$PROJECT_ROOT/"
    else
        SOURCE="localhost"
        sync_servers "localhost" "$DEPLOY_SERVER" "$PROJECT_ROOT/" "$PROJECT_ROOT/"
    fi
elif [[ "$DIRECTION" == "to-code" ]]; then
    if [[ "$CURRENT_SERVER" == "deploy" ]]; then
        SOURCE="$DEPLOY_SERVER"
        sync_servers "$DEPLOY_SERVER" "$CODE_SERVER" "$PROJECT_ROOT/" "$PROJECT_ROOT/"
    else
        SOURCE="localhost"
        sync_servers "localhost" "$CODE_SERVER" "$PROJECT_ROOT/" "$PROJECT_ROOT/"
    fi
else
    handle_error "Invalid direction: $DIRECTION"
fi

echo "[$CURRENT_DATE] Sync operation completed successfully"
