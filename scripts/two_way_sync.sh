#!/bin/bash

# Define variables
CODE_SERVER="192.168.100.28"
DEPLOY_SERVER="192.168.100.100"
PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="$PROJECT_ROOT/logs/sync"
LOG_FILE="$LOG_DIR/sync_$(date +%Y%m%d%H%M%S).log"
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
    exit 1
}

# Function to check if we're running on code or deployment server
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

# Function to sync from source to destination
sync_servers() {
    local SOURCE=$1
    local DESTINATION=$2
    local SOURCE_PATH=$3
    local DEST_PATH=$4

    echo "[$CURRENT_DATE] Syncing from $SOURCE to $DESTINATION"
    
    # Build rsync command
    RSYNC_CMD="rsync -avzP --delete"
    
    # Add exclude file if it exists
    if [ -f "$EXCLUDE_FILE" ]; then
        RSYNC_CMD="$RSYNC_CMD --exclude-from=$EXCLUDE_FILE"
    fi
    
    # Add dry-run flag if specified
    if [ "$DRY_RUN" = true ]; then
        RSYNC_CMD="$RSYNC_CMD --dry-run"
        echo "[$CURRENT_DATE] DRY RUN - No actual changes will be made"
    fi
    
    # Construct the complete command
    FULL_CMD="$RSYNC_CMD $SOURCE_PATH root@$DESTINATION:$DEST_PATH"
    
    echo "[$CURRENT_DATE] Executing: $FULL_CMD"
    eval $FULL_CMD || handle_error "Sync failed from $SOURCE to $DESTINATION"
    
    echo "[$CURRENT_DATE] Sync from $SOURCE to $DESTINATION completed successfully"
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
        sync_servers "$CODE_SERVER" "$DEPLOY_SERVER" "$PROJECT_ROOT/" "$PROJECT_ROOT/"
    else
        sync_servers "localhost" "$DEPLOY_SERVER" "$PROJECT_ROOT/" "$PROJECT_ROOT/"
    fi
elif [[ "$DIRECTION" == "to-code" ]]; then
    if [[ "$CURRENT_SERVER" == "deploy" ]]; then
        sync_servers "$DEPLOY_SERVER" "$CODE_SERVER" "$PROJECT_ROOT/" "$PROJECT_ROOT/"
    else
        sync_servers "localhost" "$CODE_SERVER" "$PROJECT_ROOT/" "$PROJECT_ROOT/"
    fi
else
    handle_error "Invalid direction: $DIRECTION"
fi

echo "[$CURRENT_DATE] Sync operation completed successfully" 