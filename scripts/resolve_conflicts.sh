#!/bin/bash

# Conflict resolution script for two-way sync
# This script will be called by the main sync script when a conflict is detected

# Source configuration
source /opt/sutazaiapp/scripts/config/sync_config.sh

# Define log file
LOG_FILE="$PROJECT_ROOT/logs/sync/conflicts_$(date +%Y%m%d%H%M%S).log"
CURRENT_DATE=$(date +%Y-%m-%d\ %H:%M:%S)

# Command-line arguments
FILE_PATH=""
SOURCE_FILE=""
DEST_FILE=""
CONFLICT_TYPE=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --file)
            FILE_PATH="$2"
            shift 2
            ;;
        --source)
            SOURCE_FILE="$2"
            shift 2
            ;;
        --dest)
            DEST_FILE="$2"
            shift 2
            ;;
        --type)
            CONFLICT_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --file FILE_PATH --source SOURCE_FILE --dest DEST_FILE --type CONFLICT_TYPE"
            exit 1
            ;;
    esac
done

# Ensure log directory exists
mkdir -p "$(dirname $LOG_FILE)"

# Configure logging
exec > >(tee -a "$LOG_FILE") 2>&1

# Function for error handling
handle_error() {
    echo "[$CURRENT_DATE] ERROR: $1"
    exit 1
}

# Check if required arguments are provided
if [[ -z "$FILE_PATH" || -z "$SOURCE_FILE" || -z "$DEST_FILE" || -z "$CONFLICT_TYPE" ]]; then
    handle_error "Missing required arguments"
fi

echo "[$CURRENT_DATE] Resolving conflict for file: $FILE_PATH"
echo "[$CURRENT_DATE] Source file: $SOURCE_FILE"
echo "[$CURRENT_DATE] Destination file: $DEST_FILE"
echo "[$CURRENT_DATE] Conflict type: $CONFLICT_TYPE"

# Determine conflict resolution strategy
if [[ "$CONFLICT_RESOLUTION" == "newer" ]]; then
    echo "[$CURRENT_DATE] Using 'newer' conflict resolution strategy"
    
    # Get modification times
    SOURCE_MTIME=$(stat -c %Y "$SOURCE_FILE")
    DEST_MTIME=$(stat -c %Y "$DEST_FILE")
    
    if [[ $SOURCE_MTIME -gt $DEST_MTIME ]]; then
        echo "[$CURRENT_DATE] Source file is newer, using source version"
        cp "$SOURCE_FILE" "$FILE_PATH"
    else
        echo "[$CURRENT_DATE] Destination file is newer, using destination version"
        cp "$DEST_FILE" "$FILE_PATH"
    fi
elif [[ "$CONFLICT_RESOLUTION" == "code-server" ]]; then
    echo "[$CURRENT_DATE] Using 'code-server' conflict resolution strategy"
    
    # Determine which file is from the code server
    CURRENT_IP=$(hostname -I | awk '{print $1}')
    if [[ "$CURRENT_IP" == "$CODE_SERVER" ]]; then
        echo "[$CURRENT_DATE] Running on code server, using source version"
        cp "$SOURCE_FILE" "$FILE_PATH"
    else
        echo "[$CURRENT_DATE] Running on deploy server, using destination version"
        cp "$DEST_FILE" "$FILE_PATH"
    fi
elif [[ "$CONFLICT_RESOLUTION" == "deploy-server" ]]; then
    echo "[$CURRENT_DATE] Using 'deploy-server' conflict resolution strategy"
    
    # Determine which file is from the deploy server
    CURRENT_IP=$(hostname -I | awk '{print $1}')
    if [[ "$CURRENT_IP" == "$DEPLOY_SERVER" ]]; then
        echo "[$CURRENT_DATE] Running on deploy server, using source version"
        cp "$SOURCE_FILE" "$FILE_PATH"
    else
        echo "[$CURRENT_DATE] Running on code server, using destination version"
        cp "$DEST_FILE" "$FILE_PATH"
    fi
else
    handle_error "Unknown conflict resolution strategy: $CONFLICT_RESOLUTION"
fi

echo "[$CURRENT_DATE] Conflict resolution completed for file: $FILE_PATH" 