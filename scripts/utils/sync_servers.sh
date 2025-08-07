#!/bin/bash
# SutazAI Server Synchronization Script
# This script synchronizes files between the Code Server and Deployment Server

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
SYNC_LOG="${PROJECT_ROOT}/logs/sync.log"
mkdir -p "$(dirname "$SYNC_LOG")"

# Configuration
SSH_KEY="${HOME}/.ssh/sutazaiapp_deploy"
REMOTE_SERVER="192.168.100.100"
REMOTE_USER="root"
REMOTE_PATH="/opt/sutazaiapp"
EXCLUDE_FILE="${PROJECT_ROOT}/.syncignore"

# Logging function
log() {
    local message="$1"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "$message"
    echo "[$timestamp] $message" >> "$SYNC_LOG"
}

log "${BLUE}Starting SutazAI server synchronization...${NC}"

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    log "${RED}Error: SSH key not found at $SSH_KEY${NC}"
    log "Please run the SSH key setup commands first."
    exit 1
fi

# Create default exclude file if it doesn't exist
if [ ! -f "$EXCLUDE_FILE" ]; then
    log "${YELLOW}Creating default .syncignore file...${NC}"
    cat > "$EXCLUDE_FILE" << EOF
# SutazAI sync exclude patterns
.git/
.git*
__pycache__/
*.pyc
venv/
node_modules/
*.log
.DS_Store
.env
backups/
model_management/GPT4All/*.bin
model_management/GPT4All/*.gguf
model_management/DeepSeek-Coder-33B/*.bin
model_management/DeepSeek-Coder-33B/*.gguf
EOF
    log "${GREEN}Default .syncignore file created.${NC}"
fi

# Verify connection to the remote server
log "Verifying connection to remote server..."
if ! ssh -i "$SSH_KEY" -o ConnectTimeout=5 -o BatchMode=yes "${REMOTE_USER}@${REMOTE_SERVER}" "echo Connection successful"; then
    log "${RED}Error: Cannot connect to remote server ${REMOTE_USER}@${REMOTE_SERVER}${NC}"
    log "Please verify that:"
    log "  1. The remote server is accessible"
    log "  2. Your SSH key has been properly copied to the remote server"
    log "  3. The remote user has necessary permissions"
    exit 1
fi

# Create remote directory structure if needed
log "Creating remote directory structure..."
ssh -i "$SSH_KEY" "${REMOTE_USER}@${REMOTE_SERVER}" "mkdir -p ${REMOTE_PATH}/{logs,workspace,storage,outputs}"

# Function to sync files
sync_files() {
    local sync_type="$1"
    local args="$2"
    
    log "${BLUE}Performing ${sync_type} synchronization...${NC}"
    
    # Construct the rsync command
    rsync_cmd="rsync -avzP --delete ${args} \
        --exclude-from=\"$EXCLUDE_FILE\" \
        -e \"ssh -i $SSH_KEY\" \
        \"$PROJECT_ROOT/\" \"${REMOTE_USER}@${REMOTE_SERVER}:${REMOTE_PATH}/\""
    
    # Log the command
    log "Executing: $rsync_cmd"
    
    # Execute the command
    eval $rsync_cmd
    
    if [ $? -eq 0 ]; then
        log "${GREEN}${sync_type} synchronization completed successfully.${NC}"
    else
        log "${RED}${sync_type} synchronization failed.${NC}"
        return 1
    fi
    
    return 0
}

# Parse command line arguments
SYNC_MODE="normal"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --fast)
            SYNC_MODE="fast"
            shift
            ;;
        --full)
            SYNC_MODE="full"
            shift
            ;;
        *)
            log "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Perform synchronization based on mode
if [ "$DRY_RUN" = true ]; then
    log "Performing dry run (no changes will be made)..."
    sync_files "dry run" "--dry-run"
else
    case $SYNC_MODE in
        "fast")
            # Fast sync - only sync code files, skip large files
            sync_files "fast" "--size-only --exclude='*.bin' --exclude='*.gguf'"
            ;;
        "full")
            # Full sync - sync everything including models
            sync_files "full" ""
            ;;
        "normal")
            # Normal sync - default behavior
            sync_files "normal" ""
            ;;
    esac
fi

# Verify remote services after sync
log "${BLUE}Verifying remote services...${NC}"
ssh -i "$SSH_KEY" "${REMOTE_USER}@${REMOTE_SERVER}" "cd ${REMOTE_PATH} && chmod +x scripts/*.sh && bash scripts/health_check.sh"

log "${GREEN}Server synchronization completed!${NC}"
exit 0 