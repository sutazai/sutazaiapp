#!/bin/bash

# Strict error handling
set -euo pipefail

# Git sync helper script for SutazAI - handles uncommitted changes

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

# Log file
LOG_FILE="$PROJECT_ROOT/logs/git_sync.log"

# Ensure log directory exists
mkdir -p "$(dirname $LOG_FILE)"

# Configure logging
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $1"
}

log "Starting Git sync helper"

# Check if we're on the code server
CURRENT_IP=$(hostname -I | awk '{print $1}')
if [[ "$CURRENT_IP" != "$CODE_SERVER" ]]; then
    log "ERROR: This script should only be run on the code server"
    exit 1
fi

# Function to display uncommitted changes
display_changes() {
    log "Checking for uncommitted changes"
    cd "$PROJECT_ROOT"
    
    if [ ! -d ".git" ]; then
        log "ERROR: No Git repository found at $PROJECT_ROOT"
        exit 1
    fi
    
    UNCOMMITTED=$(git status --porcelain)
    CHANGE_COUNT=$(echo "$UNCOMMITTED" | wc -l)
    
    if [ -z "$UNCOMMITTED" ]; then
        log "No uncommitted changes found"
        return 0
    fi
    
    log "Found $CHANGE_COUNT uncommitted changes:"
    log "$UNCOMMITTED"
}

# Function to create a temporary commit
temp_commit() {
    log "Creating temporary commit for synchronization"
    cd "$PROJECT_ROOT"
    
    # Stage all changes
    git add -A
    
    # Create temporary commit
    TEMP_COMMIT_MSG="TEMP SYNC COMMIT: $(date +%Y-%m-%d\ %H:%M:%S)"
    git commit -m "$TEMP_COMMIT_MSG"
    
    log "Created temporary commit: $TEMP_COMMIT_MSG"
}

# Function to trigger sync
trigger_sync() {
    log "Triggering sync to deploy server"
    
    "$PROJECT_ROOT/scripts/two_way_sync.sh" --to-deploy
    SYNC_RESULT=$?
    
    if [ $SYNC_RESULT -eq 0 ]; then
        log "Sync completed successfully"
    else
        log "ERROR: Sync failed with exit code $SYNC_RESULT"
    fi
    
    return $SYNC_RESULT
}

# Function to revert temporary commit
revert_temp_commit() {
    log "Reverting temporary commit"
    cd "$PROJECT_ROOT"
    
    # Get the last commit message
    LAST_COMMIT_MSG=$(git log -1 --pretty=%B)
    
    # Only revert if it was our temporary commit
    if [[ $LAST_COMMIT_MSG == TEMP\ SYNC\ COMMIT:* ]]; then
        git reset --soft HEAD~1
        log "Temporary commit reverted, changes restored to unstaged"
    else
        log "WARNING: Last commit doesn't appear to be our temporary commit, not reverting"
    fi
}

# Check command line arguments for action
ACTION="check"
if [ $# -gt 0 ]; then
    ACTION="$1"
fi

case "$ACTION" in
    "check")
        display_changes
        ;;
    "sync")
        display_changes
        temp_commit
        trigger_sync
        revert_temp_commit
        log "Git sync process completed"
        ;;
    "commit")
        display_changes
        cd "$PROJECT_ROOT"
        # Stage all changes
        git add -A
        # Create proper commit
        git commit -m "Auto-commit at $(date +%Y-%m-%d\ %H:%M:%S)"
        # The post-commit hook should handle synchronization
        log "Changes committed, post-commit hook should handle synchronization"
        ;;
    *)
        log "Unknown action: $ACTION"
        log "Usage: $0 [check|sync|commit]"
        exit 1
        ;;
esac

log "Git sync helper completed"
