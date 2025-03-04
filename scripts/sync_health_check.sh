#!/bin/bash

# Health check script for two-way sync system
# This script checks the health of the synchronization system and reports any issues

# Source configuration
source /opt/sutazaiapp/scripts/config/sync_config.sh

# Define log file
LOG_FILE="$PROJECT_ROOT/logs/sync/health_$(date +%Y%m%d).log"
CURRENT_DATE=$(date +%Y-%m-%d\ %H:%M:%S)

# Ensure log directory exists
mkdir -p "$(dirname $LOG_FILE)"

# Configure logging
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$CURRENT_DATE] Starting sync system health check"

# Function to check if a process is running
check_process() {
    local PROCESS_NAME="$1"
    local PID_COUNT=$(pgrep -f "$PROCESS_NAME" | wc -l)
    
    if [[ $PID_COUNT -gt 0 ]]; then
        echo "[$CURRENT_DATE] $PROCESS_NAME is running ($PID_COUNT instances)"
        return 0
    else
        echo "[$CURRENT_DATE] ERROR: $PROCESS_NAME is not running"
        return 1
    fi
}

# Function to check if a file exists and is executable
check_file() {
    local FILE_PATH="$1"
    
    if [[ -f "$FILE_PATH" ]]; then
        echo "[$CURRENT_DATE] $FILE_PATH exists"
        
        if [[ -x "$FILE_PATH" ]]; then
            echo "[$CURRENT_DATE] $FILE_PATH is executable"
            return 0
        else
            echo "[$CURRENT_DATE] ERROR: $FILE_PATH is not executable"
            return 1
        fi
    else
        echo "[$CURRENT_DATE] ERROR: $FILE_PATH does not exist"
        return 1
    fi
}

# Check if the sync monitor service is running
check_service() {
    local SERVICE_NAME="$1"
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo "[$CURRENT_DATE] $SERVICE_NAME is running"
        return 0
    else
        echo "[$CURRENT_DATE] ERROR: $SERVICE_NAME is not running"
        return 1
    fi
}

# Check if SSH keys exist and have proper permissions
check_ssh_keys() {
    local SSH_DIR="/root/.ssh"
    
    if [[ -d "$SSH_DIR" ]]; then
        echo "[$CURRENT_DATE] SSH directory exists"
        
        if [[ -f "$SSH_DIR/id_ed25519" && -f "$SSH_DIR/id_ed25519.pub" ]]; then
            echo "[$CURRENT_DATE] SSH keys exist"
            
            local SSH_DIR_PERM=$(stat -c %a "$SSH_DIR")
            local SSH_KEY_PERM=$(stat -c %a "$SSH_DIR/id_ed25519")
            
            if [[ "$SSH_DIR_PERM" == "700" && "$SSH_KEY_PERM" == "600" ]]; then
                echo "[$CURRENT_DATE] SSH keys have proper permissions"
                return 0
            else
                echo "[$CURRENT_DATE] ERROR: SSH keys have incorrect permissions (dir: $SSH_DIR_PERM, key: $SSH_KEY_PERM)"
                return 1
            fi
        else
            echo "[$CURRENT_DATE] ERROR: SSH keys do not exist"
            return 1
        fi
    else
        echo "[$CURRENT_DATE] ERROR: SSH directory does not exist"
        return 1
    fi
}

# Check SSH connectivity
check_ssh_connectivity() {
    local TARGET_SERVER="$1"
    
    echo "[$CURRENT_DATE] Checking SSH connectivity to $TARGET_SERVER"
    
    if ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@$TARGET_SERVER echo "SSH connection successful" > /dev/null 2>&1; then
        echo "[$CURRENT_DATE] SSH connection to $TARGET_SERVER successful"
        return 0
    else
        echo "[$CURRENT_DATE] ERROR: SSH connection to $TARGET_SERVER failed"
        return 1
    fi
}

# Check if cron jobs are set up
check_cron_jobs() {
    if crontab -l 2>/dev/null | grep -q "two_way_sync.sh"; then
        echo "[$CURRENT_DATE] Cron job for two_way_sync.sh exists"
        return 0
    else
        echo "[$CURRENT_DATE] ERROR: Cron job for two_way_sync.sh does not exist"
        return 1
    fi
}

# Check if Git hooks are set up (only on code server)
check_git_hooks() {
    CURRENT_IP=$(hostname -I | awk '{print $1}')
    
    if [[ "$CURRENT_IP" == "$CODE_SERVER" ]]; then
        if [[ -f "$PROJECT_ROOT/.git/hooks/post-commit" && -x "$PROJECT_ROOT/.git/hooks/post-commit" ]]; then
            echo "[$CURRENT_DATE] Git post-commit hook exists and is executable"
            return 0
        else
            echo "[$CURRENT_DATE] ERROR: Git post-commit hook does not exist or is not executable"
            return 1
        fi
    else
        echo "[$CURRENT_DATE] Not checking Git hooks (not on code server)"
        return 0
    fi
}

# Check if log files are growing too large
check_log_size() {
    local MAX_SIZE_MB=100
    
    for LOG in $(find "$PROJECT_ROOT/logs" -name "*.log" -type f); do
        local SIZE_MB=$(du -m "$LOG" | cut -f1)
        
        if [[ $SIZE_MB -gt $MAX_SIZE_MB ]]; then
            echo "[$CURRENT_DATE] WARNING: Log file $LOG is too large ($SIZE_MB MB)"
        else
            echo "[$CURRENT_DATE] Log file $LOG is within size limits ($SIZE_MB MB)"
        fi
    done
}

# Run all checks
ERRORS=0

echo "[$CURRENT_DATE] Checking script files..."
for SCRIPT in "$PROJECT_ROOT/scripts/ssh_key_exchange.sh" "$PROJECT_ROOT/scripts/two_way_sync.sh" "$PROJECT_ROOT/scripts/sync_monitor.sh" "$PROJECT_ROOT/scripts/resolve_conflicts.sh"; do
    check_file "$SCRIPT" || ((ERRORS++))
done

echo "[$CURRENT_DATE] Checking services..."
check_service "sutazai-sync-monitor" || ((ERRORS++))

echo "[$CURRENT_DATE] Checking processes..."
check_process "sync_monitor.sh" || ((ERRORS++))

echo "[$CURRENT_DATE] Checking SSH keys..."
check_ssh_keys || ((ERRORS++))

echo "[$CURRENT_DATE] Checking connectivity..."
# Determine the other server
CURRENT_IP=$(hostname -I | awk '{print $1}')
if [[ "$CURRENT_IP" == "$CODE_SERVER" ]]; then
    OTHER_SERVER="$DEPLOY_SERVER"
else
    OTHER_SERVER="$CODE_SERVER"
fi
check_ssh_connectivity "$OTHER_SERVER" || ((ERRORS++))

echo "[$CURRENT_DATE] Checking cron jobs..."
check_cron_jobs || ((ERRORS++))

echo "[$CURRENT_DATE] Checking Git hooks..."
check_git_hooks || ((ERRORS++))

echo "[$CURRENT_DATE] Checking log files..."
check_log_size

# Print summary
echo "[$CURRENT_DATE] Health check completed with $ERRORS errors"

if [[ $ERRORS -eq 0 ]]; then
    echo "[$CURRENT_DATE] Sync system is healthy"
    exit 0
else
    echo "[$CURRENT_DATE] Sync system has issues. Please check the log file: $LOG_FILE"
    exit 1
fi 