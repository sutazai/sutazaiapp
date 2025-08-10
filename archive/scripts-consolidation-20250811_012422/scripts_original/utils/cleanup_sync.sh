#!/bin/bash

# Strict error handling
set -euo pipefail

# Comprehensive cleanup script for SutazAI synchronization system

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
LOG_FILE="$PROJECT_ROOT/logs/cleanup.log"
TIMESTAMP=$(date +%Y%m%d%H%M%S)

# Ensure log directory exists
mkdir -p "$(dirname $LOG_FILE)"

# Configure logging
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $1"
}

# Clean up old log files
cleanup_logs() {
    log "Cleaning up old log files"
    
    # Find log files older than 30 days
    OLD_LOGS=$(find "$PROJECT_ROOT/logs/sync" -name "*.log" -type f -mtime +30 2>/dev/null)
    
    if [ -n "$OLD_LOGS" ]; then
        echo "$OLD_LOGS" | while read LOG; do
            log "Removing old log file: $LOG"
            rm "$LOG"
        done
    else
        log "No old log files found"
    fi
}

# Remove duplicate or redundant processes
kill_redundant_processes() {
    log "Checking for redundant processes"
    
    # Check for multiple sync_monitor.sh processes
    MONITOR_PIDS=$(pgrep -f "sync_monitor.sh" | wc -l)
    
    if [ "$MONITOR_PIDS" -gt 1 ]; then
        log "Found $MONITOR_PIDS sync_monitor.sh processes, cleaning up"
        pkill -f "sync_monitor.sh"
        sleep 2
        systemctl restart sutazai-sync-monitor.service
        log "Restarted sync monitor service"
    fi
}

# Fix permissions
fix_permissions() {
    log "Fixing permissions"
    
    # Ensure scripts are executable
    find "$PROJECT_ROOT/scripts" -name "*.sh" -type f -exec chmod +x {} \;
    
    # Ensure log directories are writable
    chmod -R 755 "$PROJECT_ROOT/logs"
    
    # Secure SSH keys
    chmod 600 "$SSH_DIR/id_ed25519" 2>/dev/null || true
    chmod 644 "$SSH_DIR/id_ed25519.pub" 2>/dev/null || true
    chmod 600 "$SSH_DIR/config" 2>/dev/null || true
}

# Validate configuration
validate_config() {
    log "Validating configuration"
    
    # Check essential config variables
    if [ -z "$CODE_SERVER" ] || [ -z "$DEPLOY_SERVER" ] || [ -z "$PROJECT_ROOT" ]; then
        log "ERROR: Missing essential configuration variables"
        return 1
    fi
    
    # Get current server IP
    CURRENT_IP=$(hostname -I | awk '{print $1}')
    
    # Test SSH connectivity to deploy server (skip if we're on that server)
    if [[ "$CURRENT_IP" != "$DEPLOY_SERVER" ]]; then
        if ! ssh -o BatchMode=yes -o ConnectTimeout=5 root@$DEPLOY_SERVER "echo test" >/dev/null 2>&1; then
            log "WARNING: Cannot connect to deploy server ($DEPLOY_SERVER) via SSH"
        fi
    else
        log "Running on deploy server, skipping self-SSH check"
    fi
    
    # Test SSH connectivity to code server (skip if we're on that server)
    if [[ "$CURRENT_IP" != "$CODE_SERVER" ]]; then
        if ! ssh -o BatchMode=yes -o ConnectTimeout=5 root@$CODE_SERVER "echo test" >/dev/null 2>&1; then
            log "WARNING: Cannot connect to code server ($CODE_SERVER) via SSH"
        fi
    else
        log "Running on code server, skipping self-SSH check"
    fi
    
    log "Configuration validation completed"
}

# Optimize performance
optimize_performance() {
    log "Optimizing system performance"
    
    # Clear any stale SSH control sockets
    find ~/.ssh -name "control-*" -type s -delete
    
    # Optimize SSH config file if it doesn't have the optimized settings
    SSH_CONFIG="$SSH_DIR/config"
    if [ ! -f "$SSH_CONFIG" ] || ! grep -q "ControlMaster" "$SSH_CONFIG"; then
        log "Creating optimized SSH config"
        
        cat > "$SSH_CONFIG" << EOC
Host $DEPLOY_SERVER
    HostName $DEPLOY_SERVER
    User root
    IdentityFile $SSH_DIR/id_ed25519
    Compression yes
    ControlMaster auto
    ControlPath ~/.ssh/control-%r@%h:%p
    ControlPersist 10m
    ServerAliveInterval 60
    ServerAliveCountMax 3

Host $CODE_SERVER
    HostName $CODE_SERVER
    User root
    IdentityFile $SSH_DIR/id_ed25519
    Compression yes
    ControlMaster auto
    ControlPath ~/.ssh/control-%r@%h:%p
    ControlPersist 10m
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOC
        chmod 600 "$SSH_CONFIG"
    fi
    
    log "Performance optimization completed"
}

# Fix system services
fix_system_services() {
    log "Fixing system services"
    
    # Reset failed systemd services
    if command -v systemctl >/dev/null 2>&1; then
        log "Checking for failed systemd services"
        FAILED_SERVICES=$(systemctl --failed | grep -c "failed")
        if [ "$FAILED_SERVICES" -gt 0 ]; then
            log "Found $FAILED_SERVICES failed systemd services, resetting"
            systemctl reset-failed
            log "Reset failed systemd services"
        else
            log "No failed systemd services found"
        fi
        
        # Reload daemon to ensure all service files are recognized
        log "Reloading systemd daemon"
        systemctl daemon-reload
        log "Systemd daemon reloaded"
    else
        log "systemctl not found, skipping systemd service fixes"
    fi
    
    # Ensure correct permissions for log directory
    if [ -d "$PROJECT_ROOT/logs" ]; then
        log "Setting correct permissions for logs directory"
        chmod -R 777 "$PROJECT_ROOT/logs"
        log "Log directory permissions fixed"
    fi
    
    log "System services fixed"
}

# Main execution
log "Starting comprehensive system cleanup"

kill_redundant_processes
cleanup_logs
fix_permissions
validate_config
optimize_performance
fix_system_services

log "Comprehensive system cleanup completed successfully"