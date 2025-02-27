#!/bin/bash

# Sutazaiapp Server Synchronization Script
# Manages SSH key setup and server synchronization

set -euo pipefail

# Configuration
CODE_SERVER="192.168.100.28"
DEPLOY_SERVER="192.168.100.100"
SUTAZAIAPP_USER="sutazaiapp_dev"
SUTAZAIAPP_HOME="/opt/sutazaiapp"
SSH_DIR="$SUTAZAIAPP_HOME/.ssh"
SYNC_LOG="/var/log/sutazaiapp_sync.log"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SYNC_LOG"
}

# Error handling function
handle_error() {
    log "ERROR: An error occurred on line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Validate SSH key exists
validate_ssh_key() {
    local key_path="$SSH_DIR/sutazaiapp_sync_key"
    
    if [[ ! -f "$key_path" ]]; then
        log "Generating new SSH synchronization key"
        ssh-keygen -t ed25519 \
            -f "$key_path" \
            -N '' \
            -C "sutazaiapp_sync_key"
        
        chmod 600 "$key_path"
        chmod 644 "$key_path.pub"
    fi
}

# Configure SSH config for automated sync
configure_ssh_config() {
    local ssh_config="$SSH_DIR/config"
    
    log "Configuring SSH config for automated sync"
    cat > "$ssh_config" << EOL
Host deploy-server
    HostName $DEPLOY_SERVER
    User $SUTAZAIAPP_USER
    IdentityFile $SSH_DIR/sutazaiapp_sync_key
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

Host code-server
    HostName $CODE_SERVER
    User $SUTAZAIAPP_USER
    IdentityFile $SSH_DIR/sutazaiapp_sync_key
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOL

    chmod 600 "$ssh_config"
}

# Distribute SSH public key to target server
distribute_ssh_key() {
    local target_server="$1"
    local key_path="$SSH_DIR/sutazaiapp_sync_key.pub"
    
    log "Distributing SSH key to $target_server"
    ssh-copy-id -i "$key_path" "$SUTAZAIAPP_USER@$target_server"
}

# Test SSH connection
test_ssh_connection() {
    local target_server="$1"
    
    log "Testing SSH connection to $target_server"
    if ssh -q "$SUTAZAIAPP_USER@$target_server" exit; then
        log "SSH connection to $target_server successful"
    else
        log "ERROR: SSH connection to $target_server failed"
        return 1
    fi
}

# Main execution
main() {
    log "Starting Sutazaiapp Server Synchronization"
    
    # Ensure SSH directory exists with correct permissions
    mkdir -p "$SSH_DIR"
    chmod 700 "$SSH_DIR"
    
    # Generate or validate SSH key
    validate_ssh_key
    
    # Configure SSH config
    configure_ssh_config
    
    # Distribute key to deployment server
    distribute_ssh_key "$DEPLOY_SERVER"
    
    # Test SSH connections
    test_ssh_connection "$DEPLOY_SERVER"
    
    log "Server Synchronization Setup Complete"
}

main 