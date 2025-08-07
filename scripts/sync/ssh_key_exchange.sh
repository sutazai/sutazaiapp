#!/bin/bash
# Purpose: SSH key exchange script with enhanced configuration
# Usage: ./ssh_key_exchange.sh
# Requires: ssh-keygen, ssh-copy-id, network connectivity

set -euo pipefail

# Source configuration
source /opt/sutazaiapp/scripts/config/sync_config.sh

# Define log file
LOG_FILE="$PROJECT_ROOT/logs/ssh_setup.log"

# Ensure log directory exists
mkdir -p "$(dirname $LOG_FILE)"

# Configure logging
exec > >(tee -a $LOG_FILE) 2>&1
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Starting SSH key exchange setup"

# Function for error handling
handle_error() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ERROR: $1"
    exit 1
}

# Create SSH directory if it doesn't exist
mkdir -p $SSH_DIR
chmod 700 $SSH_DIR

# Generate SSH key if it doesn't exist
if [ ! -f "$SSH_DIR/id_ed25519" ]; then
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Generating new SSH key pair"
    ssh-keygen -t ed25519 -f "$SSH_DIR/id_ed25519" -N "" || handle_error "Failed to generate SSH key"
else
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] SSH key already exists"
fi

# Create optimized SSH config
create_ssh_config() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Creating optimized SSH config"
    
    SSH_CONFIG="$SSH_DIR/config"
    
    # Create or update SSH config with optimized settings
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
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] SSH config created with optimized settings"
}

# Function to setup SSH to a target server
setup_ssh_to_server() {
    local TARGET_SERVER=$1
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Setting up SSH to $TARGET_SERVER"
    
    # Copy SSH key to target server
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Copying SSH key to $TARGET_SERVER"
    ssh-copy-id -i "$SSH_DIR/id_ed25519.pub" -o StrictHostKeyChecking=no root@$TARGET_SERVER || handle_error "Failed to copy SSH key to $TARGET_SERVER"
    
    # Test SSH connection
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Testing SSH connection to $TARGET_SERVER"
    ssh -o BatchMode=yes -o ConnectTimeout=5 root@$TARGET_SERVER echo "SSH connection successful" || handle_error "Failed to establish SSH connection to $TARGET_SERVER"
}

# Determine current server and set up SSH to the other server
CURRENT_IP=$(hostname -I | awk '{print $1}')
if [[ "$CURRENT_IP" == "$CODE_SERVER" ]]; then
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Running on Code Server. Setting up SSH to Deployment Server."
    setup_ssh_to_server $DEPLOY_SERVER
elif [[ "$CURRENT_IP" == "$DEPLOY_SERVER" ]]; then
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Running on Deployment Server. Setting up SSH to Code Server."
    setup_ssh_to_server $CODE_SERVER
else
    handle_error "Current server IP ($CURRENT_IP) doesn't match either Code or Deployment server."
fi

# Create optimized SSH config
create_ssh_config

echo "[$(date +%Y-%m-%d\ %H:%M:%S)] SSH key exchange completed successfully." 
