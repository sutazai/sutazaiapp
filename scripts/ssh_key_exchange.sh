#!/bin/bash

# Define variables
CODE_SERVER="192.168.100.28"
DEPLOY_SERVER="192.168.100.100"
LOG_FILE="/opt/sutazaiapp/logs/ssh_setup.log"
SSH_DIR="/root/.ssh"

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

# Function to setup SSH from current server to target server
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

echo "[$(date +%Y-%m-%d\ %H:%M:%S)] SSH key exchange completed successfully." 