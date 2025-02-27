#!/bin/bash

# Sutazaiapp User Management Script
# Provides robust user creation and management

set -euo pipefail

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a /var/log/sutazaiapp_user_management.log
}

# Error handling function
handle_error() {
    log "ERROR: An error occurred on line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Configuration
SUTAZAIAPP_USER="sutazaiapp_dev"
SUTAZAIAPP_HOME="/opt/sutazaiapp"
SSH_DIR="$SUTAZAIAPP_HOME/.ssh"

# Ensure script is run as root
if [[ $EUID -ne 0 ]]; then
   log "ERROR: This script must be run as root"
   exit 1
fi

# Create user with enhanced security
create_user() {
    log "Creating non-root user: $SUTAZAIAPP_USER"
    
    # Check if user exists
    if id "$SUTAZAIAPP_USER" &>/dev/null; then
        log "User $SUTAZAIAPP_USER already exists. Updating configuration."
        usermod -aG sudo "$SUTAZAIAPP_USER"
    else
        # Create user with specific settings
        useradd -m -s /bin/bash \
                -c "Sutazaiapp Development User" \
                -G sudo,adm,systemd-journal \
                "$SUTAZAIAPP_USER"
    fi

    # Set secure sudo permissions
    echo "$SUTAZAIAPP_USER ALL=(ALL) NOPASSWD:/usr/bin/apt,/usr/bin/pip,/usr/bin/systemctl" | \
        tee "/etc/sudoers.d/$SUTAZAIAPP_USER"
    chmod 440 "/etc/sudoers.d/$SUTAZAIAPP_USER"

    # Create SSH directory with strict permissions
    mkdir -p "$SSH_DIR"
    chmod 700 "$SSH_DIR"
    chown -R "$SUTAZAIAPP_USER:$SUTAZAIAPP_USER" "$SSH_DIR"

    log "User $SUTAZAIAPP_USER created successfully with enhanced permissions"
}

# Set up SSH key for the user
setup_ssh_key() {
    local ssh_key_path="$SSH_DIR/sutazaiapp_dev_key"
    
    if [[ ! -f "$ssh_key_path" ]]; then
        log "Generating SSH key for $SUTAZAIAPP_USER"
        su - "$SUTAZAIAPP_USER" -c "
            ssh-keygen -t ed25519 -f '$ssh_key_path' -N '' -C 'sutazaiapp_dev_key'
        "
        chmod 600 "$ssh_key_path"
        chmod 644 "$ssh_key_path.pub"
    else
        log "SSH key already exists for $SUTAZAIAPP_USER"
    fi
}

# Main execution
main() {
    log "Starting Sutazaiapp User Management"
    create_user
    setup_ssh_key
    log "Sutazaiapp User Management Complete"
}

main 