#!/bin/bash

# SutazAI Service Installation Script
# This script installs and configures systemd services for SutazAI

set -euo pipefail

# Configuration
SUTAZAIAPP_HOME="/opt/sutazaiapp"
SUTAZAIAPP_USER="sutazaiapp_dev"
SERVICE_DIR="/etc/systemd/system"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a /var/log/sutazaiapp_service_install.log
}

# Error handling function
handle_error() {
    log "ERROR: An error occurred on line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Ensure script is run as root
if [[ $EUID -ne 0 ]]; then
   log "This script must be run as root" 
   exit 1
fi

# Install monitoring service
install_monitoring_service() {
    log "Installing monitoring service..."
    
    # Copy service file
    cp "$SUTAZAIAPP_HOME/scripts/systemd/sutazai-monitor.service" "$SERVICE_DIR/"
    
    # Set permissions
    chmod 644 "$SERVICE_DIR/sutazai-monitor.service"
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable and start service
    systemctl enable sutazai-monitor
    systemctl start sutazai-monitor
    
    log "Monitoring service installed successfully"
}

# Install maintenance service and timer
install_maintenance_service() {
    log "Installing maintenance service and timer..."
    
    # Copy service and timer files
    cp "$SUTAZAIAPP_HOME/scripts/systemd/sutazai-maintenance.service" "$SERVICE_DIR/"
    cp "$SUTAZAIAPP_HOME/scripts/systemd/sutazai-maintenance.timer" "$SERVICE_DIR/"
    
    # Set permissions
    chmod 644 "$SERVICE_DIR/sutazai-maintenance.service"
    chmod 644 "$SERVICE_DIR/sutazai-maintenance.timer"
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable and start timer
    systemctl enable sutazai-maintenance.timer
    systemctl start sutazai-maintenance.timer
    
    log "Maintenance service and timer installed successfully"
}

# Configure log rotation
configure_log_rotation() {
    log "Configuring log rotation..."
    
    # Create logrotate configuration
    cat > "/etc/logrotate.d/sutazaiapp" << EOL
/opt/sutazaiapp/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 sutazaiapp_dev sutazaiapp_dev
    postrotate
        systemctl reload sutazai-monitor >/dev/null 2>&1 || true
    endscript
}
EOL
    
    # Set permissions
    chmod 644 "/etc/logrotate.d/sutazaiapp"
    
    log "Log rotation configured successfully"
}

# Configure monitoring
configure_monitoring() {
    log "Configuring monitoring..."
    
    # Create monitoring configuration directory
    mkdir -p "$SUTAZAIAPP_HOME/config"
    
    # Copy monitoring configuration
    cp "$SUTAZAIAPP_HOME/config/monitoring_config.json" "$SUTAZAIAPP_HOME/config/"
    
    # Set permissions
    chown -R "$SUTAZAIAPP_USER:$SUTAZAIAPP_USER" "$SUTAZAIAPP_HOME/config"
    chmod 644 "$SUTAZAIAPP_HOME/config/monitoring_config.json"
    
    log "Monitoring configured successfully"
}

# Main execution
main() {
    log "Starting SutazAI service installation..."
    
    install_monitoring_service
    install_maintenance_service
    configure_log_rotation
    configure_monitoring
    
    log "SutazAI service installation completed successfully"
}

main 