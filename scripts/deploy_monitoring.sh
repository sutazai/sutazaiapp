#!/bin/bash

# SutazAI Monitoring Deployment Script
# This script deploys monitoring and maintenance services to the deployment server

set -euo pipefail

# Configuration
SUTAZAIAPP_HOME="/opt/sutazaiapp"
SUTAZAIAPP_USER="sutazaiapp_dev"
DEPLOY_SERVER="192.168.100.100"
SSH_KEY="/home/$SUTAZAIAPP_USER/.ssh/sutazai_deploy"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a /var/log/sutazaiapp_monitoring_deploy.log
}

# Error handling function
handle_error() {
    log "ERROR: An error occurred on line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Ensure script is run as sutazaiapp_dev user
if [[ "$(whoami)" != "$SUTAZAIAPP_USER" ]]; then
   log "This script must be run as $SUTAZAIAPP_USER" 
   exit 1
fi

# Ensure SSH key exists
if [[ ! -f "$SSH_KEY" ]]; then
    log "SSH key not found: $SSH_KEY"
    exit 1
fi

# Deploy monitoring scripts
deploy_monitoring_scripts() {
    log "Deploying monitoring scripts..."
    
    # Create remote directories
    ssh -i "$SSH_KEY" "$SUTAZAIAPP_USER@$DEPLOY_SERVER" "mkdir -p $SUTAZAIAPP_HOME/scripts/systemd"
    
    # Copy monitoring script
    scp -i "$SSH_KEY" "$SUTAZAIAPP_HOME/scripts/monitoring.py" "$SUTAZAIAPP_USER@$DEPLOY_SERVER:$SUTAZAIAPP_HOME/scripts/"
    
    # Copy maintenance script
    scp -i "$SSH_KEY" "$SUTAZAIAPP_HOME/scripts/system_maintenance.py" "$SUTAZAIAPP_USER@$DEPLOY_SERVER:$SUTAZAIAPP_HOME/scripts/"
    
    # Copy systemd service files
    scp -i "$SSH_KEY" "$SUTAZAIAPP_HOME/scripts/systemd/sutazai-monitor.service" "$SUTAZAIAPP_USER@$DEPLOY_SERVER:$SUTAZAIAPP_HOME/scripts/systemd/"
    scp -i "$SSH_KEY" "$SUTAZAIAPP_HOME/scripts/systemd/sutazai-maintenance.service" "$SUTAZAIAPP_USER@$DEPLOY_SERVER:$SUTAZAIAPP_HOME/scripts/systemd/"
    scp -i "$SSH_KEY" "$SUTAZAIAPP_HOME/scripts/systemd/sutazai-maintenance.timer" "$SUTAZAIAPP_USER@$DEPLOY_SERVER:$SUTAZAIAPP_HOME/scripts/systemd/"
    
    log "Monitoring scripts deployed successfully"
}

# Deploy monitoring configuration
deploy_monitoring_config() {
    log "Deploying monitoring configuration..."
    
    # Create remote config directory
    ssh -i "$SSH_KEY" "$SUTAZAIAPP_USER@$DEPLOY_SERVER" "mkdir -p $SUTAZAIAPP_HOME/config"
    
    # Copy monitoring configuration
    scp -i "$SSH_KEY" "$SUTAZAIAPP_HOME/config/monitoring_config.json" "$SUTAZAIAPP_USER@$DEPLOY_SERVER:$SUTAZAIAPP_HOME/config/"
    
    log "Monitoring configuration deployed successfully"
}

# Deploy service installation script
deploy_service_installer() {
    log "Deploying service installation script..."
    
    # Copy installation script
    scp -i "$SSH_KEY" "$SUTAZAIAPP_HOME/scripts/install_services.sh" "$SUTAZAIAPP_USER@$DEPLOY_SERVER:$SUTAZAIAPP_HOME/scripts/"
    
    # Make script executable
    ssh -i "$SSH_KEY" "$SUTAZAIAPP_USER@$DEPLOY_SERVER" "chmod +x $SUTAZAIAPP_HOME/scripts/install_services.sh"
    
    log "Service installation script deployed successfully"
}

# Install services on deployment server
install_services() {
    log "Installing services on deployment server..."
    
    # Run installation script with sudo
    ssh -i "$SSH_KEY" "$SUTAZAIAPP_USER@$DEPLOY_SERVER" "sudo $SUTAZAIAPP_HOME/scripts/install_services.sh"
    
    log "Services installed successfully"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check monitoring service
    if ssh -i "$SSH_KEY" "$SUTAZAIAPP_USER@$DEPLOY_SERVER" "systemctl is-active sutazai-monitor" | grep -q "active"; then
        log "Monitoring service is running"
    else
        log "ERROR: Monitoring service is not running"
        return 1
    fi
    
    # Check maintenance timer
    if ssh -i "$SSH_KEY" "$SUTAZAIAPP_USER@$DEPLOY_SERVER" "systemctl is-active sutazai-maintenance.timer" | grep -q "active"; then
        log "Maintenance timer is running"
    else
        log "ERROR: Maintenance timer is not running"
        return 1
    fi
    
    # Check log files
    if ssh -i "$SSH_KEY" "$SUTAZAIAPP_USER@$DEPLOY_SERVER" "ls -l $SUTAZAIAPP_HOME/logs/monitoring.log"; then
        log "Monitoring log file exists"
    else
        log "ERROR: Monitoring log file not found"
        return 1
    fi
    
    log "Deployment verified successfully"
    return 0
}

# Main execution
main() {
    log "Starting monitoring deployment..."
    
    deploy_monitoring_scripts
    deploy_monitoring_config
    deploy_service_installer
    install_services
    
    if verify_deployment; then
        log "Monitoring deployment completed successfully"
    else
        log "ERROR: Deployment verification failed"
        exit 1
    fi
}

main 