#!/bin/bash

# Master setup script for two-way synchronization

# Change to the project root directory
cd /opt/sutazaiapp || exit 1

# Set up logging
LOG_FILE="logs/setup.log"
mkdir -p "$(dirname $LOG_FILE)"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Starting two-way sync setup"

# Create required directories
mkdir -p scripts/config logs/sync

# Ensure all the scripts are executable
chmod +x scripts/ssh_key_exchange.sh
chmod +x scripts/two_way_sync.sh
chmod +x scripts/sync_monitor.sh
chmod +x .git/hooks/post-commit 2>/dev/null || true

# Run SSH key exchange
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Setting up SSH key exchange"
./scripts/ssh_key_exchange.sh

# Enable and start the systemd service
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Enabling and starting systemd service"
systemctl daemon-reload
systemctl enable sutazai-sync-monitor.service
systemctl start sutazai-sync-monitor.service

# Set up crontab entry for automatic synchronization (runs every hour)
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Adding synchronization to crontab"
CRON_ENTRY="0 * * * * /opt/sutazaiapp/scripts/two_way_sync.sh >> /opt/sutazaiapp/logs/sync/cron_sync_\$(date +\%Y\%m\%d).log 2>&1"

# Check if the crontab entry already exists
if crontab -l 2>/dev/null | grep -q "two_way_sync.sh"; then
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Crontab entry already exists. No changes made."
else
    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Added hourly synchronization to crontab"
fi

# Determine which server we're on
CODE_SERVER="192.168.100.28"
DEPLOY_SERVER="192.168.100.100"
CURRENT_IP=$(hostname -I | awk '{print $1}')

if [[ "$CURRENT_IP" == "$CODE_SERVER" ]]; then
    SERVER_TYPE="code"
elif [[ "$CURRENT_IP" == "$DEPLOY_SERVER" ]]; then
    SERVER_TYPE="deploy"
else
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] WARNING: Current server IP ($CURRENT_IP) doesn't match either Code or Deployment server."
    SERVER_TYPE="unknown"
fi

echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Setup completed successfully on $SERVER_TYPE server"
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] You can now use ./scripts/two_way_sync.sh to manually trigger synchronization"
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] The sync monitor service is running and will automatically sync changes" 