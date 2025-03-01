#!/bin/bash

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root"
    exit 1
fi

# Create necessary directories
mkdir -p /opt/sutazaiapp/logs/autonomous_monitor
mkdir -p /opt/sutazaiapp/config

# Set proper permissions
chown -R root:root /opt/sutazaiapp/scripts
chmod -R 755 /opt/sutazaiapp/scripts
chown -R root:root /opt/sutazaiapp/config
chmod -R 644 /opt/sutazaiapp/config
chown -R root:root /opt/sutazaiapp/logs
chmod -R 755 /opt/sutazaiapp/logs

# Stop any existing monitoring processes
systemctl stop sutazai-monitor.service 2>/dev/null
pkill -f "scripts.autonomous_monitor" 2>/dev/null

# Reload systemd configuration
systemctl daemon-reload

# Enable and start the service
systemctl enable sutazai-monitor.service
systemctl start sutazai-monitor.service

# Wait for service to start
sleep 5

# Check service status
systemctl status sutazai-monitor.service

echo "Setup complete. Check logs at /opt/sutazaiapp/logs/autonomous_monitor/monitor.log"
echo "Use 'systemctl status sutazai-monitor.service' to check service status" 