#!/bin/bash
# Install Permanent Health Monitor as a systemd service

set -e

LOG_FILE="/opt/sutazaiapp/logs/health-monitor-install.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Logging function
log() {
    echo "[$TIMESTAMP] $1" | tee -a "$LOG_FILE"
}

log "Installing permanent health monitor service..."

# Make the Python script executable
chmod +x /opt/sutazaiapp/scripts/permanent-health-monitor.py

# Install Python dependencies if needed
pip3 install docker >/dev/null 2>&1 || log "Docker Python library already installed"

# Create systemd service file
cat > /etc/systemd/system/sutazai-health-monitor.service << 'EOF'
[Unit]
Description=SutazAI Container Health Monitor
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/sutazaiapp
ExecStart=/usr/bin/python3 /opt/sutazaiapp/scripts/permanent-health-monitor.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

# Environment
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

log "Created systemd service file"

# Enable and start the service
systemctl daemon-reload
systemctl enable sutazai-health-monitor.service
systemctl start sutazai-health-monitor.service

log "Health monitor service installed and started"

# Check service status
if systemctl is-active --quiet sutazai-health-monitor.service; then
    log "SUCCESS: Health monitor service is running"
else
    log "WARNING: Health monitor service failed to start"
    systemctl status sutazai-health-monitor.service || true
fi

# Create a simple status check script
cat > /opt/sutazaiapp/scripts/check-health-monitor.sh << 'EOF'
#!/bin/bash
# Check health monitor status

echo "=== SutazAI Health Monitor Status ==="
echo

echo "Service Status:"
systemctl status sutazai-health-monitor.service --no-pager -l

echo
echo "Recent Logs:"
journalctl -u sutazai-health-monitor.service --no-pager -n 20

echo
echo "Health Statistics:"
if [[ -f /opt/sutazaiapp/logs/health_monitor_stats.json ]]; then
    cat /opt/sutazaiapp/logs/health_monitor_stats.json | python3 -m json.tool
else
    echo "No statistics available yet"
fi

echo
echo "Current Container Health:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(healthy|unhealthy|sutazai-)" | head -20
EOF

chmod +x /opt/sutazaiapp/scripts/check-health-monitor.sh

log "Created health monitor status check script"
log "Use './scripts/check-health-monitor.sh' to check monitor status"
log "Installation completed successfully"