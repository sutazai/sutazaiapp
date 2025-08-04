#!/bin/bash
"""
SutazAI Backup Automation Setup Script
Sets up cron jobs, systemd services, and permissions for backup automation
"""

set -e

# Configuration
BACKUP_USER="root"
BACKUP_ROOT="/opt/sutazaiapp/data/backups"
SCRIPTS_ROOT="/opt/sutazaiapp/scripts/backup-automation"
LOG_DIR="/opt/sutazaiapp/logs"
CONFIG_DIR="/opt/sutazaiapp/config"

echo "Setting up SutazAI Backup Automation..."

# Create necessary directories
echo "Creating backup directories..."
mkdir -p "$BACKUP_ROOT"/{daily,weekly,monthly,postgres,sqlite,config,agents,models,monitoring,logs,verification,offsite,restore_tests}
mkdir -p "$LOG_DIR"
mkdir -p "$CONFIG_DIR"

# Set permissions
echo "Setting permissions..."
chown -R $BACKUP_USER:$BACKUP_USER "$BACKUP_ROOT"
chown -R $BACKUP_USER:$BACKUP_USER "$SCRIPTS_ROOT"
chmod -R 755 "$SCRIPTS_ROOT"
chmod +x "$SCRIPTS_ROOT"/*.py
chmod +x "$SCRIPTS_ROOT"/*/*.py

# Make Python scripts executable
echo "Making scripts executable..."
find "$SCRIPTS_ROOT" -name "*.py" -exec chmod +x {} \;

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r "$SCRIPTS_ROOT/../requirements.txt" || true

# Create systemd service for backup orchestrator
echo "Creating systemd service..."
cat > /etc/systemd/system/sutazai-backup.service << EOF
[Unit]
Description=SutazAI Backup Orchestrator
After=network.target postgresql.service docker.service

[Service]
Type=simple
User=$BACKUP_USER
WorkingDirectory=$SCRIPTS_ROOT
ExecStart=/usr/bin/python3 $SCRIPTS_ROOT/sutazai-backup-orchestrator.py schedule
Restart=always
RestartSec=60
StandardOutput=append:$LOG_DIR/backup-service.log
StandardError=append:$LOG_DIR/backup-service.log

[Install]
WantedBy=multi-user.target
EOF

# Create backup monitoring service
cat > /etc/systemd/system/sutazai-backup-monitor.service << EOF
[Unit]
Description=SutazAI Backup Monitor
After=network.target

[Service]
Type=simple
User=$BACKUP_USER
WorkingDirectory=$SCRIPTS_ROOT/alerts
ExecStart=/usr/bin/python3 $SCRIPTS_ROOT/alerts/backup-monitoring-alerting-system.py
Restart=always
RestartSec=300
StandardOutput=append:$LOG_DIR/backup-monitor.log
StandardError=append:$LOG_DIR/backup-monitor.log

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

# Create cron jobs as backup
echo "Setting up cron jobs..."
crontab -l > /tmp/current_cron 2>/dev/null || echo "" > /tmp/current_cron

# Remove existing SutazAI backup cron jobs
grep -v "SutazAI Backup" /tmp/current_cron > /tmp/new_cron || echo "" > /tmp/new_cron

# Add new cron jobs
cat >> /tmp/new_cron << EOF

# SutazAI Backup Automation
# Daily backup at 2:00 AM
0 2 * * * /usr/bin/python3 $SCRIPTS_ROOT/sutazai-backup-orchestrator.py daily >> $LOG_DIR/backup-cron.log 2>&1

# Weekly backup on Sunday at 3:00 AM
0 3 * * 0 /usr/bin/python3 $SCRIPTS_ROOT/sutazai-backup-orchestrator.py weekly >> $LOG_DIR/backup-cron.log 2>&1

# Monthly backup on 1st at 4:00 AM
0 4 1 * * /usr/bin/python3 $SCRIPTS_ROOT/sutazai-backup-orchestrator.py monthly >> $LOG_DIR/backup-cron.log 2>&1

# Backup monitoring every hour
0 * * * * /usr/bin/python3 $SCRIPTS_ROOT/alerts/backup-monitoring-alerting-system.py >> $LOG_DIR/backup-monitor-cron.log 2>&1

# Quick status check every 6 hours
0 */6 * * * /usr/bin/python3 $SCRIPTS_ROOT/utils/backup-status-checker.py >> $LOG_DIR/backup-status.log 2>&1

EOF

# Install new cron jobs
crontab /tmp/new_cron
rm /tmp/current_cron /tmp/new_cron

# Create log rotation configuration
echo "Setting up log rotation..."
cat > /etc/logrotate.d/sutazai-backup << EOF
$LOG_DIR/*backup*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF

# Create backup status script
echo "Creating backup status script..."
cat > /usr/local/bin/backup-status << EOF
#!/bin/bash
exec /usr/bin/python3 $SCRIPTS_ROOT/utils/backup-status-checker.py "\$@"
EOF

chmod +x /usr/local/bin/backup-status

# Create backup control script
cat > /usr/local/bin/backup-control << EOF
#!/bin/bash

case "\$1" in
    start)
        systemctl start sutazai-backup.service
        systemctl start sutazai-backup-monitor.service
        echo "SutazAI backup services started"
        ;;
    stop)
        systemctl stop sutazai-backup.service
        systemctl stop sutazai-backup-monitor.service
        echo "SutazAI backup services stopped"
        ;;
    restart)
        systemctl restart sutazai-backup.service
        systemctl restart sutazai-backup-monitor.service
        echo "SutazAI backup services restarted"
        ;;
    status)
        systemctl status sutazai-backup.service
        systemctl status sutazai-backup-monitor.service
        ;;
    enable)
        systemctl enable sutazai-backup.service
        systemctl enable sutazai-backup-monitor.service
        echo "SutazAI backup services enabled"
        ;;
    disable)
        systemctl disable sutazai-backup.service
        systemctl disable sutazai-backup-monitor.service
        echo "SutazAI backup services disabled"
        ;;
    run-now)
        /usr/bin/python3 $SCRIPTS_ROOT/sutazai-backup-orchestrator.py daily
        ;;
    *)
        echo "Usage: \$0 {start|stop|restart|status|enable|disable|run-now}"
        exit 1
        ;;
esac
EOF

chmod +x /usr/local/bin/backup-control

# Test backup configuration
echo "Testing backup configuration..."
if python3 -c "import json; json.load(open('$CONFIG_DIR/backup-config.json'))" 2>/dev/null; then
    echo "✓ Backup configuration is valid"
else
    echo "⚠️ Backup configuration may need adjustment"
fi

# Create initial backup directories with README files
echo "Creating README files..."
cat > "$BACKUP_ROOT/README.md" << EOF
# SutazAI Backup System

This directory contains automated backups for the SutazAI system.

## Directory Structure

- \`daily/\` - Daily backup files
- \`weekly/\` - Weekly backup files  
- \`monthly/\` - Monthly backup files
- \`postgres/\` - PostgreSQL database backups
- \`sqlite/\` - SQLite database backups
- \`config/\` - Configuration file backups
- \`agents/\` - Agent state backups
- \`models/\` - AI model backups
- \`monitoring/\` - Monitoring data backups
- \`logs/\` - Log archives
- \`verification/\` - Backup verification reports
- \`offsite/\` - Offsite replication status
- \`restore_tests/\` - Restore test results

## Usage

Check backup status:
\`\`\`bash
backup-status
\`\`\`

Control backup services:
\`\`\`bash
backup-control start|stop|restart|status
\`\`\`

Run backup manually:
\`\`\`bash
backup-control run-now
\`\`\`

## 3-2-1 Backup Strategy

This system implements the 3-2-1 backup strategy:
- 3 copies of data (original + 2 backups)
- 2 different media types (local + offsite)
- 1 offsite copy for disaster recovery
EOF

echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Review and customize configuration files in $CONFIG_DIR"
echo "2. Set up offsite backup destination (if needed)"
echo "3. Configure email/Slack notifications (optional)"
echo "4. Enable and start backup services:"
echo "   backup-control enable"
echo "   backup-control start"
echo "5. Check backup status:"
echo "   backup-status"
echo ""
echo "The backup system is now ready to use!"