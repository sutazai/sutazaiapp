#!/bin/bash
# Purpose: Sets up automated compliance monitoring for SutazAI
# Usage: ./setup-compliance-monitoring.sh
# Requires: Root or sudo access for cron setup

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"
MONITOR_SCRIPT="$PROJECT_ROOT/scripts/monitoring/continuous-compliance-monitor.py"
LOG_DIR="$PROJECT_ROOT/logs"
REPORT_DIR="$PROJECT_ROOT/compliance-reports"

echo "Setting up SutazAI Continuous Compliance Monitoring..."

# Create necessary directories
mkdir -p "$LOG_DIR" "$REPORT_DIR"

# Install Python dependencies
echo "Installing monitoring dependencies..."
pip install watchdog GitPython

# Create systemd service for daemon mode
echo "Creating systemd service..."
sudo tee /etc/systemd/system/sutazai-compliance-monitor.service > /dev/null <<EOF
[Unit]
Description=SutazAI Continuous Compliance Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_ROOT
ExecStart=/usr/bin/python3 $MONITOR_SCRIPT --daemon
Restart=always
RestartSec=10
StandardOutput=append:$LOG_DIR/compliance-monitor.log
StandardError=append:$LOG_DIR/compliance-monitor-error.log

[Install]
WantedBy=multi-user.target
EOF

# Create cron jobs for scheduled checks
echo "Setting up cron jobs..."
(crontab -l 2>/dev/null || true; cat <<EOF

# SutazAI Compliance Monitoring Schedule
# Hourly quick check
0 * * * * cd $PROJECT_ROOT && /usr/bin/python3 $MONITOR_SCRIPT --report-only 2>&1 | tee -a $LOG_DIR/compliance-hourly.log

# Daily comprehensive check with auto-fix
0 2 * * * cd $PROJECT_ROOT && /usr/bin/python3 $MONITOR_SCRIPT --fix 2>&1 | tee -a $LOG_DIR/compliance-daily.log

# Weekly deep analysis
0 3 * * 0 cd $PROJECT_ROOT && /usr/bin/python3 scripts/hygiene-enforcement-coordinator.py --phase=all 2>&1 | tee -a $LOG_DIR/compliance-weekly.log

# Monthly cleanup and optimization
0 4 1 * * cd $PROJECT_ROOT && /usr/bin/python3 scripts/monitoring/monthly-cleanup.py 2>&1 | tee -a $LOG_DIR/compliance-monthly.log
EOF
) | crontab -

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
cd "$PROJECT_ROOT"
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push

# Create monitoring dashboard script
cat > "$PROJECT_ROOT/scripts/monitoring/compliance-dashboard.sh" << 'EOF'
#!/bin/bash
# Simple compliance dashboard

echo "=== SutazAI Compliance Dashboard ==="
echo "Generated: $(date)"
echo

# Latest report
if [ -f "/opt/sutazaiapp/compliance-reports/latest.json" ]; then
    echo "Latest Compliance Score:"
    jq -r '.compliance_score' /opt/sutazaiapp/compliance-reports/latest.json
    echo
    echo "Total Violations:"
    jq -r '.total_violations' /opt/sutazaiapp/compliance-reports/latest.json
    echo
    echo "Critical Violations:"
    jq -r '.critical_violations' /opt/sutazaiapp/compliance-reports/latest.json
else
    echo "No compliance reports found. Run monitoring first."
fi

# Recent logs
echo
echo "=== Recent Activity ==="
tail -n 20 /opt/sutazaiapp/logs/compliance-monitor.log 2>/dev/null || echo "No logs yet"
EOF

chmod +x "$PROJECT_ROOT/scripts/monitoring/compliance-dashboard.sh"

# Enable and start the service
echo "Enabling compliance monitor service..."
sudo systemctl enable sutazai-compliance-monitor
sudo systemctl start sutazai-compliance-monitor

echo
echo "✅ Compliance monitoring setup complete!"
echo
echo "Monitoring schedule:"
echo "  - Continuous: Daemon running (systemctl status sutazai-compliance-monitor)"
echo "  - Hourly: Quick compliance check"
echo "  - Daily: Full check with auto-fix at 2 AM"
echo "  - Weekly: Deep analysis on Sundays at 3 AM"
echo "  - Monthly: Cleanup on 1st of month at 4 AM"
echo
echo "Commands:"
echo "  - View dashboard: $PROJECT_ROOT/scripts/monitoring/compliance-dashboard.sh"
echo "  - Check status: systemctl status sutazai-compliance-monitor"
echo "  - View logs: tail -f $LOG_DIR/compliance-monitor.log"
echo "  - Run manual check: python3 $MONITOR_SCRIPT"
echo
echo "Pre-commit hooks installed - all commits will be validated!"