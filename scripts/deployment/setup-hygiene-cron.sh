#!/bin/bash
# Setup script for hygiene audit cron job

set -euo pipefail


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_SCRIPT="$SCRIPT_DIR/hygiene-audit.sh"

# Make audit script executable
chmod +x "$AUDIT_SCRIPT"

# Create cron job
CRON_CMD="0 3 * * * $AUDIT_SCRIPT > /dev/null 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "hygiene-audit.sh"; then
    echo "Hygiene audit cron job already exists"
else
    # Add cron job
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "✅ Hygiene audit cron job added (runs daily at 3 AM)"
fi

# Show current crontab
echo ""
echo "Current cron jobs:"
crontab -l | grep hygiene || echo "No hygiene jobs found"

# Create systemd timer as alternative (more reliable than cron)
cat > "$(mktemp /tmp/sutazai-hygiene-audit.service.XXXXXX)" << EOF
[Unit]
Description=SutazAI Codebase Hygiene Audit
After=network.target

[Service]
Type=oneshot
ExecStart=$AUDIT_SCRIPT
WorkingDirectory=$SCRIPT_DIR
StandardOutput=journal
StandardError=journal
EOF

cat > "$(mktemp /tmp/sutazai-hygiene-audit.timer.XXXXXX)" << EOF
[Unit]
Description=Run SutazAI Hygiene Audit daily
Requires=sutazai-hygiene-audit.service

[Timer]
OnCalendar=daily
OnCalendar=*-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

echo ""
echo "Systemd timer files created in /tmp/"
echo "To install as systemd timer (recommended):"
echo "  sudo cp /tmp/sutazai-hygiene-audit.* /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable sutazai-hygiene-audit.timer"
echo "  sudo systemctl start sutazai-hygiene-audit.timer"
echo ""
echo "To check timer status:"
echo "  sudo systemctl status sutazai-hygiene-audit.timer"
echo "  sudo systemctl list-timers"

# Run audit immediately to test
echo ""
echo "Running hygiene audit now to test..."
"$AUDIT_SCRIPT" || true

echo ""
echo "✅ Hygiene audit automation setup complete!"