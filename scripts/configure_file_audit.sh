#!/bin/bash
# File activity monitoring for critical paths
AUDIT_RULES="/etc/audit/rules.d/sutazai.rules"

echo "Configuring SutazAI file tracking..."
cat > "$AUDIT_RULES" << EOF
# SutazAI critical paths
-w /etc/sutazai -p wa -k sutazai_config
-w /opt/sutazaiapp/models -p wa -k sutazai_models
-w /var/log/sutazai -p wa -k sutazai_logs
-w /usr/local/bin/sutazai -p wa -k sutazai_bin
EOF

# Apply audit rules
auditctl -R "$AUDIT_RULES"
sudo chmod 750 /usr/local/bin/sutazai-audit
sudo chown root:root /usr/local/bin/sutazai-audit
systemctl daemon-reload
systemctl restart auditd.service

echo "File tracking system ready"

# Add new file audit script
set -eo pipefail

CONFIG_DIR="/etc/sutazai"
LOG_DIR="/var/log/sutazai/audit"

# Create audit rules
create_audit_rules() {
    # Monitor critical directories
    auditctl -w ${CONFIG_DIR} -p wa -k sutazai_config
    auditctl -w /opt/sutazaiapp/bin -p wa -k sutazai_binaries
    auditctl -w ${LOG_DIR} -p wa -k sutazai_logs
    
    # Monitor deployment scripts
    find /opt/sutazaiapp/scripts -type f -exec auditctl -w {} -p wa -k sutazai_scripts \;
}

# Rotate audit logs
setup_audit_logging() {
    mkdir -p "${LOG_DIR}"
    chmod 700 "${LOG_DIR}"
    
    # Add logrotate configuration
    cat > /etc/logrotate.d/sutazai_audit <<EOF
${LOG_DIR}/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    sharedscripts
    postrotate
        systemctl restart auditd.service >/dev/null 2>&1 || true
    endscript
}
EOF
}

# Main execution
case $1 in
    install)
        create_audit_rules
        setup_audit_logging
        ;;
    *)
        echo "Usage: $0 install"
        exit 1
        ;;
esac