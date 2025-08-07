#!/bin/bash
# Purpose: Setup comprehensive automation cron jobs for SutazAI system
# Usage: ./setup-automation-cron.sh [--systemd] [--remove]
# Requires: Root privileges for systemd setup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/opt/sutazaiapp"
LOG_DIR="$BASE_DIR/logs"

# Configuration
USE_SYSTEMD=false
REMOVE_JOBS=false
ENABLE_EMAIL_NOTIFICATIONS=false
EMAIL_RECIPIENT="${SUTAZAI_ADMIN_EMAIL:-}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --systemd)
            USE_SYSTEMD=true
            shift
            ;;
        --remove)
            REMOVE_JOBS=true
            shift
            ;;
        --email)
            ENABLE_EMAIL_NOTIFICATIONS=true
            EMAIL_RECIPIENT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--systemd] [--remove] [--email recipient@domain.com]"
            exit 1
            ;;
    esac
done

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        ERROR) echo -e "${RED}[$timestamp] ERROR: $message${NC}" ;;
        WARN) echo -e "${YELLOW}[$timestamp] WARN: $message${NC}" ;;
        INFO) echo -e "${BLUE}[$timestamp] INFO: $message${NC}" ;;
        SUCCESS) echo -e "${GREEN}[$timestamp] SUCCESS: $message${NC}" ;;
    esac
}

# Check if running as root (required for systemd setup)
check_privileges() {
    if [[ "$USE_SYSTEMD" == "true" && "$EUID" -ne 0 ]]; then
        log "ERROR" "Root privileges required for systemd setup. Use sudo or run as root."
        exit 1
    fi
}

# Make all automation scripts executable
setup_script_permissions() {
    log "INFO" "Setting up script permissions..."
    
    local automation_scripts=(
        "daily-health-check.sh"
        "log-rotation-cleanup.sh"
        "database-maintenance.sh"
        "certificate-renewal.sh"
        "agent-restart-monitor.sh"
        "performance-report-generator.sh"
        "security-scanner.sh"
        "backup-verification.sh"
    )
    
    for script in "${automation_scripts[@]}"; do
        local script_path="$SCRIPT_DIR/$script"
        if [[ -f "$script_path" ]]; then
            chmod +x "$script_path"
            log "SUCCESS" "Made executable: $script"
        else
            log "WARN" "Script not found: $script_path"
        fi
    done
}

# Create cron jobs
setup_cron_jobs() {
    log "INFO" "Setting up cron jobs for automation..."
    
    # Remove existing SutazAI automation cron jobs
    if crontab -l 2>/dev/null | grep -q "sutazai-automation"; then
        log "INFO" "Removing existing SutazAI automation cron jobs..."
        crontab -l 2>/dev/null | grep -v "sutazai-automation" | crontab -
    fi
    
    if [[ "$REMOVE_JOBS" == "true" ]]; then
        log "SUCCESS" "Removed existing automation cron jobs"
        return 0
    fi
    
    # Email notification setup
    local email_suffix=""
    if [[ "$ENABLE_EMAIL_NOTIFICATIONS" == "true" && -n "$EMAIL_RECIPIENT" ]]; then
        email_suffix=" --email $EMAIL_RECIPIENT"
    fi
    
    # Define cron jobs with comments
    local cron_jobs=(
        # Daily health check at 6:00 AM
        "0 6 * * * $SCRIPT_DIR/daily-health-check.sh$email_suffix > /dev/null 2>&1 # sutazai-automation-health-check"
        
        # Log rotation and cleanup at 2:00 AM daily
        "0 2 * * * $SCRIPT_DIR/log-rotation-cleanup.sh > /dev/null 2>&1 # sutazai-automation-log-cleanup"
        
        # Database maintenance at 3:00 AM daily
        "0 3 * * * $SCRIPT_DIR/database-maintenance.sh > /dev/null 2>&1 # sutazai-automation-db-maintenance"
        
        # Certificate renewal check at 4:00 AM daily
        "0 4 * * * $SCRIPT_DIR/certificate-renewal.sh > /dev/null 2>&1 # sutazai-automation-cert-renewal"
        
        # Agent restart monitoring every 5 minutes
        "*/5 * * * * $SCRIPT_DIR/agent-restart-monitor.sh > /dev/null 2>&1 # sutazai-automation-agent-monitor"
        
        # Performance report generation at 7:00 AM daily
        "0 7 * * * $SCRIPT_DIR/performance-report-generator.sh --format both --period daily > /dev/null 2>&1 # sutazai-automation-performance-daily"
        
        # Weekly performance report on Sundays at 8:00 AM
        "0 8 * * 0 $SCRIPT_DIR/performance-report-generator.sh --format both --period weekly > /dev/null 2>&1 # sutazai-automation-performance-weekly"
        
        # Security scanning at 1:00 AM daily
        "0 1 * * * $SCRIPT_DIR/security-scanner.sh --report-format both > /dev/null 2>&1 # sutazai-automation-security-scan"
        
        # Full security scan weekly on Saturdays at 11:00 PM
        "0 23 * * 6 $SCRIPT_DIR/security-scanner.sh --full-scan --report-format both > /dev/null 2>&1 # sutazai-automation-security-full"
        
        # Backup verification at 5:00 AM daily
        "0 5 * * * $SCRIPT_DIR/backup-verification.sh > /dev/null 2>&1 # sutazai-automation-backup-verify"
        
        # Comprehensive backup verification weekly on Sundays at 5:30 AM
        "30 5 * * 0 $SCRIPT_DIR/backup-verification.sh --verify-all > /dev/null 2>&1 # sutazai-automation-backup-verify-full"
    )
    
    # Add jobs to crontab
    local temp_cron_file=$(mktemp)
    crontab -l 2>/dev/null > "$temp_cron_file" || true
    
    for job in "${cron_jobs[@]}"; do
        echo "$job" >> "$temp_cron_file"
        log "INFO" "Added cron job: $(echo "$job" | cut -d'#' -f2 | sed 's/sutazai-automation-//' | xargs)"
    done
    
    # Install new crontab
    if crontab "$temp_cron_file"; then
        log "SUCCESS" "Installed ${#cron_jobs[@]} automation cron jobs"
    else
        log "ERROR" "Failed to install cron jobs"
        rm -f "$temp_cron_file"
        return 1
    fi
    
    rm -f "$temp_cron_file"
}

# Create systemd services and timers
setup_systemd_services() {
    log "INFO" "Setting up systemd services and timers for automation..."
    
    if [[ "$REMOVE_JOBS" == "true" ]]; then
        log "INFO" "Removing existing systemd automation services..."
        
        # Stop and disable existing services
        local services=(
            "sutazai-health-check"
            "sutazai-log-cleanup"
            "sutazai-db-maintenance"
            "sutazai-cert-renewal"
            "sutazai-agent-monitor"
            "sutazai-performance-daily"
            "sutazai-performance-weekly"
            "sutazai-security-scan"
            "sutazai-security-full"
            "sutazai-backup-verify"
            "sutazai-backup-verify-full"
        )
        
        for service in "${services[@]}"; do
            systemctl stop "${service}.timer" 2>/dev/null || true
            systemctl disable "${service}.timer" 2>/dev/null || true
            rm -f "/etc/systemd/system/${service}.service" "/etc/systemd/system/${service}.timer"
        done
        
        systemctl daemon-reload
        log "SUCCESS" "Removed systemd automation services"
        return 0
    fi
    
    # Email notification setup
    local email_suffix=""
    if [[ "$ENABLE_EMAIL_NOTIFICATIONS" == "true" && -n "$EMAIL_RECIPIENT" ]]; then
        email_suffix=" --email $EMAIL_RECIPIENT"
    fi
    
    # Define systemd service configurations
    declare -A services=(
        ["sutazai-health-check"]="$SCRIPT_DIR/daily-health-check.sh$email_suffix"
        ["sutazai-log-cleanup"]="$SCRIPT_DIR/log-rotation-cleanup.sh"
        ["sutazai-db-maintenance"]="$SCRIPT_DIR/database-maintenance.sh"
        ["sutazai-cert-renewal"]="$SCRIPT_DIR/certificate-renewal.sh"
        ["sutazai-agent-monitor"]="$SCRIPT_DIR/agent-restart-monitor.sh"
        ["sutazai-performance-daily"]="$SCRIPT_DIR/performance-report-generator.sh --format both --period daily"
        ["sutazai-performance-weekly"]="$SCRIPT_DIR/performance-report-generator.sh --format both --period weekly"
        ["sutazai-security-scan"]="$SCRIPT_DIR/security-scanner.sh --report-format both"
        ["sutazai-security-full"]="$SCRIPT_DIR/security-scanner.sh --full-scan --report-format both"
        ["sutazai-backup-verify"]="$SCRIPT_DIR/backup-verification.sh"
        ["sutazai-backup-verify-full"]="$SCRIPT_DIR/backup-verification.sh --verify-all"
    )
    
    # Define timer schedules
    declare -A timers=(
        ["sutazai-health-check"]="*-*-* 06:00:00"
        ["sutazai-log-cleanup"]="*-*-* 02:00:00"
        ["sutazai-db-maintenance"]="*-*-* 03:00:00"
        ["sutazai-cert-renewal"]="*-*-* 04:00:00"
        ["sutazai-agent-monitor"]="*:0/5:0"
        ["sutazai-performance-daily"]="*-*-* 07:00:00"
        ["sutazai-performance-weekly"]="Sun *-*-* 08:00:00"
        ["sutazai-security-scan"]="*-*-* 01:00:00"
        ["sutazai-security-full"]="Sat *-*-* 23:00:00"
        ["sutazai-backup-verify"]="*-*-* 05:00:00"
        ["sutazai-backup-verify-full"]="Sun *-*-* 05:30:00"
    )
    
    # Create service and timer files
    for service_name in "${!services[@]}"; do
        local service_command="${services[$service_name]}"
        local timer_schedule="${timers[$service_name]}"
        
        # Create service file
        cat > "/etc/systemd/system/${service_name}.service" << EOF
[Unit]
Description=SutazAI Automation - ${service_name}
After=network.target docker.service
Wants=docker.service

[Service]
Type=oneshot
ExecStart=${service_command}
WorkingDirectory=${BASE_DIR}
StandardOutput=journal
StandardError=journal
User=root
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[Install]
WantedBy=multi-user.target
EOF
        
        # Create timer file
        cat > "/etc/systemd/system/${service_name}.timer" << EOF
[Unit]
Description=Timer for SutazAI Automation - ${service_name}
Requires=${service_name}.service

[Timer]
OnCalendar=${timer_schedule}
Persistent=true
RandomizedDelaySec=30

[Install]
WantedBy=timers.target
EOF
        
        log "SUCCESS" "Created systemd service: ${service_name}"
    done
    
    # Reload systemd and enable timers
    systemctl daemon-reload
    
    for service_name in "${!services[@]}"; do
        systemctl enable "${service_name}.timer"
        systemctl start "${service_name}.timer"
        log "SUCCESS" "Enabled and started timer: ${service_name}.timer"
    done
    
    log "SUCCESS" "Installed ${#services[@]} systemd automation services"
}

# Install mail command for email notifications
setup_email_notifications() {
    if [[ "$ENABLE_EMAIL_NOTIFICATIONS" == "true" ]]; then
        log "INFO" "Setting up email notifications..."
        
        if ! command -v mail >/dev/null 2>&1; then
            log "INFO" "Installing mail command for notifications..."
            
            if command -v apt-get >/dev/null 2>&1; then
                apt-get update && apt-get install -y mailutils
            elif command -v yum >/dev/null 2>&1; then
                yum install -y mailx
            else
                log "WARN" "Cannot install mail command automatically. Email notifications may not work."
            fi
        fi
        
        if command -v mail >/dev/null 2>&1; then
            log "SUCCESS" "Email notifications configured for: $EMAIL_RECIPIENT"
        else
            log "ERROR" "Failed to setup email notifications"
        fi
    fi
}

# Create automation monitoring dashboard
create_monitoring_dashboard() {
    log "INFO" "Creating automation monitoring dashboard..."
    
    local dashboard_dir="$BASE_DIR/dashboard"
    mkdir -p "$dashboard_dir"
    
    local dashboard_file="$dashboard_dir/automation-status.html"
    
    cat > "$dashboard_file" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SutazAI Automation Status</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .status-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }
        .status-title { font-weight: bold; font-size: 1.1em; margin-bottom: 10px; }
        .status-ok { border-left-color: #28a745; }
        .status-warn { border-left-color: #ffc107; }
        .status-error { border-left-color: #dc3545; }
        .refresh-btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .last-update { text-align: center; margin-top: 20px; font-size: 0.9em; color: #666; }
    </style>
    <script>
        function refreshData() {
            location.reload();
        }
        
        function loadLatestReports() {
            // This would be enhanced with actual AJAX calls to load latest reports
            document.getElementById('last-update').innerHTML = 'Last updated: ' + new Date().toLocaleString();
        }
        
        // Auto-refresh every 5 minutes
        setInterval(refreshData, 300000);
        
        // Load data on page load
        window.onload = function() {
            loadLatestReports();
        };
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SutazAI Automation Status Dashboard</h1>
            <button class="refresh-btn" onclick="refreshData()">Refresh Status</button>
        </div>
        
        <div class="status-grid">
            <div class="status-card status-ok">
                <div class="status-title">Daily Health Check</div>
                <p>Runs daily at 6:00 AM</p>
                <p>Status: <span id="health-status">Checking...</span></p>
                <p>Last run: <span id="health-last">Loading...</span></p>
            </div>
            
            <div class="status-card status-ok">
                <div class="status-title">Log Cleanup</div>
                <p>Runs daily at 2:00 AM</p>
                <p>Status: <span id="log-status">Checking...</span></p>
                <p>Last run: <span id="log-last">Loading...</span></p>
            </div>
            
            <div class="status-card status-ok">
                <div class="status-title">Database Maintenance</div>
                <p>Runs daily at 3:00 AM</p>
                <p>Status: <span id="db-status">Checking...</span></p>
                <p>Last run: <span id="db-last">Loading...</span></p>
            </div>
            
            <div class="status-card status-ok">
                <div class="status-title">Certificate Renewal</div>
                <p>Runs daily at 4:00 AM</p>
                <p>Status: <span id="cert-status">Checking...</span></p>
                <p>Last run: <span id="cert-last">Loading...</span></p>
            </div>
            
            <div class="status-card status-ok">
                <div class="status-title">Agent Monitor</div>
                <p>Runs every 5 minutes</p>
                <p>Status: <span id="agent-status">Checking...</span></p>
                <p>Last run: <span id="agent-last">Loading...</span></p>
            </div>
            
            <div class="status-card status-ok">
                <div class="status-title">Performance Reports</div>
                <p>Daily at 7:00 AM, Weekly on Sundays</p>
                <p>Status: <span id="perf-status">Checking...</span></p>
                <p>Last run: <span id="perf-last">Loading...</span></p>
            </div>
            
            <div class="status-card status-ok">
                <div class="status-title">Security Scanning</div>
                <p>Daily at 1:00 AM, Full scan Saturdays</p>
                <p>Status: <span id="security-status">Checking...</span></p>
                <p>Last run: <span id="security-last">Loading...</span></p>
            </div>
            
            <div class="status-card status-ok">
                <div class="status-title">Backup Verification</div>
                <p>Daily at 5:00 AM, Full on Sundays</p>
                <p>Status: <span id="backup-status">Checking...</span></p>
                <p>Last run: <span id="backup-last">Loading...</span></p>
            </div>
        </div>
        
        <div class="last-update">
            <p id="last-update">Page loaded: <script>document.write(new Date().toLocaleString());</script></p>
        </div>
    </div>
</body>
</html>
EOF
    
    log "SUCCESS" "Created automation monitoring dashboard: $dashboard_file"
}

# Show current automation status
show_automation_status() {
    log "INFO" "Current automation status:"
    echo
    
    if [[ "$USE_SYSTEMD" == "true" ]]; then
        echo "=== Systemd Timers Status ==="
        systemctl list-timers --no-pager | grep sutazai || echo "No SutazAI timers found"
    else
        echo "=== Cron Jobs Status ==="
        crontab -l 2>/dev/null | grep sutazai-automation || echo "No SutazAI automation cron jobs found"
    fi
    
    echo
    echo "=== Recent Log Files ==="
    find "$LOG_DIR" -name "*automation*" -o -name "*health*" -o -name "*performance*" -o -name "*security*" -o -name "*backup*" -type f -mtime -1 2>/dev/null | head -10 || echo "No recent automation logs found"
    
    echo
    echo "=== Available Reports ==="
    find "$BASE_DIR/reports" -name "*.json" -o -name "*.html" -type f -mtime -1 2>/dev/null | head -10 || echo "No recent reports found"
}

# Main execution
main() {
    log "INFO" "Setting up SutazAI automation system"
    
    if [[ "$REMOVE_JOBS" == "true" ]]; then
        log "INFO" "Removing existing automation jobs..."
    else
        log "INFO" "Installation mode: $([ "$USE_SYSTEMD" == "true" ] && echo "SYSTEMD" || echo "CRON")"
        if [[ "$ENABLE_EMAIL_NOTIFICATIONS" == "true" ]]; then
            log "INFO" "Email notifications enabled for: $EMAIL_RECIPIENT"
        fi
    fi
    
    # Check privileges if needed
    check_privileges
    
    # Setup script permissions
    setup_script_permissions
    
    # Create necessary directories
    mkdir -p "$LOG_DIR" "$BASE_DIR/reports" "$BASE_DIR/dashboard"
    
    # Setup email notifications if enabled
    setup_email_notifications
    
    # Setup automation jobs
    if [[ "$USE_SYSTEMD" == "true" ]]; then
        setup_systemd_services
    else
        setup_cron_jobs
    fi
    
    # Create monitoring dashboard
    if [[ "$REMOVE_JOBS" == "false" ]]; then
        create_monitoring_dashboard
    fi
    
    # Show current status
    show_automation_status
    
    if [[ "$REMOVE_JOBS" == "true" ]]; then
        log "SUCCESS" "SutazAI automation system removed successfully"
    else
        log "SUCCESS" "SutazAI automation system setup completed successfully"
        
        echo
        echo "============================================"
        echo "     AUTOMATION SETUP SUMMARY"
        echo "============================================"
        echo "Installation Type: $([ "$USE_SYSTEMD" == "true" ] && echo "Systemd Timers" || echo "Cron Jobs")"
        echo "Email Notifications: $([ "$ENABLE_EMAIL_NOTIFICATIONS" == "true" ] && echo "Enabled ($EMAIL_RECIPIENT)" || echo "Disabled")"
        echo "Monitoring Dashboard: $BASE_DIR/dashboard/automation-status.html"
        echo "Log Directory: $LOG_DIR"
        echo "Reports Directory: $BASE_DIR/reports"
        echo ""
        echo "Scheduled Tasks:"
        echo "- Daily Health Check: 6:00 AM"
        echo "- Log Cleanup: 2:00 AM daily"
        echo "- Database Maintenance: 3:00 AM daily"
        echo "- Certificate Renewal: 4:00 AM daily"
        echo "- Agent Monitoring: Every 5 minutes"
        echo "- Performance Reports: 7:00 AM daily, 8:00 AM Sundays"
        echo "- Security Scanning: 1:00 AM daily, 11:00 PM Saturdays"
        echo "- Backup Verification: 5:00 AM daily, 5:30 AM Sundays"
        echo ""
        echo "Next Steps:"
        echo "1. Monitor logs in $LOG_DIR"
        echo "2. Check reports in $BASE_DIR/reports"
        echo "3. Access dashboard at file://$BASE_DIR/dashboard/automation-status.html"
        if [[ "$USE_SYSTEMD" == "true" ]]; then
            echo "4. Check timer status: systemctl list-timers | grep sutazai"
        else
            echo "4. Check cron status: crontab -l | grep sutazai-automation"
        fi
        echo "============================================"
    fi
}

# Run main function
main "$@"