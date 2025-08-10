#!/bin/bash

# Backup Scheduler Setup Script for SutazAI System
# Sets up automated backup scheduling with monitoring and alerting
# Author: DevOps Manager
# Date: 2025-08-09

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRON_USER="root"
SYSTEMD_SERVICE_DIR="/etc/systemd/system"
BACKUP_SCHEDULE="daily"  # daily, hourly, custom
CUSTOM_SCHEDULE=""  # cron format if using custom

# Backup timing configuration
DAILY_BACKUP_TIME="02:00"    # 2:00 AM
HOURLY_BACKUP_MINUTES="30"   # 30 minutes past each hour
WEEKLY_BACKUP_DAY="sunday"   # Day of week for weekly full backup
WEEKLY_BACKUP_TIME="01:00"   # 1:00 AM on specified day

# Monitoring configuration
HEALTHCHECK_URL="${BACKUP_HEALTHCHECK_URL:-}"
NOTIFICATION_EMAIL="${BACKUP_NOTIFICATION_EMAIL:-}"
SLACK_WEBHOOK="${BACKUP_SLACK_WEBHOOK:-}"
LOG_RETENTION_DAYS=30

# Command line options
SCHEDULE_TYPE="${1:-daily}"
USE_SYSTEMD="${2:-auto}"

# Logging
LOG_FILE="/opt/sutazaiapp/logs/backup_scheduler_setup.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

# Display usage information
usage() {
    echo "Usage: $0 [schedule_type] [use_systemd]"
    echo ""
    echo "Schedule Types:"
    echo "  daily     - Run backup daily at 2:00 AM (default)"
    echo "  hourly    - Run backup every hour at 30 minutes past"
    echo "  weekly    - Run backup weekly on Sunday at 1:00 AM"
    echo "  custom    - Use custom cron schedule (set CUSTOM_SCHEDULE env var)"
    echo ""
    echo "Systemd Options:"
    echo "  auto      - Auto-detect and prefer systemd if available"
    echo "  yes       - Force use of systemd timers"
    echo "  no        - Force use of traditional cron"
    echo ""
    echo "Environment Variables:"
    echo "  CUSTOM_SCHEDULE           - Custom cron schedule (e.g., '0 */6 * * *')"
    echo "  BACKUP_HEALTHCHECK_URL    - URL to ping on successful backup"
    echo "  BACKUP_NOTIFICATION_EMAIL - Email for backup notifications"
    echo "  BACKUP_SLACK_WEBHOOK      - Slack webhook for notifications"
    echo ""
    echo "Examples:"
    echo "  $0 daily                  # Set up daily backups at 2:00 AM"
    echo "  $0 hourly no              # Set up hourly backups using cron"
    echo "  CUSTOM_SCHEDULE='0 */4 * * *' $0 custom  # Every 4 hours"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if running as root
    if [ "$(id -u)" -ne 0 ]; then
        error_exit "This script must be run as root to set up system scheduling"
    fi
    
    # Check if backup scripts exist
    local required_scripts=(
        "$SCRIPT_DIR/master-backup.sh"
        "$SCRIPT_DIR/backup-redis.sh"
        "$SCRIPT_DIR/backup-neo4j.sh"
        "$SCRIPT_DIR/backup-vector-databases.sh"
    )
    
    for script in "${required_scripts[@]}"; do
        if [ ! -x "$script" ]; then
            error_exit "Required backup script not found or not executable: $script"
        fi
    done
    
    # Check Docker availability
    if ! command -v docker > /dev/null 2>&1; then
        error_exit "Docker is not installed or not in PATH"
    fi
    
    log "Prerequisites check completed"
}

# Detect best scheduling method
detect_scheduling_method() {
    if [ "$USE_SYSTEMD" = "yes" ]; then
        echo "systemd"
        return 0
    elif [ "$USE_SYSTEMD" = "no" ]; then
        echo "cron"
        return 0
    fi
    
    # Auto-detect
    if systemctl --version > /dev/null 2>&1 && [ -d "$SYSTEMD_SERVICE_DIR" ]; then
        echo "systemd"
    elif command -v crontab > /dev/null 2>&1; then
        echo "cron"
    else
        error_exit "Neither systemd nor cron is available"
    fi
}

# Generate cron schedule
generate_cron_schedule() {
    local schedule_type="$1"
    
    case "$schedule_type" in
        "daily")
            echo "0 $(echo "$DAILY_BACKUP_TIME" | cut -d: -f2) $(echo "$DAILY_BACKUP_TIME" | cut -d: -f1) * *"
            ;;
        "hourly")
            echo "$HOURLY_BACKUP_MINUTES * * * *"
            ;;
        "weekly")
            local day_num
            case "$WEEKLY_BACKUP_DAY" in
                "sunday") day_num=0 ;;
                "monday") day_num=1 ;;
                "tuesday") day_num=2 ;;
                "wednesday") day_num=3 ;;
                "thursday") day_num=4 ;;
                "friday") day_num=5 ;;
                "saturday") day_num=6 ;;
                *) day_num=0 ;;
            esac
            echo "0 $(echo "$WEEKLY_BACKUP_TIME" | cut -d: -f2) $(echo "$WEEKLY_BACKUP_TIME" | cut -d: -f1) * $day_num"
            ;;
        "custom")
            if [ -n "$CUSTOM_SCHEDULE" ]; then
                echo "$CUSTOM_SCHEDULE"
            else
                error_exit "CUSTOM_SCHEDULE environment variable not set"
            fi
            ;;
        *)
            error_exit "Unknown schedule type: $schedule_type"
            ;;
    esac
}

# Set up cron-based scheduling
setup_cron_scheduling() {
    local schedule_type="$1"
    
    log "Setting up cron-based backup scheduling..."
    
    local cron_schedule
    cron_schedule=$(generate_cron_schedule "$schedule_type")
    
    # Create backup wrapper script
    local wrapper_script="/usr/local/bin/sutazai-backup-wrapper.sh"
    cat > "$wrapper_script" << 'EOF'
#!/bin/bash
# SutazAI Backup Cron Wrapper
# Auto-generated by backup scheduler setup

set -euo pipefail

# Configuration
BACKUP_SCRIPT="/opt/sutazaiapp/scripts/maintenance/master-backup.sh"
LOCK_FILE="/var/run/sutazai-backup.lock"
LOG_FILE="/opt/sutazaiapp/logs/scheduled_backup_$(date +%Y%m%d_%H%M%S).log"

# Notification settings
HEALTHCHECK_URL="${BACKUP_HEALTHCHECK_URL:-}"
NOTIFICATION_EMAIL="${BACKUP_NOTIFICATION_EMAIL:-}"
SLACK_WEBHOOK="${BACKUP_SLACK_WEBHOOK:-}"

# Lock file handling
cleanup_lock() {
    rm -f "$LOCK_FILE"
}

trap cleanup_lock EXIT

# Check for existing backup process
if [ -f "$LOCK_FILE" ]; then
    local pid
    pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "Backup process already running (PID: $pid), exiting"
        exit 0
    else
        echo "Stale lock file found, removing"
        rm -f "$LOCK_FILE"
    fi
fi

# Create lock file
echo $$ > "$LOCK_FILE"

# Function to send notifications
send_notification() {
    local subject="$1"
    local message="$2"
    local success="$3"
    
    # Email notification
    if [ -n "$NOTIFICATION_EMAIL" ] && command -v mail > /dev/null 2>&1; then
        echo -e "$message" | mail -s "$subject" "$NOTIFICATION_EMAIL" 2>/dev/null || true
    fi
    
    # Slack notification
    if [ -n "$SLACK_WEBHOOK" ] && command -v curl > /dev/null 2>&1; then
        local emoji="✅"
        if [ "$success" != "true" ]; then
            emoji="❌"
        fi
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$emoji $subject\\n\\n$message\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
    
    # Healthcheck ping (only on success)
    if [ "$success" = "true" ] && [ -n "$HEALTHCHECK_URL" ] && command -v curl > /dev/null 2>&1; then
        curl -fsS "$HEALTHCHECK_URL" > /dev/null 2>&1 || true
    fi
}

# Run backup
echo "Starting scheduled backup at $(date)"
START_TIME=$(date +%s)

if "$BACKUP_SCRIPT" > "$LOG_FILE" 2>&1; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "Backup completed successfully in ${DURATION}s"
    send_notification "SutazAI Backup Successful" \
        "Scheduled backup completed successfully\\n\\nDuration: ${DURATION}s\\nLog: $LOG_FILE" \
        "true"
    exit 0
else
    local exit_code=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "Backup failed with exit code $exit_code after ${DURATION}s"
    send_notification "SutazAI Backup FAILED" \
        "Scheduled backup failed with exit code $exit_code\\n\\nDuration: ${DURATION}s\\nLog: $LOG_FILE" \
        "false"
    exit $exit_code
fi
EOF
    
    # Set environment variables in wrapper script
    if [ -n "$HEALTHCHECK_URL" ]; then
        sed -i "s|HEALTHCHECK_URL=\".*\"|HEALTHCHECK_URL=\"$HEALTHCHECK_URL\"|" "$wrapper_script"
    fi
    if [ -n "$NOTIFICATION_EMAIL" ]; then
        sed -i "s|NOTIFICATION_EMAIL=\".*\"|NOTIFICATION_EMAIL=\"$NOTIFICATION_EMAIL\"|" "$wrapper_script"
    fi
    if [ -n "$SLACK_WEBHOOK" ]; then
        sed -i "s|SLACK_WEBHOOK=\".*\"|SLACK_WEBHOOK=\"$SLACK_WEBHOOK\"|" "$wrapper_script"
    fi
    
    chmod +x "$wrapper_script"
    log "Created backup wrapper script: $wrapper_script"
    
    # Add to cron
    local cron_entry="$cron_schedule $wrapper_script >/dev/null 2>&1"
    
    # Check if entry already exists
    if crontab -l 2>/dev/null | grep -q "sutazai-backup-wrapper"; then
        log "Removing existing SutazAI backup cron entry"
        crontab -l 2>/dev/null | grep -v "sutazai-backup-wrapper" | crontab -
    fi
    
    # Add new entry
    (crontab -l 2>/dev/null; echo "# SutazAI Database Backup - $schedule_type schedule"; echo "$cron_entry") | crontab -
    
    log "Added cron entry: $cron_entry"
    log "Cron-based scheduling setup completed"
}

# Set up systemd-based scheduling
setup_systemd_scheduling() {
    local schedule_type="$1"
    
    log "Setting up systemd-based backup scheduling..."
    
    # Create systemd service
    local service_file="$SYSTEMD_SERVICE_DIR/sutazai-backup.service"
    cat > "$service_file" << EOF
[Unit]
Description=SutazAI Database Backup Service
Documentation=file:///opt/sutazaiapp/scripts/maintenance/master-backup.sh
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
User=root
Group=root
WorkingDirectory=/opt/sutazaiapp
ExecStart=/opt/sutazaiapp/scripts/maintenance/master-backup.sh
TimeoutStartSec=1800
StandardOutput=journal
StandardError=journal

# Environment variables
Environment=BACKUP_HEALTHCHECK_URL=${HEALTHCHECK_URL}
Environment=BACKUP_NOTIFICATION_EMAIL=${NOTIFICATION_EMAIL}
Environment=BACKUP_SLACK_WEBHOOK=${SLACK_WEBHOOK}

[Install]
WantedBy=multi-user.target
EOF
    
    # Create systemd timer
    local timer_file="$SYSTEMD_SERVICE_DIR/sutazai-backup.timer"
    local timer_schedule
    
    case "$schedule_type" in
        "daily")
            timer_schedule="OnCalendar=daily $DAILY_BACKUP_TIME"
            ;;
        "hourly")
            timer_schedule="OnCalendar=hourly"
            ;;
        "weekly")
            timer_schedule="OnCalendar=weekly"
            ;;
        "custom")
            # Convert cron to systemd calendar format (simplified)
            timer_schedule="OnCalendar=daily $DAILY_BACKUP_TIME"
            log "WARNING: Custom schedule conversion to systemd format not implemented, using daily"
            ;;
    esac
    
    cat > "$timer_file" << EOF
[Unit]
Description=SutazAI Database Backup Timer
Documentation=file:///opt/sutazaiapp/scripts/maintenance/master-backup.sh
Requires=sutazai-backup.service

[Timer]
$timer_schedule
RandomizedDelaySec=300
Persistent=true

[Install]
WantedBy=timers.target
EOF
    
    # Reload systemd and enable timer
    systemctl daemon-reload
    systemctl enable sutazai-backup.timer
    systemctl start sutazai-backup.timer
    
    log "Created systemd service: $service_file"
    log "Created systemd timer: $timer_file"
    log "Systemd-based scheduling setup completed"
}

# Set up log rotation
setup_log_rotation() {
    log "Setting up log rotation for backup logs..."
    
    local logrotate_config="/etc/logrotate.d/sutazai-backup"
    cat > "$logrotate_config" << EOF
/opt/sutazaiapp/logs/*backup*.log {
    daily
    rotate $LOG_RETENTION_DAYS
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
    postrotate
        # Clean up old backup files older than retention period
        find /opt/sutazaiapp/backups -type f -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
        # Clean up old logs older than retention period
        find /opt/sutazaiapp/logs -name "*backup*" -type f -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
    endscript
}
EOF
    
    log "Created log rotation config: $logrotate_config"
}

# Set up monitoring and alerting
setup_monitoring() {
    log "Setting up backup monitoring..."
    
    # Create backup monitoring script
    local monitor_script="/usr/local/bin/sutazai-backup-monitor.sh"
    cat > "$monitor_script" << 'EOF'
#!/bin/bash
# SutazAI Backup Monitor
# Checks backup health and sends alerts if issues detected

set -euo pipefail

BACKUP_DIR="/opt/sutazaiapp/backups"
LOGS_DIR="/opt/sutazaiapp/logs"
MAX_BACKUP_AGE_HOURS=25  # 25 hours for daily backups
ALERT_EMAIL="${BACKUP_NOTIFICATION_EMAIL:-}"
ALERT_WEBHOOK="${BACKUP_SLACK_WEBHOOK:-}"

# Check if recent backups exist
check_backup_freshness() {
    local latest_backup
    latest_backup=$(find "$BACKUP_DIR" -name "master_backup_report_*.json" -mtime -1 2>/dev/null | head -1)
    
    if [ -z "$latest_backup" ]; then
        echo "ERROR: No recent master backup reports found"
        return 1
    fi
    
    local backup_age_hours
    backup_age_hours=$(( ($(date +%s) - $(stat -c %Y "$latest_backup")) / 3600 ))
    
    if [ $backup_age_hours -gt $MAX_BACKUP_AGE_HOURS ]; then
        echo "ERROR: Latest backup is $backup_age_hours hours old (max: $MAX_BACKUP_AGE_HOURS)"
        return 1
    fi
    
    echo "OK: Latest backup is $backup_age_hours hours old"
    return 0
}

# Check backup integrity
check_backup_integrity() {
    local failed_backups=0
    
    # Check recent backup reports for failures
    local latest_report
    latest_report=$(find "$BACKUP_DIR" -name "master_backup_report_*.json" -mtime -1 2>/dev/null | head -1)
    
    if [ -n "$latest_report" ] && command -v jq > /dev/null 2>&1; then
        local status
        status=$(jq -r '.status' "$latest_report" 2>/dev/null || echo "UNKNOWN")
        
        if [ "$status" != "SUCCESS" ]; then
            echo "ERROR: Latest backup status is $status"
            failed_backups=$((failed_backups + 1))
        fi
        
        local failed_count
        failed_count=$(jq -r '.summary.failed_backups' "$latest_report" 2>/dev/null || echo "0")
        
        if [ "$failed_count" -gt 0 ]; then
            echo "ERROR: $failed_count backup jobs failed"
            failed_backups=$((failed_backups + failed_count))
        fi
    else
        echo "WARNING: Cannot parse backup report or jq not available"
    fi
    
    if [ $failed_backups -eq 0 ]; then
        echo "OK: Backup integrity checks passed"
        return 0
    else
        return 1
    fi
}

# Send alert
send_alert() {
    local message="$1"
    
    if [ -n "$ALERT_EMAIL" ] && command -v mail > /dev/null 2>&1; then
        echo -e "$message" | mail -s "SutazAI Backup Alert" "$ALERT_EMAIL" 2>/dev/null || true
    fi
    
    if [ -n "$ALERT_WEBHOOK" ] && command -v curl > /dev/null 2>&1; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 SutazAI Backup Alert\\n\\n$message\"}" \
            "$ALERT_WEBHOOK" 2>/dev/null || true
    fi
}

# Main monitoring function
main() {
    local errors=""
    
    if ! check_backup_freshness; then
        errors="$errors\n- Backup freshness check failed"
    fi
    
    if ! check_backup_integrity; then
        errors="$errors\n- Backup integrity check failed"
    fi
    
    if [ -n "$errors" ]; then
        local alert_message="SutazAI backup monitoring detected issues:\n$errors\n\nPlease check the backup system immediately."
        echo -e "$alert_message"
        send_alert "$alert_message"
        exit 1
    else
        echo "All backup monitoring checks passed"
        exit 0
    fi
}

main "$@"
EOF
    
    chmod +x "$monitor_script"
    log "Created backup monitor script: $monitor_script"
    
    # Add monitoring to cron (run every 4 hours)
    local monitor_cron="0 */4 * * * $monitor_script >/dev/null 2>&1"
    
    if ! crontab -l 2>/dev/null | grep -q "sutazai-backup-monitor"; then
        (crontab -l 2>/dev/null; echo "# SutazAI Backup Monitoring"; echo "$monitor_cron") | crontab -
        log "Added backup monitoring to cron"
    fi
}

# Display scheduling status
show_scheduling_status() {
    local method="$1"
    
    log "Backup scheduling status:"
    
    case "$method" in
        "cron")
            if crontab -l 2>/dev/null | grep -q "sutazai-backup"; then
                log "✓ Cron backup job is configured:"
                crontab -l 2>/dev/null | grep "sutazai-backup"
            else
                log "✗ No cron backup job found"
            fi
            ;;
        "systemd")
            if systemctl is-enabled sutazai-backup.timer > /dev/null 2>&1; then
                log "✓ Systemd backup timer is enabled:"
                systemctl status sutazai-backup.timer --no-pager -l
            else
                log "✗ Systemd backup timer is not enabled"
            fi
            ;;
    esac
    
    # Check for backup monitoring
    if crontab -l 2>/dev/null | grep -q "sutazai-backup-monitor"; then
        log "✓ Backup monitoring is configured"
    else
        log "✗ Backup monitoring is not configured"
    fi
    
    # Check for log rotation
    if [ -f "/etc/logrotate.d/sutazai-backup" ]; then
        log "✓ Log rotation is configured"
    else
        log "✗ Log rotation is not configured"
    fi
}

# Create backup scheduler documentation
create_documentation() {
    log "Creating backup scheduler documentation..."
    
    local doc_file="/opt/sutazaiapp/docs/BACKUP_SCHEDULER.md"
    mkdir -p "$(dirname "$doc_file")"
    
    cat > "$doc_file" << EOF
# SutazAI Backup Scheduler Documentation

## Overview

The SutazAI backup system provides automated, reliable database backups with comprehensive monitoring and alerting capabilities.

## Backup Schedule

- **Schedule Type**: $SCHEDULE_TYPE
- **Scheduling Method**: $(detect_scheduling_method)

## Configuration

### Databases Backed Up
- PostgreSQL (sutazai-postgres)
- Redis (sutazai-redis)
- Neo4j (sutazai-neo4j)
- Qdrant (sutazai-qdrant)
- ChromaDB (sutazai-chromadb)
- FAISS (sutazai-faiss)

### Backup Storage
- **Location**: $BACKUP_ROOT
- **Retention**: 30 days (configurable)
- **Compression**: All backups are compressed with gzip

### RTO/RPO Targets
- **RTO (Recovery Time Objective)**: < 30 minutes
- **RPO (Recovery Point Objective)**: < 1 hour data loss

## Scripts and Components

### Backup Scripts
- \`master-backup.sh\` - Main orchestration script
- \`backup-redis.sh\` - Redis-specific backup (RDB + AOF)
- \`backup-neo4j.sh\` - Neo4j graph database backup
- \`backup-vector-databases.sh\` - Vector databases backup

### Restoration Scripts
- \`restore-databases.sh\` - Database restoration with safety checks
- Supports individual database restore or full system restore

### Testing and Monitoring
- \`test-backup-system.sh\` - Comprehensive backup system testing
- \`sutazai-backup-monitor.sh\` - Automated monitoring and alerting

## Manual Operations

### Run Backup Manually
\`\`\`bash
sudo /opt/sutazaiapp/scripts/maintenance/master-backup.sh
\`\`\`

### Test Backup System
\`\`\`bash
sudo /opt/sutazaiapp/scripts/maintenance/test-backup-system.sh true  # Dry run
sudo /opt/sutazaiapp/scripts/maintenance/test-backup-system.sh false # Real test
\`\`\`

### Restore Database
\`\`\`bash
# Restore specific database from latest backup
sudo /opt/sutazaiapp/scripts/maintenance/restore-databases.sh postgres

# Restore from specific backup file
sudo /opt/sutazaiapp/scripts/maintenance/restore-databases.sh redis /path/to/backup.rdb.gz

# Restore all databases
sudo /opt/sutazaiapp/scripts/maintenance/restore-databases.sh all
\`\`\`

## Monitoring and Alerts

### Notification Channels
- Email: ${NOTIFICATION_EMAIL:-Not configured}
- Slack: ${SLACK_WEBHOOK:+Configured}${SLACK_WEBHOOK:-Not configured}
- Healthcheck URL: ${HEALTHCHECK_URL:-Not configured}

### Monitoring Frequency
- Backup monitoring runs every 4 hours
- Alerts sent on backup failures or missing backups

## Log Files

- **Backup logs**: \`/opt/sutazaiapp/logs/backup_*.log\`
- **Restoration logs**: \`/opt/sutazaiapp/logs/restoration_*.log\`
- **Test logs**: \`/opt/sutazaiapp/logs/backup-tests/\`
- **Scheduled backup logs**: \`/opt/sutazaiapp/logs/scheduled_backup_*.log\`

## Troubleshooting

### Check Backup Status
\`\`\`bash
# View recent backup logs
sudo ls -la /opt/sutazaiapp/logs/backup_*

# Check backup reports
sudo ls -la /opt/sutazaiapp/backups/master_backup_report_*

# Test system manually
sudo /opt/sutazaiapp/scripts/maintenance/test-backup-system.sh true
\`\`\`

### Common Issues

1. **Insufficient disk space**: Ensure at least 5GB free space in backup directory
2. **Database connection errors**: Verify all database containers are running
3. **Permission errors**: Backup scripts must run as root or with appropriate permissions
4. **Docker issues**: Ensure Docker daemon is running and accessible

### Emergency Procedures

1. **Immediate backup**: Run master backup script manually
2. **System restore**: Use restore-databases.sh with appropriate backup files
3. **Backup verification**: Run test-backup-system.sh to validate system integrity

## Configuration Files

- **Cron**: \`/var/spool/cron/crontabs/root\` (if using cron)
- **Systemd**: \`/etc/systemd/system/sutazai-backup.*\` (if using systemd)
- **Log rotation**: \`/etc/logrotate.d/sutazai-backup\`

## Security Considerations

- All backup files are compressed to save space
- Pre-restoration backups are created for safety
- Backup integrity is verified after each operation
- Access to backup files should be restricted to authorized personnel

---

**Generated**: $(date -Iseconds)
**Setup Log**: $LOG_FILE
EOF
    
    log "Documentation created: $doc_file"
}

# Main execution
main() {
    log "========================================="
    log "SutazAI Backup Scheduler Setup"
    log "Schedule Type: $SCHEDULE_TYPE"
    log "Use Systemd: $USE_SYSTEMD"
    log "========================================="
    
    # Show usage if requested
    if [ "$SCHEDULE_TYPE" = "help" ] || [ "$SCHEDULE_TYPE" = "--help" ] || [ "$SCHEDULE_TYPE" = "-h" ]; then
        usage
        exit 0
    fi
    
    # Check prerequisites
    check_prerequisites
    
    # Detect scheduling method
    local scheduling_method
    scheduling_method=$(detect_scheduling_method)
    log "Using scheduling method: $scheduling_method"
    
    # Set up scheduling based on method
    case "$scheduling_method" in
        "systemd")
            setup_systemd_scheduling "$SCHEDULE_TYPE"
            ;;
        "cron")
            setup_cron_scheduling "$SCHEDULE_TYPE"
            ;;
        *)
            error_exit "Unknown scheduling method: $scheduling_method"
            ;;
    esac
    
    # Set up additional components
    setup_log_rotation
    setup_monitoring
    
    # Create documentation
    create_documentation
    
    # Show final status
    show_scheduling_status "$scheduling_method"
    
    log "========================================="
    log "Backup scheduler setup completed successfully"
    log "Next backup will run according to $SCHEDULE_TYPE schedule"
    log "Setup log: $LOG_FILE"
    log "Documentation: /opt/sutazaiapp/docs/BACKUP_SCHEDULER.md"
    log "========================================="
}

# Execute main function
main "$@"