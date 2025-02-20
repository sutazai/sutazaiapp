#!/bin/bash

# System Log Rotation Check Script

log_message "=== Starting System Log Rotation Check ==="

# Check logrotate configuration
log_message "Logrotate Configuration:"
cat /etc/logrotate.conf | while read -r line; do
    log_message "$line"
done

# Verify log rotation is configured
# Check for:
# - Log file sizes
# - Rotation frequency
# - Retention policy
# - Compression settings

# Add these checks
check_log_rotation() {
    echo "🔍 Verifying log rotation configuration..." | tee -a $LOG_FILE
    
    # Check log file sizes
    local max_size=$(grep -i 'size' /etc/logrotate.conf | grep -oP '\d+[M|G]')
    if [ -z "$max_size" ]; then
        handle_error "Log rotation size not configured"
    fi
    
    # Check rotation frequency
    local frequency=$(grep -i 'daily\|weekly\|monthly' /etc/logrotate.conf)
    if [ -z "$frequency" ]; then
        handle_error "Log rotation frequency not configured"
    fi
    
    # Check retention policy
    local retention=$(grep -i 'rotate' /etc/logrotate.conf | grep -oP '\d+')
    if [ -z "$retention" ] || [ "$retention" -lt 7 ]; then
        handle_error "Insufficient log retention (minimum 7 days required)"
    fi
    
    # Check compression settings
    if ! grep -q 'compress' /etc/logrotate.conf; then
        handle_error "Log compression not enabled"
    fi
    
    echo "✅ Log rotation properly configured" | tee -a $LOG_FILE
}

# Call this function
check_log_rotation

# Automated log rotation check and enforcement
AUTO_ROTATE() {
    echo "Starting automated log rotation check..."
    
    # Check and rotate logs
    for logfile in $(find /var/log -type f -name "*.log"); do
        filesize=$(du -m $logfile | awk '{print $1}')
        if [ $filesize -gt 100 ]; then
            echo "Rotating $logfile (size: ${filesize}MB)"
            logrotate -f /etc/logrotate.conf
        fi
    done
    
    echo "Log rotation check completed at $(date)" >> /var/log/log_rotation.log
}

log_message "=== System Log Rotation Check Completed ===" 