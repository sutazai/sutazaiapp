#!/bin/bash

# Script to set up cron jobs for the two-way sync system

# Set up logging
LOG_FILE="/opt/sutazaiapp/logs/setup_cron.log"
CURRENT_DATE=$(date +%Y-%m-%d\ %H:%M:%S)

# Ensure log directory exists
mkdir -p "$(dirname $LOG_FILE)"

# Configure logging
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$CURRENT_DATE] Setting up cron jobs for two-way sync system"

# Function to add a cron job
add_cron_job() {
    local SCHEDULE="$1"
    local COMMAND="$2"
    local JOB_NAME="$3"
    
    echo "[$CURRENT_DATE] Setting up $JOB_NAME cron job"
    
    # Check if the cron job already exists
    if crontab -l 2>/dev/null | grep -q "$COMMAND"; then
        echo "[$CURRENT_DATE] $JOB_NAME cron job already exists, skipping"
    else
        echo "[$CURRENT_DATE] Adding $JOB_NAME cron job"
        (crontab -l 2>/dev/null; echo "$SCHEDULE $COMMAND # $JOB_NAME") | crontab -
    fi
}

# Add hourly sync job
add_cron_job "0 * * * *" "/opt/sutazaiapp/scripts/two_way_sync.sh >> /opt/sutazaiapp/logs/sync/cron_sync_\$(date +\%Y\%m\%d).log 2>&1" "Hourly Sync"

# Add daily health check job
add_cron_job "0 0 * * *" "/opt/sutazaiapp/scripts/sync_health_check.sh >> /opt/sutazaiapp/logs/sync/health_check_\$(date +\%Y\%m\%d).log 2>&1" "Daily Health Check"

# Add weekly log rotation job
add_cron_job "0 1 * * 0" "find /opt/sutazaiapp/logs -name \"*.log\" -type f -mtime +7 -exec gzip {} \\;" "Weekly Log Rotation"

# Add monthly old log cleanup job
add_cron_job "0 2 1 * *" "find /opt/sutazaiapp/logs -name \"*.log.gz\" -type f -mtime +30 -delete" "Monthly Old Log Cleanup"

# Verify cron jobs were added
echo "[$CURRENT_DATE] Verifying cron jobs"
crontab -l | grep -E "(two_way_sync|sync_health_check|Log Rotation|Log Cleanup)"

echo "[$CURRENT_DATE] Cron jobs setup completed" 