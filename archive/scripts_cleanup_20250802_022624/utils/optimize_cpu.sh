#!/bin/bash
# CPU optimization script for SutazAI sync system

# Source configuration
source /opt/sutazaiapp/scripts/config/sync_config.sh

# Log file
LOG_FILE="$PROJECT_ROOT/logs/cpu_optimization.log"

# Ensure log directory exists
mkdir -p "$(dirname $LOG_FILE)"

# Configure logging
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $1"
}

log "Starting CPU usage optimization"

# Function to identify high CPU processes
identify_high_cpu_processes() {
    log "Identifying processes with high CPU usage"
    
    # Get top 5 CPU-intensive processes
    HIGH_CPU_PROCESSES=$(ps aux --sort=-%cpu | head -n 6)
    log "Top CPU-consuming processes:"
    log "$HIGH_CPU_PROCESSES"
    
    # Check for specifically sync-related processes
    SYNC_PROCESSES=$(ps aux | grep -E 'sync_monitor|rsync' | grep -v grep)
    if [ -n "$SYNC_PROCESSES" ]; then
        log "Sync-related processes:"
        log "$SYNC_PROCESSES"
    fi
}

# Function to optimize sync processes
optimize_sync_processes() {
    log "Optimizing sync processes"
    
    # Check for running rsync processes
    RSYNC_PROCESSES=$(pgrep rsync)
    if [ -n "$RSYNC_PROCESSES" ]; then
        log "Active rsync processes found, adjusting priority"
        echo "$RSYNC_PROCESSES" | while read PID; do
            renice +10 -p $PID 2>/dev/null
            log "Reduced priority of rsync process $PID"
        done
    fi
    
    # Check for multiple sync_monitor processes
    MONITOR_COUNT=$(pgrep -f sync_monitor.sh | wc -l)
    if [ "$MONITOR_COUNT" -gt 1 ]; then
        log "Multiple sync_monitor processes found ($MONITOR_COUNT), cleaning up"
        pkill -f sync_monitor.sh
        sleep 2
        systemctl restart sutazai-sync-monitor.service
        log "Restarted sync_monitor service"
    fi
}

# Function to optimize SSH connections
optimize_ssh_connections() {
    log "Optimizing SSH connections"
    
    # Close stale SSH connections
    STALE_SSH=$(ps aux | grep ssh | grep -E 'Timeout|Minutes' | awk '{print $2}')
    if [ -n "$STALE_SSH" ]; then
        log "Found stale SSH connections, terminating"
        echo "$STALE_SSH" | xargs kill 2>/dev/null
    fi
    
    # Clear SSH control masters
    find ~/.ssh -name 'control-*' -type s -delete
    log "Cleared SSH control sockets"
}

# Function to update sync settings for better performance
update_sync_settings() {
    log "Updating sync configuration for better performance"
    
    # Check if we need to limit bandwidth
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')
    if (( $(echo "$CPU_USAGE > 90" | bc -l) )); then
        log "CPU usage is very high (${CPU_USAGE}%), limiting bandwidth to reduce load"
        # Temporarily set bandwidth limit to 5MB/s if not already set
        if [ "$MAX_BANDWIDTH" == "0" ]; then
            sed -i 's/MAX_BANDWIDTH="0"/MAX_BANDWIDTH="5000"/' "$PROJECT_ROOT/scripts/config/sync_config.sh"
            log "Set MAX_BANDWIDTH to 5000 KB/s"
        fi
    fi
    
    # Increase sync interval if CPU is consistently high
    sed -i 's/SYNC_INTERVAL=300/SYNC_INTERVAL=600/' "$PROJECT_ROOT/scripts/config/sync_config.sh"
    log "Increased SYNC_INTERVAL to 600 seconds (10 minutes) to reduce CPU load"
}

# Function to suggest further optimizations
suggest_optimizations() {
    log "Suggesting additional optimizations"
    
    # Check disk I/O which could affect performance
    DISK_IO=$(iostat -x | grep -A 1 'avg-cpu' -B 1)
    log "Current disk I/O statistics:"
    log "$DISK_IO"
    
    # Create a list of optimization recommendations
    RECOMMENDATIONS=""
    
    # Add Git optimization if Git is found
    if [ -d "$PROJECT_ROOT/.git" ]; then
        RECOMMENDATIONS="$RECOMMENDATIONS\n- Consider optimizing Git repository (git gc --aggressive)"
    fi
    
    # Recommend logrotate if logs are large
    LOG_SIZE=$(du -sh "$PROJECT_ROOT/logs" 2>/dev/null | cut -f1)
    if [ -n "$LOG_SIZE" ] && [[ "$LOG_SIZE" == *"G"* ]]; then
        RECOMMENDATIONS="$RECOMMENDATIONS\n- Set up log rotation to manage large log files ($LOG_SIZE)"
    fi
    
    # Add general optimization tips
    RECOMMENDATIONS="$RECOMMENDATIONS\n- Consider using lsyncd for more efficient sync monitoring"
    RECOMMENDATIONS="$RECOMMENDATIONS\n- Implement incremental sync for large data transfers"
    RECOMMENDATIONS="$RECOMMENDATIONS\n- Check for any cron jobs that might overlap with sync operations"
    
    if [ -n "$RECOMMENDATIONS" ]; then
        log "Recommendations for further optimization:"
        log "$RECOMMENDATIONS"
    fi
}

# Run optimization functions
identify_high_cpu_processes
optimize_sync_processes
optimize_ssh_connections
update_sync_settings
suggest_optimizations

log "CPU usage optimization completed"
log "Please restart the sync monitor service: systemctl restart sutazai-sync-monitor.service" 