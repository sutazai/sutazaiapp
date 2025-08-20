#!/bin/bash
# Real-time Resource Monitoring Script
# Created: 2025-08-19
# Purpose: Monitor system resources and alert on high usage

set -euo pipefail

# Configuration
ALERT_THRESHOLD_MEM=80
ALERT_THRESHOLD_CPU=80
CHECK_INTERVAL=30
LOG_FILE="/opt/sutazaiapp/logs/resource-monitor.log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check memory usage
check_memory() {
    local mem_percent=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
    local mem_details=$(free -h | grep Mem)
    
    if [ "$mem_percent" -gt "$ALERT_THRESHOLD_MEM" ]; then
        log_message "ALERT: High memory usage: ${mem_percent}% - $mem_details"
        
        # Show top memory consumers
        log_message "Top memory consumers:"
        ps aux --sort=-%mem | head -5 | while read line; do
            log_message "$line"
        done
        
        return 1
    fi
    
    return 0
}

# Function to check CPU usage
check_cpu() {
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print int(100 - $1)}')
    
    if [ "$cpu_usage" -gt "$ALERT_THRESHOLD_CPU" ]; then
        log_message "ALERT: High CPU usage: ${cpu_usage}%"
        
        # Show top CPU consumers
        log_message "Top CPU consumers:"
        ps aux --sort=-%cpu | head -5 | while read line; do
            log_message "$line"
        done
        
        return 1
    fi
    
    return 0
}

# Function to check Docker containers
check_docker() {
    local unhealthy_count=$(docker ps --filter health=unhealthy --format "{{.Names}}" | wc -l)
    
    if [ "$unhealthy_count" -gt 0 ]; then
        log_message "WARNING: $unhealthy_count unhealthy Docker containers:"
        docker ps --filter health=unhealthy --format "table {{.Names}}\t{{.Status}}"
    fi
    
    # Check for stopped containers that should be running
    local critical_containers=("sutazai-backend" "sutazai-frontend" "sutazai-postgres" "sutazai-redis")
    
    for container in "${critical_containers[@]}"; do
        if ! docker ps -q -f name="$container" &>/dev/null; then
            log_message "CRITICAL: Container $container is not running!"
        fi
    done
}

# Function to auto-remediate issues
auto_remediate() {
    local action_taken=false
    
    # Check if memory is critically high
    local mem_percent=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
    
    if [ "$mem_percent" -gt 90 ]; then
        log_message "CRITICAL: Memory at ${mem_percent}%, initiating auto-remediation..."
        
        # Kill TypeScript servers
        if pgrep -f tsserver &>/dev/null; then
            log_message "Killing TypeScript servers..."
            pkill -f tsserver
            action_taken=true
        fi
        
        # Clean Docker
        log_message "Cleaning Docker resources..."
        docker system prune -f --volumes &>/dev/null
        action_taken=true
        
        # Clear caches
        sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
        action_taken=true
    fi
    
    if [ "$action_taken" = true ]; then
        log_message "Auto-remediation completed"
        sleep 5
    fi
}

# Function to generate status summary
status_summary() {
    echo "=== System Resource Status ==="
    echo "Time: $(date)"
    echo ""
    echo "Memory Usage:"
    free -h
    echo ""
    echo "CPU Load:"
    uptime
    echo ""
    echo "Docker Containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Size}}" | head -10
    echo ""
    echo "Disk Usage:"
    df -h | grep -E "^/dev|Filesystem"
    echo "============================="
}

# Main monitoring loop
main() {
    log_message "Starting resource monitoring (PID: $$)"
    log_message "Thresholds - Memory: ${ALERT_THRESHOLD_MEM}%, CPU: ${ALERT_THRESHOLD_CPU}%"
    
    while true; do
        # Check resources
        HIGH_RESOURCE=false
        
        if ! check_memory; then
            HIGH_RESOURCE=true
        fi
        
        if ! check_cpu; then
            HIGH_RESOURCE=true
        fi
        
        # Check Docker health
        check_docker
        
        # Auto-remediate if needed
        if [ "$HIGH_RESOURCE" = true ]; then
            auto_remediate
        fi
        
        # Print status every 5 minutes
        if [ $(($(date +%s) % 300)) -lt "$CHECK_INTERVAL" ]; then
            status_summary | tee -a "$LOG_FILE"
        fi
        
        # Wait before next check
        sleep "$CHECK_INTERVAL"
    done
}

# Handle signals gracefully
trap "log_message 'Monitoring stopped'; exit 0" SIGINT SIGTERM

# Run in background if requested
if [ "${1:-}" = "--daemon" ]; then
    log_message "Starting in daemon mode..."
    nohup "$0" > /dev/null 2>&1 &
    echo "Resource monitor started with PID: $!"
else
    main
fi