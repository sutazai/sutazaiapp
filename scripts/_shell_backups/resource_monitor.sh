#!/bin/bash

# Resource thresholds
CPU_THRESHOLD=90
MEMORY_THRESHOLD=90
DISK_THRESHOLD=90

monitor_resources() {
    while true; do
        # Convert CPU usage to integer by rounding
        local cpu=$(top -bn1 | grep "Cpu(s)" | awk '{printf("%.0f"), $2 + $4}')
        local memory=$(free | awk '/Mem/{printf("%.0f"), $3/$2*100}')
        local disk=$(df / | awk 'END{print $5}' | tr -d '%')
        
        if (( cpu > CPU_THRESHOLD )); then
            log_warn "High CPU usage: ${cpu}%"
            throttle_processes
        fi
        
        if (( memory > MEMORY_THRESHOLD )); then
            log_warn "High memory usage: ${memory}%"
            free_memory
        fi
        
        if (( disk > DISK_THRESHOLD )); then
            log_warn "High disk usage: ${disk}%"
            cleanup_disk
        fi
        
        sleep 10
    done
}

throttle_processes() {
    # Identify top CPU consuming processes
    local top_processes=$(ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | head -n 6 | tail -n 5)
    log_warn "Throttling top processes:\n$top_processes"
    
    # Reduce priority of top processes
    echo "$top_processes" | awk '{print $1}' | xargs -I{} renice +10 -p {}
}

free_memory() {
    # Clear page cache, dentries and inodes
    sudo sync
    sudo sysctl -w vm.drop_caches=3 >/dev/null
    log_info "Freed memory by clearing caches"
}

cleanup_disk() {
    # Remove old logs and temporary files
    find /var/log -type f -name "*.log" -mtime +7 -exec rm -f {} \;
    log_info "Cleaned up disk space by removing old logs"
}

start_resource_monitor() {
    # Start monitoring CPU, memory, disk usage
    monitor_resources &
}

stop_resource_monitor() {
    # Stop monitoring and generate report
    pkill -f "monitor_resources"
}

monitor_resources 