#!/bin/bash

# System Health Monitor
system_health_monitor() {
    while true; do
        # Check CPU load
        local cpu_load=$(awk '{print $1}' < /proc/loadavg)
        if (( $(echo "$cpu_load > $(nproc)" | bc -l) )); then
            trigger_event "high_cpu_load" "$cpu_load"
        fi

        # Check memory usage
        local mem_usage=$(free | awk '/Mem/{printf("%.0f"), $3/$2*100}')
        if (( mem_usage > 90 )); then
            trigger_event "high_memory_usage" "$mem_usage"
        fi

        # Check disk space
        local disk_usage=$(df / --output=pcent | tail -1 | tr -d '%')
        if (( disk_usage > 90 )); then
            trigger_event "low_disk_space" "$disk_usage"
        fi

        sleep 60
    done
}

# Service Watchdog
service_watchdog() {
    local services=("sutazai-core" "sutazai-api" "sutazai-worker")
    while true; do
        for service in "${services[@]}"; do
            if ! systemctl is-active --quiet "$service"; then
                trigger_event "service_down" "$service"
                systemctl restart "$service"
            fi
        done
        sleep 30
    done
}

# Configuration Manager
configuration_manager() {
    # Watch for config changes
    inotifywait -m -r -e modify,create,delete /etc/sutazai | while read path action file; do
        case "$file" in
            *.conf)
                trigger_event "config_changed" "$file"
                ;;
            *.env)
                trigger_event "env_changed" "$file"
                ;;
        esac
    done
}

# Deployment Coordinator
deployment_coordinator() {
    while true; do
        if [[ -f /var/ready_for_deployment ]]; then
            trigger_event "deployment_start"
            execute_deployment
            rm -f /var/ready_for_deployment
            trigger_event "deployment_complete"
        fi
        sleep 10
    done
}

# Resource Optimizer
resource_optimizer() {
    while true; do
        # Optimize memory usage
        if (( $(free | awk '/Mem/{printf("%.0f"), $3/$2*100}') > 80 )); then
            sync
            echo 3 > /proc/sys/vm/drop_caches
        fi

        # Optimize CPU usage
        local top_processes=$(ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | head -n 6 | tail -n 5)
        echo "$top_processes" | awk '{print $1}' | xargs -I{} renice +10 -p {}

        sleep 300
    done
}

# Security Auditor
security_auditor() {
    while true; do
        # Check for world-writable files
        local world_writable=$(find / -xdev -type f -perm -0002 2>/dev/null)
        if [[ -n "$world_writable" ]]; then
            trigger_event "security_issue" "World-writable files found"
        fi

        # Check for unauthorized changes
        local changed_files=$(rpm -Va --nomtime --nosize --nomd5 --nolinkto 2>/dev/null)
        if [[ -n "$changed_files" ]]; then
            trigger_event "security_issue" "System files modified"
        fi

        sleep 3600
    done
} 