#!/bin/bash

# CPU Monitor - Monitors CPU usage and manages high-consuming processes
# Usage: bash cpu_monitor.sh [CPU_LIMIT] [INTERVAL] [ACTION]
# Example: bash cpu_monitor.sh 80 5 renice

# Default parameters
CPU_LIMIT=${1:-80}    # Default CPU usage limit percentage
INTERVAL=${2:-30}     # Default check interval in seconds
ACTION=${3:-renice}   # Default action: renice, kill

# Set up logging
LOG_DIR="/opt/sutazaiapp/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/cpu_monitor.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log "Starting CPU monitor (Limit: ${CPU_LIMIT}%, Interval: ${INTERVAL}s, Action: ${ACTION})"

# Define a list of critical processes that should never be killed
CRITICAL_PROCESSES=(
    "systemd"
    "sshd"
    "docker"
    "containerd"
    "mongodb"
    "postgres"
    "mysql"
    "nginx"
    "system"
    "init"
)

# Define a list of known troublemakers
TROUBLEMAKERS=(
    "bandit"
    "comprehensive_code_check.sh"
)

is_critical() {
    local process_name="$1"
    for critical in "${CRITICAL_PROCESSES[@]}"; do
        if [[ "$process_name" == "$critical" || "$process_name" == *"$critical"* ]]; then
            return 0
        fi
    done
    return 1
}

is_troublemaker() {
    local process_name="$1"
    local cmdline="$2"
    for tm in "${TROUBLEMAKERS[@]}"; do
        if [[ "$process_name" == "$tm" || "$cmdline" == *"$tm"* ]]; then
            return 0
        fi
    done
    return 1
}

while true; do
    # Get top CPU-consuming processes
    high_cpu_processes=$(ps -eo pid,pcpu,pmem,comm,args --sort=-pcpu | head -n 20)
    
    # Extract the highest CPU process
    while IFS= read -r process_line; do
        # Skip the header line
        if [[ "$process_line" == *"PID"* ]]; then
            continue
        fi
        
        pid=$(echo "$process_line" | awk '{print $1}')
        cpu=$(echo "$process_line" | awk '{print $2}')
        mem=$(echo "$process_line" | awk '{print $3}')
        name=$(echo "$process_line" | awk '{print $4}')
        cmdline=$(echo "$process_line" | awk '{$1=$2=$3=$4=""; print $0}' | sed 's/^[ \t]*//')
        
        # Check if CPU usage exceeds the limit
        if (( $(echo "$cpu > $CPU_LIMIT" | bc -l) )); then
            # Check if process exists and is not the current script
            if [ -e "/proc/$pid" ] && [ "$pid" != "$$" ]; then
                if is_critical "$name"; then
                    log "High CPU detected (${cpu}%) in CRITICAL process: $pid - $name - Ignoring"
                elif is_troublemaker "$name" "$cmdline"; then
                    log "Detected TROUBLEMAKER: $pid - $name ($cpu% CPU, $mem% MEM) - $cmdline"
                    
                    if [ "$ACTION" == "kill" ]; then
                        log "KILLING troublemaker process: $pid - $name"
                        kill -15 "$pid"
                        sleep 2
                        # Force kill if still running
                        if [ -e "/proc/$pid" ]; then
                            log "Forced termination of process: $pid - $name"
                            kill -9 "$pid"
                        fi
                    else
                        log "RENICE troublemaker process: $pid - $name"
                        renice +19 -p "$pid" >/dev/null 2>&1
                        # Set CPU affinity to one core
                        taskset -pc 0 "$pid" >/dev/null 2>&1
                    fi
                else
                    log "High CPU detected: $pid - $name ($cpu% CPU, $mem% MEM) - $cmdline"
                    if [ "$ACTION" == "kill" ] && (( $(echo "$cpu > $((CPU_LIMIT * 2))" | bc -l) )); then
                        # Only kill non-troublemakers if they're using more than twice the CPU limit
                        log "KILLING excessive CPU process: $pid - $name"
                        kill -15 "$pid"
                    else
                        log "RENICE high CPU process: $pid - $name"
                        renice +10 -p "$pid" >/dev/null 2>&1
                    fi
                fi
            fi
        fi
    done <<< "$high_cpu_processes"
    
    # Sleep for the specified interval
    sleep "$INTERVAL"
done 