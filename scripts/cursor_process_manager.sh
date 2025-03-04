#!/bin/bash

# =========================================================
# CURSOR PROCESS MANAGER - ENHANCED
# A script to detect and manage ANY high CPU consuming processes on the system
# =========================================================

# Configuration
LOG_DIR="/opt/sutazaiapp/logs"
LOG_FILE="${LOG_DIR}/cursor_process_manager.log"
MAX_CPU_PER_PROCESS=20      # Maximum CPU% allowed per process
CHECK_INTERVAL=5            # Seconds between checks
MAX_LIFETIME=600            # Maximum lifetime in seconds for a bash process
# Whitelisted processes that should never be killed
WHITELIST=("systemd" "sshd" "cursor-process-manager")

# Clear log file on startup to prevent it from growing too large
echo "" > "$LOG_FILE"

# Create log directory if not exists
mkdir -p "$LOG_DIR"

# Logging function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "==== ENHANCED PROCESS MANAGER STARTED ===="
log "Configuration: MAX_CPU=$MAX_CPU_PER_PROCESS%, CHECK_INTERVAL=$CHECK_INTERVAL seconds, MAX_LIFETIME=$MAX_LIFETIME seconds"

# Main monitoring loop
while true; do
  log "Scanning for high CPU processes..."
  
  # Get all processes sorted by CPU usage (top 20)
  HIGH_CPU_PROCESSES=$(ps -eo pid,ppid,pcpu,etimes,comm --sort=-pcpu | head -n 21 | tail -n +2)
  
  # Count high CPU processes
  PROCESS_COUNT=$(echo "$HIGH_CPU_PROCESSES" | grep -v "^$" | wc -l)
  log "Found $PROCESS_COUNT high CPU processes to check"
  
  # Process each line
  while read -r pid ppid cpu_usage uptime cmd; do
    # Skip if empty line
    [[ -z "$pid" ]] && continue
    
    # Skip if process no longer exists
    if [[ ! -e "/proc/$pid" ]]; then
      continue
    fi
    
    # Check if process is in whitelist
    SKIP=0
    for wl in "${WHITELIST[@]}"; do
      if [[ "$cmd" == "$wl" || "$cmd" == *"$wl"* ]]; then
        log "WHITELISTED: PID $pid ($cmd) using ${cpu_usage}% CPU"
        SKIP=1
        break
      fi
    done
    [[ $SKIP -eq 1 ]] && continue
    
    # Process any high CPU python processes immediately
    if [[ "$cmd" == *"python"* && $(echo "$cpu_usage > 50" | bc -l) -eq 1 ]]; then
      log "KILLING HIGH CPU PYTHON: PID $pid (${cpu_usage}%, Uptime ${uptime}s, Command: $cmd)"
      kill -15 "$pid" >/dev/null 2>&1
      sleep 1
      # If still running, force kill
      if [[ -e "/proc/$pid" ]]; then
        kill -9 "$pid" >/dev/null 2>&1
        log "Force killed Python process: $pid"
      fi
      continue
    fi
    
    # Process bash processes
    if [[ "$cmd" == "bash" || "$cmd" == *"bash"* ]]; then
      # Check full command to see if it's a legitimate terminal session
      FULL_CMD=$(ps -p "$pid" -o cmd= 2>/dev/null)
      
      # If it's a terminal session with a user, we should be more careful
      if [[ "$FULL_CMD" == *"pts/"* && "$FULL_CMD" == *"-l"* ]]; then
        # Only kill if really high CPU or very long-lived
        if (( $(echo "$cpu_usage > 70" | bc -l) || uptime > 3600 )); then
          log "KILLING HIGH CPU BASH TERMINAL: PID $pid (${cpu_usage}%, Uptime ${uptime}s)"
          kill -15 "$pid" >/dev/null 2>&1
        fi
      else
        # Non-terminal bash, kill if over threshold
        if (( $(echo "$cpu_usage > $MAX_CPU_PER_PROCESS" | bc -l) )); then
          log "KILLING HIGH CPU BASH PROCESS: PID $pid (${cpu_usage}%, Uptime ${uptime}s)"
          kill -15 "$pid" >/dev/null 2>&1
          sleep 1
          # If still running, force kill
          if [[ -e "/proc/$pid" ]]; then
            kill -9 "$pid" >/dev/null 2>&1
            log "Force killed bash process: $pid"
          fi
        fi
      fi
      continue
    fi
    
    # Any other high CPU process (over 80%)
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
      log "KILLING VERY HIGH CPU PROCESS: PID $pid (${cpu_usage}%, Uptime ${uptime}s, Command: $cmd)"
      kill -15 "$pid" >/dev/null 2>&1
      sleep 1
      # If still running, force kill
      if [[ -e "/proc/$pid" ]]; then
        kill -9 "$pid" >/dev/null 2>&1
        log "Force killed generic high CPU process: $pid"
      fi
    fi
    
  done <<< "$HIGH_CPU_PROCESSES"
  
  # Get current CPU usage
  CPU_IDLE=$(top -bn1 | grep "Cpu(s)" | awk '{print $8}')
  CPU_USAGE=$(echo "100 - $CPU_IDLE" | bc)
  
  log "Monitoring complete. Current CPU usage: ${CPU_USAGE}%. Sleeping for $CHECK_INTERVAL seconds..."
  sleep "$CHECK_INTERVAL"
done
