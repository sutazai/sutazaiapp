#!/bin/bash

# =========================================================
# BASH PROCESS MANAGER
# A robust script to detect and manage high CPU consuming bash processes
# =========================================================

# Configuration
LOG_DIR="/opt/sutazaiapp/logs"
LOG_FILE="${LOG_DIR}/bash_process_manager.log"
MAX_CPU_PER_PROCESS=20      # Maximum CPU% allowed per bash process
MAX_BASH_PROCESSES=5        # Maximum number of bash processes allowed from same parent
CHECK_INTERVAL=15           # Seconds between checks
MAX_LIFETIME=3600           # Maximum lifetime in seconds for a bash process
PARENT_WHITELIST=(
  "sshd"                    # Don't kill SSH sessions
  "systemd"                 # Don't kill system services
)
PROCESS_WHITELIST=(
  "root 1"                  # Don't kill init
  "systemd"                 # Don't kill systemd services
  "sshd"                    # Don't kill SSH server
  "bash_process_manager.sh" # Don't kill self
)

# Create log directory if not exists
mkdir -p "$LOG_DIR"

# Logging function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "==== Starting Bash Process Manager ===="
log "Configuration: MAX_CPU=$MAX_CPU_PER_PROCESS%, MAX_PROCESSES=$MAX_BASH_PROCESSES, CHECK_INTERVAL=$CHECK_INTERVAL seconds, MAX_LIFETIME=$MAX_LIFETIME seconds"

# Function to check if a process should be whitelisted
is_whitelisted() {
  local pid=$1
  local cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr -d '\0')
  local process_info="$USER $pid $cmdline"
  
  # Check against process whitelist
  for pattern in "${PROCESS_WHITELIST[@]}"; do
    if [[ "$process_info" == *"$pattern"* ]]; then
      return 0  # Process is whitelisted
    fi
  done
  
  # Check if parent is whitelisted
  local ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  if [[ -n "$ppid" && "$ppid" != "1" ]]; then
    local parent_cmd=$(ps -p "$ppid" -o comm= 2>/dev/null)
    for pattern in "${PARENT_WHITELIST[@]}"; do
      if [[ "$parent_cmd" == *"$pattern"* ]]; then
        return 0  # Parent is whitelisted
      fi
    done
  fi
  
  return 1  # Not whitelisted
}

# Function to get process start time in seconds since epoch
get_process_start_time() {
  local pid=$1
  # Get process start time from stat
  stat -c %Y /proc/$pid 2>/dev/null || echo 0
}

# Function to calculate process lifetime in seconds
get_process_lifetime() {
  local pid=$1
  local start_time=$(get_process_start_time "$pid")
  local current_time=$(date +%s)
  echo $((current_time - start_time))
}

# Main monitoring loop
while true; do
  log "Scanning for high CPU bash processes..."
  
  # Get all bash processes and their information
  BASH_PROCESSES=$(ps -eo pid,ppid,pcpu,etimes,args --sort=-pcpu | grep -w "bash" | grep -v "grep" | grep -v "bash_process_manager.sh")
  
  # Count bash processes by parent PID
  declare -A parent_count
  while read -r pid ppid cpu_usage uptime cmd; do
    # Skip if not a bash process or empty line
    [[ -z "$pid" || "$cmd" != *"bash"* ]] && continue
    
    # Increment count for this parent
    parent_count[$ppid]=$((${parent_count[$ppid]:-0} + 1))
    
  done <<< "$BASH_PROCESSES"
  
  # Process each bash process
  while read -r pid ppid cpu_usage uptime cmd; do
    # Skip if empty line
    [[ -z "$pid" ]] && continue
    
    # Skip non-bash processes 
    [[ "$cmd" != *"bash"* ]] && continue
    
    # Skip if process no longer exists
    if [[ ! -e "/proc/$pid" ]]; then
      continue
    fi
    
    # Calculate lifetime
    lifetime=$(get_process_lifetime "$pid")
    
    # Check if whitelisted
    if is_whitelisted "$pid"; then
      log "WHITELISTED: PID $pid (PPID $ppid, CPU ${cpu_usage}%, Lifetime ${lifetime}s) - $cmd"
      continue
    fi
    
    # Conditions to terminate:
    killed=false
    kill_reason=""
    
    # 1. CPU usage exceeds threshold
    if (( $(echo "$cpu_usage > $MAX_CPU_PER_PROCESS" | bc -l) )); then
      kill_reason="High CPU (${cpu_usage}% > ${MAX_CPU_PER_PROCESS}%)"
      killed=true
    fi
    
    # 2. Too many processes from same parent
    if [[ ${parent_count[$ppid]:-0} -gt $MAX_BASH_PROCESSES ]]; then
      if [[ -z "$kill_reason" ]]; then
        kill_reason="Too many processes from same parent (${parent_count[$ppid]} > $MAX_BASH_PROCESSES)"
      else
        kill_reason="$kill_reason, Too many processes from same parent (${parent_count[$ppid]} > $MAX_BASH_PROCESSES)"
      fi
      killed=true
    fi
    
    # 3. Exceeded maximum lifetime
    if [[ $lifetime -gt $MAX_LIFETIME ]]; then
      if [[ -z "$kill_reason" ]]; then
        kill_reason="Exceeded maximum lifetime (${lifetime}s > ${MAX_LIFETIME}s)"
      else
        kill_reason="$kill_reason, Exceeded maximum lifetime (${lifetime}s > ${MAX_LIFETIME}s)"
      fi
      killed=true
    fi
    
    # Take action
    if [[ "$killed" == "true" ]]; then
      log "TERMINATING: PID $pid (PPID $ppid, CPU ${cpu_usage}%, Lifetime ${lifetime}s) - Reason: $kill_reason"
      # Try graceful termination first
      kill -15 "$pid" >/dev/null 2>&1
      # Wait short time and force kill if still running
      sleep 1
      if [[ -e "/proc/$pid" ]]; then
        log "FORCE KILLING: PID $pid still running, sending SIGKILL"
        kill -9 "$pid" >/dev/null 2>&1
      fi
    fi
    
  done <<< "$BASH_PROCESSES"
  
  # Get count of terminated processes
  orig_bash_count=$(echo "$BASH_PROCESSES" | grep -c "bash")
  current_bash_count=$(ps -ef | grep -c "bash")
  
  log "Monitoring complete. Found $orig_bash_count bash processes, current count: $current_bash_count. Sleeping for $CHECK_INTERVAL seconds..."
  sleep "$CHECK_INTERVAL"
done 