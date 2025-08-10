#!/bin/bash

# Strict error handling
set -euo pipefail


# Script to monitor and manage high CPU usage processes

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

LOG_FILE="/opt/sutazaiapp/logs/cpu_monitor.log"
THRESHOLD=50  # CPU usage percentage threshold
MAX_RUNTIME=3600  # Maximum runtime in seconds for a process (1 hour)
WHITELIST=(  # Processes that should not be terminated
  "systemd"
  "sshd"
  "rsync"  # Don't kill the sync process
)

# Ensure log directory exists
mkdir -p "$(dirname $LOG_FILE)"

log() {
  echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $1" >> "$LOG_FILE"
}

is_whitelisted() {
  local process_name="$1"
  for item in "${WHITELIST[@]}"; do
    if [[ "$process_name" == *"$item"* ]]; then
      return 0  # Process is whitelisted
    fi
  done
  return 1  # Process is not whitelisted
}

log "Starting CPU monitoring service"

# Set this script to a low priority
renice 19 -p $$ > /dev/null 2>&1
ionice -c 3 -p $$ > /dev/null 2>&1

# Check if another instance is already running
if pgrep -f "$(basename $0)" | grep -v $$ > /dev/null; then
  log "Another instance is already running. Exiting."
  exit 0
fi

# Timeout mechanism to prevent infinite loops
LOOP_TIMEOUT=${LOOP_TIMEOUT:-300}  # 5 minute default timeout
loop_start=$(date +%s)
while true; do
  # Get list of high CPU processes (more efficient than repeated top calls)
  HIGH_CPU_PROCESSES=$(ps -eo pid,pcpu,etimes,cmd --sort=-%cpu | awk -v threshold="$THRESHOLD" '$2 > threshold {print $1","$2","$3","$4}' | head -n 10)
  
  if [ -n "$HIGH_CPU_PROCESSES" ]; then
    log "Detected high CPU usage processes:"
    
    echo "$HIGH_CPU_PROCESSES" | while IFS=',' read -r PID CPU_USAGE RUNTIME CMD; do
      log "PID: $PID, CPU: $CPU_USAGE%, Runtime: $RUNTIME seconds, Command: $CMD"
      
      # Check if process is whitelisted
      if ! is_whitelisted "$CMD"; then
        # Check if process has been running too long with high CPU
        if [ "$RUNTIME" -gt "$MAX_RUNTIME" ]; then
          log "Terminating runaway process $PID ($CMD) - CPU: $CPU_USAGE%, Runtime: $RUNTIME seconds"
          kill -15 $PID
          sleep 2
          # Force kill if still running
          if kill -0 $PID 2>/dev/null; then
            log "Process $PID did not terminate gracefully, using force kill"
            kill -9 $PID
          fi
        else
          # Try to reduce the priority of the process
          log "Reducing priority of process $PID ($CMD)"
          renice 19 -p $PID > /dev/null 2>&1
          ionice -c 3 -p $PID > /dev/null 2>&1
        fi
      else
        log "Process $PID ($CMD) is whitelisted, not modifying"
      fi
    # Check for timeout
    current_time=$(date +%s)
    if [[ $((current_time - loop_start)) -gt $LOOP_TIMEOUT ]]; then
        echo 'Loop timeout reached after ${LOOP_TIMEOUT}s, exiting...' >&2
        break
    fi

    done
  else
    log "No high CPU usage processes detected above $THRESHOLD%"
  fi
  
  # Sleep for 5 minutes before next check
  sleep 300
done 