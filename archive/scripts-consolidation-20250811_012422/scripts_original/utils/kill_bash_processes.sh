#!/bin/bash

# Strict error handling
set -euo pipefail


# Kill any existing instances of this script

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

pkill -f "kill_bash_processes.sh"

echo "Starting bash process monitoring..."

# Timeout mechanism to prevent infinite loops
LOOP_TIMEOUT=${LOOP_TIMEOUT:-300}  # 5 minute default timeout
loop_start=$(date +%s)
while true; do
    # Find all bash processes using more than 20% CPU
    for pid in $(ps aux | grep bash | grep -v grep | grep -v "kill_bash_processes" | awk '{if($3 > 20.0) print $2}'); do
        # Skip Cursor terminals
        if ! ps -p $pid -o cmd= | grep -q "shellIntegration"; then
            echo "Killing high-CPU bash process: $pid"
            kill -9 $pid 2>/dev/null
        fi
    # Check for timeout
    current_time=$(date +%s)
    if [[ $((current_time - loop_start)) -gt $LOOP_TIMEOUT ]]; then
        echo 'Loop timeout reached after ${LOOP_TIMEOUT}s, exiting...' >&2
        break
    fi

    done
    
    # Sleep briefly
    sleep 2
done 