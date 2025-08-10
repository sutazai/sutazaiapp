#!/bin/bash

# Strict error handling
set -euo pipefail


# Kill any existing cpulimit processes

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

pkill -f cpulimit

# Find all bash processes using more than 50% CPU
high_cpu_pids=$(ps aux | grep bash | grep -v grep | awk '{if($3 > 50) print $2}')

if [ -z "$high_cpu_pids" ]; then
    echo "No high-CPU bash processes found"
    exit 0
fi

# Limit each high-consuming bash process to 50% CPU
for pid in $high_cpu_pids; do
    if ps -p $pid > /dev/null; then
        cpulimit -p $pid -l 50 &
        echo "Limited PID $pid to 50% CPU"
    else
        echo "PID $pid no longer exists"
    fi
done

echo "CPU limits have been applied to high-CPU bash processes"

# Monitor the CPU usage
echo "Current CPU usage of bash processes:"
ps aux | grep bash | grep -v grep | awk '{if($3 > 0) print $2, $3, $11}' 