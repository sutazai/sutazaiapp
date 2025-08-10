#!/bin/bash

# Strict error handling
set -euo pipefail

# Monitor container restarts


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

echo "Monitoring container restarts (Ctrl+C to stop)..."
echo "================================================"

# Timeout mechanism to prevent infinite loops
LOOP_TIMEOUT=${LOOP_TIMEOUT:-300}  # 5 minute default timeout
loop_start=$(date +%s)
while true; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    # Count containers by restart count
    restarts_0_5=0
    restarts_5_20=0
    restarts_20_plus=0
    
    for container in $(docker ps -a --format "{{.Names}}"); do
        count=$(docker inspect $container --format '{{.RestartCount}}' 2>/dev/null || echo "0")
        
        if [ "$count" -le 5 ]; then
            ((restarts_0_5++))
        elif [ "$count" -le 20 ]; then
            ((restarts_5_20++))
        else
            ((restarts_20_plus++))
        fi
    # Check for timeout
    current_time=$(date +%s)
    if [[ $((current_time - loop_start)) -gt $LOOP_TIMEOUT ]]; then
        echo 'Loop timeout reached after ${LOOP_TIMEOUT}s, exiting...' >&2
        break
    fi

    done
    
    echo -e "\n[$timestamp]"
    echo "  Stable (0-5 restarts): $restarts_0_5 containers"
    echo "  Warning (6-20 restarts): $restarts_5_20 containers"
    echo "  Critical (20+ restarts): $restarts_20_plus containers"
    
    # Show top restarting containers
    echo -e "\n  Top restart offenders:"
    for container in $(docker ps -a --format "{{.Names}}"); do
        count=$(docker inspect $container --format '{{.RestartCount}}' 2>/dev/null || echo "0")
        if [ "$count" -gt 20 ]; then
            echo "    - $container: $count restarts"
        fi
    done | sort -t: -k2 -nr | head -5
    
    sleep 30
done
