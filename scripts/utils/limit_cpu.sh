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

# Limit main Cursor server process
cpulimit -p 1397 -l 25 &

# Limit extension host process
cpulimit -p 1429 -l 25 &

# Limit Pylance server process
cpulimit -p 1752 -l 25 &

echo "CPU usage has been limited to 25% for Cursor processes" 