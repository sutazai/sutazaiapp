#\!/bin/bash

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

cd /opt/sutazaiapp
if [ -f system_monitor.log ] && [  -gt 1048576 ]; then
    mv system_monitor.log system_monitor.log.20250718
    touch system_monitor.log
    find . -name "system_monitor.log.*" -mtime +7 -delete
fi
