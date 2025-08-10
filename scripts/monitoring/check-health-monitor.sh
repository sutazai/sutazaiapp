#!/bin/bash

# Strict error handling
set -euo pipefail

# Check health monitor status


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

echo "=== SutazAI Health Monitor Status ==="
echo

echo "Service Status:"
systemctl status sutazai-health-monitor.service --no-pager -l

echo
echo "Recent Logs:"
journalctl -u sutazai-health-monitor.service --no-pager -n 20

echo
echo "Health Statistics:"
if [[ -f /opt/sutazaiapp/logs/health_monitor_stats.json ]]; then
    cat /opt/sutazaiapp/logs/health_monitor_stats.json | python3 -m json.tool
else
    echo "No statistics available yet"
fi

echo
echo "Current Container Health:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(healthy|unhealthy|sutazai-)" | head -20
