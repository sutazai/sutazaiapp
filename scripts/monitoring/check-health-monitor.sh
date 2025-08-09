#!/bin/bash
# Check health monitor status

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
