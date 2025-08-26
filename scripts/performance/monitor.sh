#!/bin/bash
# SutazAI Performance Monitor
# Run this script to check system performance

echo "=== SutazAI Performance Report ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "=== System Resources ==="
echo "CPU Usage:"
top -bn1 | head -5 | tail -2
echo ""
echo "Memory Usage:"
free -h
echo ""
echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')"
echo ""

echo "=== High CPU Processes ==="
ps aux | head -1
ps aux | sort -k3 -rn | head -10 | grep -v "ps aux"
echo ""

echo "=== Docker Containers ==="
docker stats --no-stream --format "table {{.Container}}\t{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
echo ""

echo "=== Claude Processes ==="
ps aux | grep claude | grep -v grep || echo "No Claude processes found"
echo ""

echo "=== Backend Status ==="
curl -s http://localhost:10010/health | python3 -m json.tool 2>/dev/null || echo "Backend not responding"
echo ""

echo "=== Recommendations ==="
cpu_percent=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
if (( $(echo "$cpu_percent > 80" | bc -l) )); then
    echo "⚠️  High CPU usage detected. Consider:"
    echo "   - Running performance optimizer: python3 /opt/sutazaiapp/scripts/performance/optimize_system.py"
    echo "   - Checking for runaway processes"
    echo "   - Reviewing Docker container limits"
fi

mem_percent=$(free | grep Mem | awk '{print ($3/$2) * 100.0}' | cut -d'.' -f1)
if [ "$mem_percent" -gt 80 ]; then
    echo "⚠️  High memory usage detected. Consider:"
    echo "   - Restarting memory-intensive containers"
    echo "   - Clearing caches: sync && echo 3 > /proc/sys/vm/drop_caches"
fi

echo ""
echo "=== Performance History ==="
if [ -f /opt/sutazaiapp/logs/performance.log ]; then
    tail -5 /opt/sutazaiapp/logs/performance.log
else
    echo "No performance history available"
fi
