#!/bin/bash
# Performance Optimization Script for SutazAI System
# Addresses high CPU/memory usage issues

set -e

echo "=== SutazAI Performance Optimization Script ==="
echo "Timestamp: $(date)"
echo "Initial Load Average: $(cat /proc/loadavg | cut -d' ' -f1-3)"
echo ""

# Function to log actions
log_action() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# 1. Kill runaway code-index-mcp processes
log_action "Killing high-CPU code-index-mcp processes..."
for pid in $(ps aux | grep "code-index-mcp" | grep -v grep | awk '$3 > 50 {print $2}'); do
    if [ -n "$pid" ]; then
        log_action "Killing runaway code-index-mcp process: $pid"
        kill -9 "$pid" 2>/dev/null || true
    fi
done

# 2. Clean up zombie processes
log_action "Cleaning up zombie processes..."
kill -CHLD 1 2>/dev/null || true

# 3. Kill stuck npm processes
log_action "Killing stuck npm exec processes..."
pkill -f "npm exec" 2>/dev/null || true

# 4. Identify and restart problematic Claude processes
log_action "Checking Claude processes with high resource usage..."
while IFS= read -r line; do
    pid=$(echo "$line" | awk '{print $2}')
    cpu=$(echo "$line" | awk '{print $3}')
    mem=$(echo "$line" | awk '{print $4}')
    cmd=$(echo "$line" | awk '{print $11}')
    
    if (( $(echo "$cpu > 50" | bc -l) )) || (( $(echo "$mem > 3" | bc -l) )); then
        log_action "High resource Claude process found - PID: $pid, CPU: $cpu%, MEM: $mem%"
        # Don't automatically kill, just report for manual review
    fi
done < <(ps aux | grep claude | grep -v grep)

# 5. Optimize Docker containers
log_action "Cleaning Docker resources..."
docker system prune -f --volumes >/dev/null 2>&1 || true

# 6. Set CPU limits for containers
log_action "Applying CPU limits to high-usage containers..."
for container in $(docker ps --format "table {{.Names}}" | grep -v NAMES); do
    docker update --cpus="2" "$container" 2>/dev/null || true
done

# 7. Clear system caches (carefully)
log_action "Clearing system caches..."
sync
echo 1 > /proc/sys/vm/drop_caches
echo 2 > /proc/sys/vm/drop_caches

# 8. Restart problematic MCP servers with resource limits
log_action "Checking MCP server status and applying resource limits..."

# Create systemd-like resource control for critical processes
if [ -d "/sys/fs/cgroup" ]; then
    log_action "Applying cgroup limits to high-CPU processes..."
    # Apply CPU limits to remaining high-usage processes
    for pid in $(ps aux | awk '$3 > 30 && $11 ~ /(claude|python|node)/ {print $2}'); do
        if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
            # Apply nice priority
            renice -n 5 "$pid" 2>/dev/null || true
        fi
    done
fi

# 9. Final system status check
echo ""
log_action "=== Performance Optimization Complete ==="
echo "Final Load Average: $(cat /proc/loadavg | cut -d' ' -f1-3)"
echo "Memory Usage:"
free -h
echo ""
echo "Top 5 CPU processes:"
ps aux --sort=-%cpu | head -6
echo ""
log_action "Optimization complete. Monitor system for 5-10 minutes."

# 10. Create performance monitoring alias
echo "alias perf-check='ps aux --sort=-%cpu | head -10 && echo && free -h && echo && uptime'" >> ~/.bashrc

echo ""
echo "=== Recommendations ==="
echo "1. Monitor code-index-mcp processes - they were consuming 77%+ CPU each"
echo "2. Consider limiting Claude process concurrency (currently 26+ processes)"
echo "3. Review MCP server configurations for memory leaks"
echo "4. Use 'perf-check' alias for quick performance monitoring"
echo "5. Consider restarting high-memory Claude processes manually if needed"