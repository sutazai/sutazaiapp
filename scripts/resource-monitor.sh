#!/bin/bash
# Resource Monitoring Script for SutazAI System
# Continuous monitoring with alerts

echo "=== SutazAI Resource Monitor ==="
echo "Starting continuous monitoring... Press Ctrl+C to stop"

# Thresholds
CPU_THRESHOLD=80
MEM_THRESHOLD=70
LOAD_THRESHOLD=5.0

monitor_loop() {
    while true; do
        clear
        echo "=== System Resource Monitor - $(date) ==="
        echo ""
        
        # Load average check
        LOAD=$(cat /proc/loadavg | cut -d' ' -f1)
        if (( $(echo "$LOAD > $LOAD_THRESHOLD" | bc -l) )); then
            echo "🚨 HIGH LOAD ALERT: $LOAD (threshold: $LOAD_THRESHOLD)"
        else
            echo "✅ Load Average: $LOAD"
        fi
        
        # Memory check
        MEM_PERCENT=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
        if [ "$MEM_PERCENT" -gt "$MEM_THRESHOLD" ]; then
            echo "🚨 HIGH MEMORY ALERT: ${MEM_PERCENT}% (threshold: ${MEM_THRESHOLD}%)"
        else
            echo "✅ Memory Usage: ${MEM_PERCENT}%"
        fi
        
        echo ""
        echo "=== Top CPU Consumers ==="
        ps aux --sort=-%cpu | head -6 | while read -r line; do
            if echo "$line" | grep -q "PID"; then
                echo "$line"
            else
                cpu=$(echo "$line" | awk '{print $3}')
                if (( $(echo "$cpu > $CPU_THRESHOLD" | bc -l) )); then
                    echo "🚨 $line"
                else
                    echo "   $line"
                fi
            fi
        done
        
        echo ""
        echo "=== Code-Index-MCP Processes ==="
        ps aux | grep "code-index-mcp" | grep -v grep | head -5
        
        echo ""
        echo "=== Claude Process Count ==="
        CLAUDE_COUNT=$(ps aux | grep claude | grep -v grep | wc -l)
        echo "Active Claude processes: $CLAUDE_COUNT"
        if [ "$CLAUDE_COUNT" -gt 15 ]; then
            echo "🚨 Too many Claude processes detected"
        fi
        
        echo ""
        echo "=== Docker Container Resources ==="
        docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -6
        
        echo ""
        echo "=== System Resources ==="
        free -h
        
        sleep 10
    done
}

# Set up signal handling
trap 'echo "Monitoring stopped."; exit 0' INT

# Start monitoring
monitor_loop