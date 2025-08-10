#!/bin/bash

# Strict error handling
set -euo pipefail

# Parallel Execution Monitor - Run in separate terminal
# Updates every 10 seconds with system status


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

# Timeout mechanism to prevent infinite loops
LOOP_TIMEOUT=${LOOP_TIMEOUT:-300}  # 5 minute default timeout
loop_start=$(date +%s)
while true; do
    clear
    echo "=============================================="
    echo "     PARALLEL EXECUTION MONITOR"
    echo "=============================================="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    echo "=== CONTAINER STATUS (Top 10) ==="
    docker ps --format "table {{.Names}}\t{{.Status}}" | head -11
    echo ""
    
    echo "=== RESOURCE USAGE ==="
    docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}" | head -11
    echo ""
    
    echo "=== CLEANUP PROGRESS ==="
    DOCKERFILES=$(find /opt/sutazaiapp -name 'Dockerfile*' -type f 2>/dev/null | wc -l)
    SCRIPTS=$(find /opt/sutazaiapp/scripts -type f \( -name '*.py' -o -name '*.sh' \) 2>/dev/null | wc -l)
    BASEAGENTS=$(find /opt/sutazaiapp -name 'base_agent.py' -type f 2>/dev/null | wc -l)
    FANTASY=$(grep -r -E "(wizard|magic|teleport|fantasy)" /opt/sutazaiapp --include="*.py" --include="*.md" 2>/dev/null | wc -l)
    
    echo "Dockerfiles:      587 → $DOCKERFILES $([ $DOCKERFILES -lt 100 ] && echo '✅' || echo '⏳')"
    echo "Scripts:          447 → $SCRIPTS $([ $SCRIPTS -lt 100 ] && echo '✅' || echo '⏳')"
    echo "BaseAgent files:    2 → $BASEAGENTS $([ $BASEAGENTS -eq 1 ] && echo '✅' || echo '⏳')"
    echo "Fantasy elements: 366 → $FANTASY $([ $FANTASY -lt 50 ] && echo '✅' || echo '⏳')"
    echo ""
    
    echo "=== SERVICE HEALTH ==="
    
    # Backend
    if curl -s http://localhost:10010/health 2>/dev/null | grep -q "healthy"; then
        echo "✅ Backend API:      HEALTHY"
    else
        echo "❌ Backend API:      UNHEALTHY"
    fi
    
    # Frontend
    if curl -s http://localhost:10011/ > /dev/null 2>&1; then
        echo "✅ Frontend UI:      OPERATIONAL"
    else
        echo "❌ Frontend UI:      DOWN"
    fi
    
    # Ollama
    if curl -s http://localhost:10104/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama:           RESPONDING"
    else
        echo "❌ Ollama:           NOT RESPONDING"
    fi
    
    # RabbitMQ
    if docker exec sutazai-rabbitmq rabbitmqctl status > /dev/null 2>&1; then
        echo "✅ RabbitMQ:         RUNNING"
    else
        echo "❌ RabbitMQ:         ERROR"
    fi
    
    # PostgreSQL
    if docker exec sutazai-postgres psql -U sutazai -c "SELECT 1" > /dev/null 2>&1; then
        echo "✅ PostgreSQL:       CONNECTED"
    else
        echo "❌ PostgreSQL:       DISCONNECTED"
    fi
    
    echo ""
    echo "=== MEMORY SUMMARY ==="
    TOTAL_MEM=$(docker stats --no-stream --format "{{.MemUsage}}" | awk '{sum+=$1} END {printf "%.2f", sum/1024}')
    echo "Total Memory Usage: ${TOTAL_MEM} GB"
    
    echo ""
    echo "Press Ctrl+C to exit monitoring"
    echo "Refreshing in 10 seconds..."
    
    sleep 10
    # Check for timeout
    current_time=$(date +%s)
    if [[ $((current_time - loop_start)) -gt $LOOP_TIMEOUT ]]; then
        echo 'Loop timeout reached after ${LOOP_TIMEOUT}s, exiting...' >&2
        break
    fi

done