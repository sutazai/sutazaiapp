#!/bin/bash
# Phased System Restart Script
# Implements controlled startup per Rule 11 & 20

set -euo pipefail

echo "🚀 PHASED SYSTEM RESTART"
echo "======================="

# Function to wait for container health
wait_for_health() {
    local container=$1
    local max_wait=60
    local count=0
    
    echo -n "Waiting for $container to be healthy..."
    while [ $count -lt $max_wait ]; do
        if docker inspect "$container" --format='{{.State.Health.Status}}' 2>/dev/null | grep -q "healthy"; then
            echo " ✓"
            return 0
        fi
        echo -n "."
        sleep 2
        ((count+=2))
    done
    echo " ⚠️ (timeout)"
    return 1
}

# Function to check system load
check_load() {
    local load=$(uptime | awk -F'load average:' '{print $2}' | cut -d, -f1 | xargs)
    echo "Current load: $load"
    
    # Wait if load is too high
    while (( $(echo "$load > 8.0" | bc -l) )); then
        echo "Load too high ($load), waiting..."
        sleep 10
        load=$(uptime | awk -F'load average:' '{print $2}' | cut -d, -f1 | xargs)
    done
}

# Phase 0: Core Infrastructure
echo -e "\n📦 Phase 0: Starting Core Infrastructure..."
echo "========================================="

core_services=(
    "sutazai-postgres"
    "sutazai-redis"
    "sutazai-neo4j"
    "sutazai-ollama"
    "sutazai-chromadb"
)

for service in "${core_services[@]}"; do
    if ! docker ps --format "{{.Names}}" | grep -q "^$service$"; then
        echo "Starting $service..."
        docker start "$service" 2>/dev/null || echo "  ⚠️ Could not start $service"
        sleep 5
    else
        echo "✓ $service already running"
    fi
done

# Wait for infrastructure
echo -e "\nWaiting for infrastructure to stabilize..."
sleep 10
check_load

# Phase 1: Backend Services
echo -e "\n🔧 Phase 1: Starting Backend Services..."
echo "======================================="

backend_services=(
    "sutazai-backend"
    "sutazai-ollama-queue"
)

for service in "${backend_services[@]}"; do
    if ! docker ps --format "{{.Names}}" | grep -q "^$service$"; then
        echo "Starting $service..."
        docker start "$service" 2>/dev/null || echo "  ⚠️ Could not start $service"
        wait_for_health "$service"
    else
        echo "✓ $service already running"
    fi
done

check_load

# Phase 2: Critical Agents (10300-10319)
echo -e "\n⚡ Phase 2: Starting Critical Agents..."
echo "======================================"

critical_agents=(
    "sutazai-agentzero-coordinator"
    "sutazai-agent-orchestrator"
    "sutazai-task-assignment-coordinator"
    "sutazai-autonomous-system-controller"
)

for agent in "${critical_agents[@]}"; do
    if ! docker ps --format "{{.Names}}" | grep -q "^$agent$"; then
        echo "Starting $agent..."
        docker start "$agent" 2>/dev/null || echo "  ⚠️ Could not start $agent"
        sleep 3
    else
        echo "✓ $agent already running"
    fi
    check_load
done

# Show current status
echo -e "\n📊 Current System Status"
echo "======================"
echo "Running containers: $(docker ps | grep -c sutazai || echo 0)"
echo "System load: $(uptime | awk -F'load average:' '{print $2}')"
echo ""
echo "Top CPU consumers:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -10

echo -e "\n✅ Phase 2 complete!"
echo ""
echo "🔧 Next steps:"
echo "1. Monitor system: watch 'docker stats --no-stream'"
echo "2. If stable (CPU <50%, load <6), continue with:"
echo "   ./scripts/start-performance-agents.sh"
echo "3. Then specialized agents:"
echo "   ./scripts/start-specialized-agents.sh"
echo ""
echo "⚠️  Wait 5 minutes between phases!"