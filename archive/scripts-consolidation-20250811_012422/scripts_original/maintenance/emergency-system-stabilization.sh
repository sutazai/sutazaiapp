#!/bin/bash
# Emergency System Stabilization Script
# Target: Reduce CPU from 77.6% to <30%

set -euo pipefail


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

echo "🚨 EMERGENCY SYSTEM STABILIZATION 🚨"
echo "Current CPU: 77.6% | Target: <30%"
echo "Load Average: 20.32 | Target: <6.0"
echo "========================================="

# Step 1: Stop all containers consuming >40% CPU
echo -e "\n🛑 Step 1: Stopping high CPU containers..."

high_cpu_containers=(
    "sutazai-bigagi-syste"           # 27.3%
    "sutazai-codebase-tea"           # 41.6%
    "sutazai-data-pipelin"           # 51.0%
    "sutazai-explainable-"           # 51.7%
    "sutazai-genetic-algo"           # 50.5%
    "sutazai-langflow-wor"           # 48.2%
    "sutazai-localagi-orc"           # 50.6%
    "sutazai-meta-learnin"           # 35.3%
    "sutazai-multi-modal-"           # 51.7%
    "sutazai-neuromorphic"           # 50.4%
)

for container in "${high_cpu_containers[@]}"; do
    full_name=$(docker ps --format "{{.Names}}" | grep "^$container" || true)
    if [ -n "$full_name" ]; then
        echo "Stopping $full_name..."
        docker update --restart=no "$full_name" 2>/dev/null || true
        docker stop "$full_name" 2>/dev/null || true
    fi
done

# Step 2: Stop all containers in restart loops
echo -e "\n🔄 Step 2: Stopping containers in restart loops..."

docker ps -a --format "{{.Names}}\t{{.Status}}" | grep -E "(Restarting|restarting)" | cut -f1 | while read container; do
    echo "Stopping restarting container: $container"
    docker update --restart=no "$container" 2>/dev/null || true
    docker stop "$container" 2>/dev/null || true
done

# Step 3: Apply CPU limits to remaining containers
echo -e "\n⚡ Step 3: Applying CPU limits to running containers..."

for container in $(docker ps --format "{{.Names}}" | grep "^sutazai-"); do
    echo "Limiting CPU for $container..."
    docker update --cpus="0.5" "$container" 2>/dev/null || true
done

# Step 4: Clear system caches
echo -e "\n🧹 Step 4: Clearing system caches..."
sync
echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || echo "Need sudo for cache clearing"

# Step 5: Stop non-critical services
echo -e "\n📦 Step 5: Stopping non-critical services..."

non_critical=(
    "sutazai-model-traini"
    "sutazai-complex-prob"
    "sutazai-automated-in"
)

for container in "${non_critical[@]}"; do
    full_name=$(docker ps --format "{{.Names}}" | grep "^$container" || true)
    if [ -n "$full_name" ]; then
        echo "Stopping non-critical: $full_name"
        docker stop "$full_name" 2>/dev/null || true
    fi
done

# Step 6: Show current status
echo -e "\n📊 Step 6: Checking system status..."
sleep 5

echo -e "\nCurrent running containers:"
docker ps | grep -c sutazai || echo "0"

echo -e "\nTop CPU consumers:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}" | head -10

echo -e "\nSystem load:"
uptime

echo -e "\n✅ Emergency stabilization complete!"
echo "========================================="
echo ""
echo "🔧 Next steps:"
echo "1. Monitor CPU: watch 'docker stats --no-stream'"
echo "2. Start critical services only:"
echo "   - docker start sutazai-postgres"
echo "   - docker start sutazai-redis"
echo "   - docker start sutazai-backend"
echo "   - docker start sutazai-ollama"
echo ""
echo "3. Start agents one by one:"
echo "   - Phase 1: Critical agents (ports 10300-10319)"
echo "   - Phase 2: Performance agents (ports 10320-10419)"
echo "   - Phase 3: Specialized agents (ports 10420-10599)"
echo ""
echo "⚠️  DO NOT start all containers at once!"