#!/bin/bash
# Aggressive CPU Reduction Script
# Target: Immediate load reduction

echo "🔥 AGGRESSIVE CPU REDUCTION"
echo "=========================="
echo "Current load: 18.85, 22.77, 33.59"
echo ""

# Step 1: Stop ALL agent containers except infrastructure
echo "🛑 Stopping all agent containers..."

docker ps --format "{{.Names}}" | grep "^sutazai-" | grep -v -E "(postgres|redis|neo4j|ollama|backend|chromadb|qdrant|frontend)" | while read container; do
    echo "Stopping: $container"
    docker stop "$container" >/dev/null 2>&1 &
done

# Wait for stops to complete
echo "Waiting for containers to stop..."
wait

# Step 2: Check remaining
echo -e "\n📊 Remaining containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.CPUPerc}}" | grep sutazai

# Step 3: Apply strict CPU limits to infrastructure
echo -e "\n⚡ Applying strict CPU limits..."

docker update --cpus="2.0" sutazai-ollama 2>/dev/null || true
docker update --cpus="1.0" sutazai-postgres 2>/dev/null || true
docker update --cpus="0.5" sutazai-redis 2>/dev/null || true
docker update --cpus="1.0" sutazai-neo4j 2>/dev/null || true
docker update --cpus="1.0" sutazai-backend 2>/dev/null || true

echo -e "\n✅ Aggressive reduction complete!"
echo ""
echo "Wait 2-3 minutes for load to drop, then check:"
echo "  uptime"
echo "  docker stats --no-stream"