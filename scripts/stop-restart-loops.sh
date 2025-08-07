#!/bin/bash
# Immediate fix to stop container restart loops

echo "🛑 Stopping containers with excessive restarts..."
echo "=============================================="

# Get containers with high restart counts and stop them
high_restart_containers=$(docker ps -a --format "{{.Names}}" | while read container; do
    count=$(docker inspect $container --format '{{.RestartCount}}' 2>/dev/null || echo "0")
    if [ "$count" -gt 20 ]; then
        echo "$container:$count"
    fi
done)

echo "Containers with excessive restarts (>20):"
echo "$high_restart_containers"
echo ""

# Stop containers with excessive restarts
echo "Stopping high-restart containers..."
for line in $high_restart_containers; do
    container=$(echo $line | cut -d: -f1)
    count=$(echo $line | cut -d: -f2)
    echo "Stopping $container (restart count: $count)..."
    docker update --restart=no $container
    docker stop $container
done

echo ""
echo "✅ High-restart containers stopped"
echo ""
echo "To restart specific containers with proper configuration:"
echo "  docker start <container-name>"
echo ""
echo "To check container status:"
echo "  docker ps -a | grep sutazai"