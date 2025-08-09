#!/bin/bash
# Check for containers with restart loops

echo "Checking for containers with restart loops..."
echo "============================================"
echo ""

# Get all containers and check restart counts
for container in $(docker ps -a --format "{{.Names}}"); do 
    count=$(docker inspect $container --format '{{.RestartCount}}' 2>/dev/null)
    if [ -n "$count" ] && [ "$count" -gt 0 ]; then
        status=$(docker inspect $container --format '{{.State.Status}}')
        uptime=$(docker ps -a --format "table {{.Names}}\t{{.Status}}" | grep "^$container" | awk '{print $2, $3, $4}')
        echo "⚠️  $container:"
        echo "   Restart Count: $count"
        echo "   Current Status: $status"
        echo "   Uptime: $uptime"
        echo ""
    fi
done

# Check health status of all containers
echo ""
echo "Checking container health status..."
echo "===================================="
echo ""

unhealthy_count=0
for container in $(docker ps --format "{{.Names}}"); do
    health=$(docker inspect $container --format '{{.State.Health.Status}}' 2>/dev/null)
    if [ "$health" = "unhealthy" ]; then
        echo "❌ $container is unhealthy"
        docker inspect $container --format '{{range .State.Health.Log}}{{.Output}}{{end}}' | tail -3
        echo ""
        ((unhealthy_count++))
    elif [ "$health" = "starting" ]; then
        echo "⏳ $container health is starting"
    fi
done

if [ $unhealthy_count -eq 0 ]; then
    echo "✅ All containers are healthy"
fi

# Check for containers that exited
echo ""
echo "Checking for exited containers..."
echo "================================="
echo ""

exited=$(docker ps -a --filter "status=exited" --format "table {{.Names}}\t{{.Status}}")
if [ -n "$exited" ]; then
    echo "$exited"
else
    echo "✅ No exited containers found"
fi