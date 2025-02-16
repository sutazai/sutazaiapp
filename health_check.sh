#!/bin/bash

# Automated health check script
services=("ai_service" "super_agent" "sutazai")

for service in "${services[@]}"; do
    echo "🔍 Checking health of $service..."
    status=$(docker inspect --format='{{.State.Health.Status}}' $service)
    
    if [ "$status" != "healthy" ]; then
        echo "❌ $service is not healthy! Status: $status"
        echo "🔄 Attempting to restart $service..."
        docker-compose restart $service
    else
        echo "✅ $service is healthy"
    fi
done

# Check system health
docker ps -q | xargs docker inspect --format '{{.State.Health.Status}}' | grep -v "healthy" && echo "System is unhealthy" || echo "System is healthy" 