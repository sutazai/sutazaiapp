#!/bin/bash
# Restart services if they fail
SERVICES=("ai_service" "super_agent" "sutazai")
for service in "${SERVICES[@]}"; do
    if ! docker ps | grep -q $service; then
        docker-compose restart $service
        echo "Restarted $service"
    fi
done 