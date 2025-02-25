#!/bin/bash
# Check container health
docker ps --format "{{.Names}}" | while read container; do
    health=$(docker inspect --format '{{.State.Health.Status}}' $container)
    if [ "$health" != "healthy" ]; then
        echo "Container $container is unhealthy: $health"
    fi
done 