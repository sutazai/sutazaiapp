#!/bin/bash

# Fix unhealthy Docker services - Rule #2: Never Break Existing Functionality
set -e

echo "🔧 Fixing unhealthy Docker services..."
echo "Current unhealthy services: sutazai-ollama, sutazai-semgrep"
echo ""

# Function to check service health
check_health() {
    local service=$1
    docker inspect --format='{{.State.Health.Status}}' "$service" 2>/dev/null || echo "not found"
}

# Fix Ollama service
echo "1. Fixing sutazai-ollama..."
echo "   Issue: Resource allocation mismatch (24MB used of 23GB allocated)"

# Check current resource allocation
echo "   Current memory stats:"
docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.MemLimit}}" sutazai-ollama || true

# Update resource limits to reasonable values
echo "   Updating resource limits..."
docker update \
    --memory="4g" \
    --memory-swap="8g" \
    --cpus="2" \
    sutazai-ollama || echo "   Warning: Could not update resources"

# Restart the service
echo "   Restarting sutazai-ollama..."
docker restart sutazai-ollama

# Wait for service to start
echo "   Waiting for service to initialize..."
sleep 10

# Pull a small model to ensure Ollama is working
echo "   Testing Ollama functionality..."
docker exec sutazai-ollama ollama pull tinyllama:latest 2>/dev/null || {
    echo "   Note: Could not pull test model, but service may still be functional"
}

# Check new health status
NEW_HEALTH=$(check_health sutazai-ollama)
echo "   New health status: $NEW_HEALTH"
echo ""

# Fix Semgrep service
echo "2. Fixing sutazai-semgrep..."
echo "   Issue: Missing health check or initialization problem"

# Check if health check exists
echo "   Checking container configuration..."
docker inspect sutazai-semgrep --format='{{json .Config.Healthcheck}}' | python3 -m json.tool || true

# Restart the service
echo "   Restarting sutazai-semgrep..."
docker restart sutazai-semgrep

# Wait for service to start
echo "   Waiting for service to initialize..."
sleep 10

# If still unhealthy, try to fix by running a basic command
if [ "$(check_health sutazai-semgrep)" = "unhealthy" ]; then
    echo "   Service still unhealthy, attempting manual fix..."
    docker exec sutazai-semgrep semgrep --version 2>/dev/null || {
        echo "   Installing semgrep in container..."
        docker exec sutazai-semgrep pip install semgrep --quiet
    }
fi

# Check new health status
NEW_HEALTH=$(check_health sutazai-semgrep)
echo "   New health status: $NEW_HEALTH"
echo ""

# Final status check
echo "📊 Final Service Status:"
echo "================================"
docker ps --filter "name=sutazai-ollama" --filter "name=sutazai-semgrep" \
    --format "table {{.Names}}\t{{.Status}}\t{{.State}}"

echo ""
echo "🔍 Health Check Summary:"
for service in sutazai-ollama sutazai-semgrep; do
    HEALTH=$(check_health $service)
    if [ "$HEALTH" = "healthy" ]; then
        echo "✅ $service: $HEALTH"
    elif [ "$HEALTH" = "unhealthy" ]; then
        echo "❌ $service: $HEALTH (may need manual intervention)"
    else
        echo "⚠️  $service: $HEALTH"
    fi
done

echo ""
echo "💡 If services remain unhealthy:"
echo "   1. Check logs: docker logs sutazai-[service] --tail 50"
echo "   2. Inspect health check: docker inspect sutazai-[service] --format='{{json .State.Health}}'"
echo "   3. Manual restart: docker restart sutazai-[service]"
echo "   4. Rebuild if needed: docker compose up -d --force-recreate sutazai-[service]"