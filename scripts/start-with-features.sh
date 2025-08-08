#!/bin/bash
# Start SutazAI with optional features based on environment variables

set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check feature flags
ENABLE_FSDP=${ENABLE_FSDP:-false}
ENABLE_TABBY=${ENABLE_TABBY:-false}

echo "Starting SutazAI..."
echo "ENABLE_FSDP: $ENABLE_FSDP"
echo "ENABLE_TABBY: $ENABLE_TABBY"

# Build profiles string
PROFILES=""

if [ "$ENABLE_FSDP" = "true" ]; then
    echo "Enabling FSDP service..."
    PROFILES="$PROFILES --profile fsdp"
fi

if [ "$ENABLE_TABBY" = "true" ]; then
    echo "Enabling TabbyML service..."
    PROFILES="$PROFILES --profile tabby"
fi

# Ensure network exists
docker network create sutazai-network 2>/dev/null || true

# Start core services (always run)
echo "Starting core services..."
docker-compose up -d \
    postgres \
    redis \
    neo4j \
    ollama \
    backend \
    frontend \
    prometheus \
    grafana \
    loki

# Start optional services if enabled
if [ -n "$PROFILES" ]; then
    echo "Starting optional services with profiles:$PROFILES"
    docker-compose $PROFILES up -d
fi

# Show status
echo ""
echo "Services started. Checking status..."
sleep 5
docker-compose ps

echo ""
echo "SutazAI is starting up..."
echo "Backend API: http://localhost:10010"
echo "Frontend UI: http://localhost:10011"
echo "Grafana: http://localhost:10201 (admin/admin)"

if [ "$ENABLE_FSDP" = "true" ]; then
    echo "FSDP Training: http://localhost:8596"
fi

if [ "$ENABLE_TABBY" = "true" ]; then
    echo "TabbyML: http://localhost:10303"
fi