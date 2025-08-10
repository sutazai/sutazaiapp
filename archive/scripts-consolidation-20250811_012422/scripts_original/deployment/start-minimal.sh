#!/bin/bash
# Minimal startup script for SutazAI - only essential services

set -e


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

echo "Starting SutazAI in minimal mode..."

# Clean up before starting
echo "Cleaning up old containers..."
docker compose down --remove-orphans 2>/dev/null || true

# Remove unused networks
docker network prune -f 2>/dev/null || true

# Create network if needed
docker network create sutazai-network 2>/dev/null || true

# Start only essential services (no monitoring, no optional features)
echo "Starting core services only..."
docker compose up -d \
    postgres \
    redis \
    backend \
    frontend \
    ollama

echo "Waiting for services to start..."
sleep 10

# Check health
echo "Checking service health..."
docker compose ps

echo ""
echo "Minimal SutazAI is starting..."
echo "Backend API: http://localhost:10010"
echo "Frontend UI: http://localhost:10011"
echo ""
echo "To enable full monitoring, run: docker compose up -d prometheus grafana loki"
echo "To enable optional features, edit .env and run: ./scripts/start-with-features.sh"