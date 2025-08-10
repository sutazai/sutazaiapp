#!/bin/bash
# Purpose: Fix Kong DNS resolution by using container IPs instead of names
# Usage: ./fix-kong-dns-resolution.sh
# Requires: Kong and services running

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

echo "=== Fixing Kong DNS Resolution ==="
echo "Updating upstream targets to use IP addresses..."
echo ""

# Function to get container IP on service-mesh network
get_container_ip() {
    local container=$1
    local network="sutazaiapp_service-mesh"
    
    # Try to get IP on service-mesh network
    ip=$(docker inspect "$container" 2>/dev/null | jq -r ".[0].NetworkSettings.Networks[\"$network\"].IPAddress // empty")
    
    # If not found, try sutazai-network
    if [ -z "$ip" ] || [ "$ip" = "null" ]; then
        network="sutazai-network"
        ip=$(docker inspect "$container" 2>/dev/null | jq -r ".[0].NetworkSettings.Networks[\"$network\"].IPAddress // empty")
    fi
    
    if [ -n "$ip" ] && [ "$ip" != "null" ]; then
        echo "$ip"
    else
        echo ""
    fi
}

# Function to update Kong upstream target
update_upstream_target() {
    local upstream=$1
    local container=$2
    local port=$3
    
    echo "→ Updating $upstream..."
    
    # Get container IP
    ip=$(get_container_ip "$container")
    
    if [ -z "$ip" ]; then
        echo "  ⚠️  Could not find IP for $container"
        return 1
    fi
    
    # Remove all existing targets
    targets=$(curl -s "http://localhost:10007/upstreams/$upstream/targets" | jq -r '.data[].id // empty')
    for target_id in $targets; do
        curl -s -X DELETE "http://localhost:10007/upstreams/$upstream/targets/$target_id" >/dev/null
    done
    
    # Add new target with IP address
    response=$(curl -s -X POST "http://localhost:10007/upstreams/$upstream/targets" \
        -H "Content-Type: application/json" \
        -d "{\"target\": \"$ip:$port\", \"weight\": 100}")
    
    if echo "$response" | jq -e '.id' >/dev/null 2>&1; then
        echo "  ✓ Updated to $ip:$port"
    else
        echo "  ✗ Failed to update target"
        echo "$response" | jq '.'
    fi
}

# Update critical services first
echo "=== Updating Core Services ==="
update_upstream_target "backend-upstream" "sutazai-backend" "8000"
update_upstream_target "redis-upstream" "sutazai-redis" "6379"
update_upstream_target "ollama-upstream" "sutazai-ollama" "10104"
update_upstream_target "prometheus-upstream" "sutazai-prometheus" "9090"

# Update vector databases
echo ""
echo "=== Updating Vector Databases ==="
update_upstream_target "chromadb-upstream" "sutazai-chromadb" "8000"
update_upstream_target "faiss-upstream" "sutazai-faiss" "8080"
update_upstream_target "qdrant-upstream" "sutazai-qdrant" "6333"

# Update databases
echo ""
echo "=== Updating Databases ==="
# Find postgres container
postgres_container=$(docker ps --format "{{.Names}}" | grep -E "postgres|postgresql" | grep -v "kong-database" | head -1)
if [ -n "$postgres_container" ]; then
    update_upstream_target "postgres-upstream" "$postgres_container" "5432"
else
    echo "  ⚠️  PostgreSQL container not found"
fi

update_upstream_target "neo4j-upstream" "sutazai-neo4j" "7474"

# Update monitoring services
echo ""
echo "=== Updating Monitoring Services ==="
update_upstream_target "grafana-upstream" "sutazai-grafana" "3000" || echo "  ⚠️  Grafana not found"
update_upstream_target "health-monitor-upstream" "sutazai-health-monitor" "8080" || echo "  ⚠️  Health monitor not found"

# Update agent services
echo ""
echo "=== Updating Agent Services ==="
update_upstream_target "crewai-upstream" "sutazai-crewai-phase2" "8080"
update_upstream_target "autogpt-upstream" "sutazai-autogpt-phase2" "8080"
update_upstream_target "aider-upstream" "sutazai-aider-phase2" "8080"

# Update workflow services
echo ""
echo "=== Updating Workflow Services ==="
update_upstream_target "dify-upstream" "sutazai-dify-automation-specialist-phase3" "3000"

# Find and update other services
echo ""
echo "=== Updating Additional Services ==="

# RabbitMQ
rabbitmq_ip=$(get_container_ip "rabbitmq")
if [ -n "$rabbitmq_ip" ]; then
    curl -s -X DELETE "http://localhost:10007/upstreams/rabbitmq-upstream/targets" >/dev/null 2>&1
    curl -s -X POST "http://localhost:10007/upstreams/rabbitmq-upstream/targets" \
        -H "Content-Type: application/json" \
        -d "{\"target\": \"$rabbitmq_ip:15672\", \"weight\": 100}" >/dev/null
    echo "  ✓ Updated rabbitmq-upstream to $rabbitmq_ip:15672"
fi

# Frontend (might be different container names)
frontend_container=$(docker ps --format "{{.Names}}" | grep -E "frontend|streamlit|ui" | head -1)
if [ -n "$frontend_container" ]; then
    update_upstream_target "frontend-upstream" "$frontend_container" "8501"
fi

echo ""
echo "=== Verifying Updates ==="
# Check a few critical upstreams
for upstream in "backend-upstream" "redis-upstream" "ollama-upstream"; do
    target=$(curl -s "http://localhost:10007/upstreams/$upstream/targets" | jq -r '.data[0].target // "none"')
    echo "  $upstream: $target"
done

echo ""
echo "=== Testing Connectivity ==="
# Test if Kong can now reach services
test_upstream() {
    local upstream=$1
    health=$(curl -s "http://localhost:10007/upstreams/$upstream/health" | jq -r '.data[0].health // "UNKNOWN"' 2>/dev/null)
    if [ "$health" = "HEALTHY" ]; then
        echo "  ✓ $upstream is healthy"
    else
        echo "  ⚠️  $upstream status: $health"
    fi
}

# Give Kong a moment to perform health checks
echo "Waiting for health checks to update..."
sleep 5

test_upstream "backend-upstream"
test_upstream "redis-upstream"
test_upstream "ollama-upstream"
test_upstream "prometheus-upstream"

echo ""
echo "=== DNS Resolution Fix Complete ==="
echo "Upstream targets now use IP addresses for better reliability"
echo "Run ./scripts/verify-service-mesh-health.sh to check overall health"