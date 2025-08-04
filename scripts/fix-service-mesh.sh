#!/bin/bash
# Purpose: Fix service mesh connectivity issues by connecting services to the correct network
# Usage: ./fix-service-mesh.sh
# Requires: Docker running with services

set -e

echo "=== Service Mesh Fix Script ==="
echo "Fixing network connectivity issues between Kong, Consul, and services..."

# Define networks
SERVICE_MESH_NETWORK="sutazaiapp_service-mesh"
SUTAZAI_NETWORK="sutazai-network"

# Function to connect container to network if not already connected
connect_to_network() {
    local container=$1
    local network=$2
    
    # Check if container exists
    if ! docker ps -a --format "{{.Names}}" | grep -q "^${container}$"; then
        echo "⚠️  Container $container not found, skipping..."
        return 1
    fi
    
    # Check if already connected
    if docker inspect "$container" | jq -r ".[0].NetworkSettings.Networks | keys[]" | grep -q "^${network}$"; then
        echo "✓ $container is already connected to $network"
    else
        echo "→ Connecting $container to $network..."
        docker network connect "$network" "$container" || echo "⚠️  Failed to connect $container to $network"
    fi
}

# Function to update Kong upstream targets with network-specific hostnames
update_kong_targets() {
    local upstream=$1
    local service=$2
    local port=$3
    local container_name=$4
    
    echo "→ Updating Kong upstream targets for $upstream..."
    
    # Remove existing targets
    targets=$(curl -s "http://localhost:10007/upstreams/$upstream/targets" | jq -r '.data[].id // empty')
    for target_id in $targets; do
        curl -s -X DELETE "http://localhost:10007/upstreams/$upstream/targets/$target_id"
    done
    
    # Add new target with container name
    curl -s -X POST "http://localhost:10007/upstreams/$upstream/targets" \
        -H "Content-Type: application/json" \
        -d "{\"target\": \"$container_name:$port\", \"weight\": 100}" | jq -c '.'
}

echo ""
echo "=== Step 1: Connect Kong and Consul to sutazai-network ==="
connect_to_network "kong" "$SUTAZAI_NETWORK"
connect_to_network "consul" "$SUTAZAI_NETWORK"
connect_to_network "rabbitmq" "$SUTAZAI_NETWORK"

echo ""
echo "=== Step 2: Connect services to service-mesh network ==="
# Get all containers on sutazai-network
containers=$(docker ps --format "{{.Names}}" | grep "sutazai-" || true)
for container in $containers; do
    connect_to_network "$container" "$SERVICE_MESH_NETWORK"
done

echo ""
echo "=== Step 3: Update Kong upstream targets with proper container names ==="
# Update targets to use actual container names
update_kong_targets "redis-upstream" "redis" "6379" "sutazai-redis"
update_kong_targets "postgres-upstream" "postgres" "5432" "sutazai-postgres"
update_kong_targets "chromadb-upstream" "chromadb" "8000" "sutazai-chromadb"
update_kong_targets "faiss-upstream" "faiss" "8080" "sutazai-faiss"
update_kong_targets "prometheus-upstream" "prometheus" "9090" "sutazai-prometheus"
update_kong_targets "rabbitmq-upstream" "rabbitmq" "15672" "rabbitmq"
update_kong_targets "dify-upstream" "dify" "3000" "sutazai-dify-automation-specialist-phase3"
update_kong_targets "crewai-upstream" "crewai" "8080" "sutazai-crewai-phase2"

echo ""
echo "=== Step 4: Update Kong DNS resolver ==="
# Configure Kong to use Docker's embedded DNS
curl -s -X PATCH http://localhost:10007/ \
    -H "Content-Type: application/json" \
    -d '{
        "dns_resolver": ["127.0.0.11:53"],
        "dns_hostsfile": "/etc/hosts",
        "dns_order": ["LAST", "SRV", "A", "AAAA", "CNAME"]
    }' | jq -c '.'

echo ""
echo "=== Step 5: Restart Kong to apply DNS changes ==="
docker restart kong

echo ""
echo "=== Step 6: Wait for Kong to be healthy ==="
sleep 10
for i in {1..30}; do
    if curl -s http://localhost:10007/status >/dev/null 2>&1; then
        echo "✓ Kong is healthy"
        break
    fi
    echo "→ Waiting for Kong to start... ($i/30)"
    sleep 2
done

echo ""
echo "=== Step 7: Register services in Consul ==="
# Register services in Consul with proper addresses
register_service() {
    local name=$1
    local container=$2
    local port=$3
    local tags=$4
    
    # Get container IP on service-mesh network
    local ip=$(docker inspect "$container" | jq -r ".[0].NetworkSettings.Networks[\"$SERVICE_MESH_NETWORK\"].IPAddress // empty")
    
    if [ -n "$ip" ]; then
        echo "→ Registering $name at $ip:$port in Consul..."
        curl -s -X PUT "http://localhost:10006/v1/agent/service/register" \
            -H "Content-Type: application/json" \
            -d "{
                \"ID\": \"${name}-1\",
                \"Name\": \"$name\",
                \"Address\": \"$ip\",
                \"Port\": $port,
                \"Tags\": $tags,
                \"Check\": {
                    \"TCP\": \"$ip:$port\",
                    \"Interval\": \"10s\",
                    \"Timeout\": \"5s\"
                }
            }"
    else
        echo "⚠️  Could not get IP for $container, skipping Consul registration"
    fi
}

# Register core services
register_service "redis" "sutazai-redis" 6379 '["cache", "storage", "pubsub"]'
register_service "postgres" "sutazai-postgres" 5432 '["database", "primary", "postgresql"]'
register_service "chromadb" "sutazai-chromadb" 8000 '["ai", "vectordb", "storage"]'
register_service "faiss" "sutazai-faiss" 8080 '["ai", "vectordb", "similarity"]'
register_service "prometheus" "sutazai-prometheus" 9090 '["metrics", "monitoring", "observability"]'
register_service "rabbitmq" "rabbitmq" 15672 '["queue", "messaging", "amqp"]'

echo ""
echo "=== Step 8: Test connectivity ==="
echo "→ Testing Kong Admin API..."
curl -s http://localhost:10007/status | jq -c '.'

echo ""
echo "→ Testing Consul API..."
curl -s http://localhost:10006/v1/agent/services | jq 'keys' | jq -c '.'

echo ""
echo "→ Testing service connectivity through Kong..."
# Test a simple route through Kong (if configured)
if curl -s http://localhost:10007/routes | jq -e '.data | length > 0' >/dev/null; then
    echo "✓ Routes are configured in Kong"
else
    echo "⚠️  No routes configured in Kong yet"
fi

echo ""
echo "=== Service Mesh Fix Complete ==="
echo "✓ Networks connected"
echo "✓ Kong targets updated"
echo "✓ Services registered in Consul"
echo ""
echo "Next steps:"
echo "1. Configure Kong routes for your services"
echo "2. Set up service discovery synchronization"
echo "3. Enable health checks and monitoring"