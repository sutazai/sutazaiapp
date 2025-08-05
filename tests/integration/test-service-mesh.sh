#!/bin/bash
# Purpose: Test service mesh connectivity and load balancing
# Usage: ./test-service-mesh.sh
# Requires: Kong, Consul, and services running

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Service Mesh Testing Suite ==="
echo "Testing connectivity, service discovery, and load balancing..."
echo ""

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "success" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" = "warning" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
    else
        echo -e "${RED}✗${NC} $message"
    fi
}

# Function to test HTTP endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local expected_code=${3:-200}
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    
    if [ "$response" = "$expected_code" ]; then
        print_status "success" "$name: HTTP $response"
        return 0
    else
        print_status "error" "$name: HTTP $response (expected $expected_code)"
        return 1
    fi
}

# Function to test TCP connectivity
test_tcp() {
    local name=$1
    local host=$2
    local port=$3
    
    if timeout 2 bash -c "echo > /dev/tcp/$host/$port" 2>/dev/null; then
        print_status "success" "$name: TCP port $port is open"
        return 0
    else
        print_status "error" "$name: TCP port $port is closed"
        return 1
    fi
}

# Function to check service in Consul
check_consul_service() {
    local service=$1
    local count=$(curl -s "http://localhost:10006/v1/health/service/$service" | jq length)
    
    if [ "$count" -gt 0 ]; then
        local healthy=$(curl -s "http://localhost:10006/v1/health/service/$service" | jq '[.[] | select(.Checks[].Status == "passing")] | length')
        print_status "success" "Consul: $service registered ($healthy/$count healthy)"
        return 0
    else
        print_status "error" "Consul: $service not registered"
        return 1
    fi
}

# Function to check Kong upstream health
check_kong_upstream() {
    local upstream=$1
    local data=$(curl -s "http://localhost:10007/upstreams/$upstream/health" 2>/dev/null || echo "{}")
    
    if [ -z "$data" ] || [ "$data" = "{}" ]; then
        print_status "error" "Kong: $upstream not found"
        return 1
    fi
    
    local total=$(echo "$data" | jq '.total // 0')
    local healthy=$(echo "$data" | jq '.data // [] | [.[] | select(.health == "HEALTHY")] | length // 0')
    
    if [ "$healthy" -gt 0 ]; then
        print_status "success" "Kong: $upstream has $healthy/$total healthy targets"
        return 0
    else
        print_status "warning" "Kong: $upstream has 0/$total healthy targets"
        return 1
    fi
}

echo "=== 1. Testing Core Infrastructure ==="
test_endpoint "Kong Admin API" "http://localhost:10007/status" 200
test_endpoint "Kong Proxy" "http://localhost:10005/" 404
test_endpoint "Consul UI" "http://localhost:10006/ui/" 200
test_tcp "RabbitMQ Management" "localhost" 10042

echo ""
echo "=== 2. Testing Service Discovery (Consul) ==="
services=("redis" "chromadb" "faiss" "prometheus" "rabbitmq" "ollama" "backend" "frontend")
for service in "${services[@]}"; do
    check_consul_service "$service"
done

echo ""
echo "=== 3. Testing Kong Upstreams ==="
upstreams=("redis-upstream" "chromadb-upstream" "faiss-upstream" "prometheus-upstream" "rabbitmq-upstream")
for upstream in "${upstreams[@]}"; do
    check_kong_upstream "$upstream"
done

echo ""
echo "=== 4. Testing Service-to-Service Communication ==="
# Test if services can reach each other through Docker network
echo "Testing internal network connectivity..."

# Get a container that's on both networks to use as test source
test_container="kong"

# Test connectivity from Kong to services
services_to_test=(
    "sutazai-redis:6379"
    "sutazai-chromadb:8000"
    "sutazai-faiss:8080"
    "sutazai-prometheus:9090"
    "rabbitmq:5672"
    "sutazai-ollama:11434"
)

for service in "${services_to_test[@]}"; do
    host=$(echo "$service" | cut -d: -f1)
    port=$(echo "$service" | cut -d: -f2)
    
    if docker exec "$test_container" nc -zv "$host" "$port" 2>&1 | grep -q "succeeded\|open"; then
        print_status "success" "Network: $host:$port is reachable"
    else
        print_status "error" "Network: $host:$port is not reachable"
    fi
done

echo ""
echo "=== 5. Testing Kong Routes ==="
# Check if any routes are configured
route_count=$(curl -s "http://localhost:10007/routes" | jq '.data | length')
if [ "$route_count" -gt 0 ]; then
    print_status "success" "Kong has $route_count routes configured"
    
    # Test first few routes
    curl -s "http://localhost:10007/routes" | jq -r '.data[0:3] | .[] | "\(.name // .id) -> \(.paths[0] // "/")"' | while read route; do
        echo "  - $route"
    done
else
    print_status "warning" "Kong has no routes configured"
fi

echo ""
echo "=== 6. Testing Load Balancing ==="
# Test if load balancing works by making multiple requests
test_load_balancing() {
    local upstream=$1
    local path=$2
    local count=10
    
    echo "Testing load balancing for $upstream (making $count requests)..."
    
    # Check if route exists for this upstream
    if ! curl -s "http://localhost:10007/routes" | jq -e ".data[] | select(.service.name == \"$upstream\")" >/dev/null 2>&1; then
        print_status "warning" "No route configured for $upstream"
        return
    fi
    
    # Make multiple requests and count unique backends
    responses=""
    for i in $(seq 1 $count); do
        response=$(curl -s -H "Host: $upstream.local" "http://localhost:10005$path" 2>/dev/null || echo "error")
        responses="$responses\n$response"
    done
    
    unique_count=$(echo -e "$responses" | sort | uniq | wc -l)
    if [ "$unique_count" -gt 1 ]; then
        print_status "success" "$upstream: Load balanced across $unique_count backends"
    else
        print_status "warning" "$upstream: All requests went to same backend"
    fi
}

# Test load balancing for services with multiple instances
test_load_balancing "backend-service" "/health"

echo ""
echo "=== 7. Service Mesh Health Summary ==="
echo ""

# Count healthy services
total_services=$(curl -s "http://localhost:10006/v1/agent/services" | jq 'length')
healthy_services=$(curl -s "http://localhost:10006/v1/health/state/passing" | jq '[.[] | select(.ServiceName != null)] | length')

echo "Consul Services: $healthy_services/$total_services healthy"

# Count healthy upstreams
total_upstreams=$(curl -s "http://localhost:10007/upstreams" | jq '.data | length')
healthy_upstreams=0
curl -s "http://localhost:10007/upstreams" | jq -r '.data[].name' | while read upstream; do
    if curl -s "http://localhost:10007/upstreams/$upstream/health" | jq -e '.data[] | select(.health == "HEALTHY")' >/dev/null 2>&1; then
        ((healthy_upstreams++))
    fi
done

echo "Kong Upstreams: $healthy_upstreams/$total_upstreams have healthy targets"

# Overall status
echo ""
if [ "$healthy_services" -gt 0 ] && [ "$healthy_upstreams" -gt 0 ]; then
    print_status "success" "Service mesh is operational"
else
    print_status "error" "Service mesh has issues that need attention"
fi

echo ""
echo "=== Testing Complete ==="