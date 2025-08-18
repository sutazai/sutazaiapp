#!/bin/bash
# Mesh Communication Test Script
# Purpose: Test service-to-service communication through the mesh
# Created: 2025-08-17 UTC

set -e

echo "============================================"
echo "SERVICE MESH COMMUNICATION TEST"
echo "============================================"
echo "Time: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to test service communication
test_service_communication() {
    local from_service=$1
    local to_service=$2
    local endpoint=$3
    local expected_status=$4
    
    echo -n "Testing $from_service → $to_service ($endpoint)..."
    
    # Create test request through mesh
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST http://localhost:10010/api/v1/mesh/proxy \
        -H "Content-Type: application/json" \
        -d "{
            \"from_service\": \"$from_service\",
            \"to_service\": \"$to_service\",
            \"endpoint\": \"$endpoint\",
            \"method\": \"GET\"
        }" 2>/dev/null || echo "000")
    
    if [ "$response" = "$expected_status" ]; then
        echo -e " ${GREEN}✓${NC} (Status: $response)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e " ${RED}✗${NC} (Expected: $expected_status, Got: $response)"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Function to test load balancing
test_load_balancing() {
    local service=$1
    local requests=$2
    
    echo "Testing load balancing for $service ($requests requests)..."
    
    declare -A instance_counts
    
    for i in $(seq 1 $requests); do
        instance=$(curl -s http://localhost:10010/api/v1/mesh/route/$service | \
            python3 -c "import sys, json; print(json.load(sys.stdin).get('instance_id', 'unknown'))" 2>/dev/null)
        
        if [ -n "$instance" ]; then
            instance_counts[$instance]=$((${instance_counts[$instance]:-0} + 1))
        fi
    done
    
    echo "Load distribution:"
    for instance in "${!instance_counts[@]}"; do
        percentage=$(echo "scale=1; ${instance_counts[$instance]} * 100 / $requests" | bc)
        echo "  - $instance: ${instance_counts[$instance]} requests ($percentage%)"
    done
    
    # Check if load is reasonably balanced
    if [ ${#instance_counts[@]} -gt 1 ]; then
        echo -e "  ${GREEN}✓${NC} Load balancing working"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "  ${RED}✗${NC} Load balancing not working (single instance)"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Function to test circuit breaker
test_circuit_breaker() {
    local service=$1
    
    echo "Testing circuit breaker for $service..."
    
    # Simulate failures to trip circuit breaker
    echo -n "  Simulating failures..."
    for i in {1..5}; do
        curl -s -X POST http://localhost:10010/api/v1/mesh/simulate-failure/$service \
            -H "Content-Type: application/json" \
            -d '{"failure_type": "timeout"}' 2>/dev/null || true
    done
    echo -e " ${GREEN}✓${NC}"
    
    # Check circuit breaker state
    state=$(curl -s http://localhost:10010/api/v1/mesh/circuit-breaker/$service | \
        python3 -c "import sys, json; print(json.load(sys.stdin).get('state', 'unknown'))" 2>/dev/null)
    
    if [ "$state" = "open" ] || [ "$state" = "half_open" ]; then
        echo -e "  Circuit breaker state: $state ${GREEN}✓${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "  Circuit breaker state: $state ${RED}✗${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    
    # Reset circuit breaker
    curl -s -X POST http://localhost:10010/api/v1/mesh/circuit-breaker/$service/reset 2>/dev/null || true
}

# Function to test service discovery
test_service_discovery() {
    echo "Testing service discovery..."
    
    services=(
        "backend-api"
        "mcp-claude-flow"
        "postgresql"
        "redis"
        "ollama"
    )
    
    for service in "${services[@]}"; do
        echo -n "  Discovering $service..."
        
        instances=$(curl -s http://localhost:10006/v1/health/service/$service | \
            python3 -c "import sys, json; data = json.load(sys.stdin); print(len(data))" 2>/dev/null || echo "0")
        
        if [ "$instances" -gt 0 ]; then
            echo -e " ${GREEN}✓${NC} ($instances instances)"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e " ${RED}✗${NC} (not found)"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
    done
}

# Function to test mesh metrics
test_mesh_metrics() {
    echo "Testing mesh metrics collection..."
    
    metrics=(
        "mesh_service_discovery_total"
        "mesh_load_balancer_requests"
        "mesh_circuit_breaker_trips"
        "mesh_request_duration_seconds"
        "mesh_active_services"
    )
    
    for metric in "${metrics[@]}"; do
        echo -n "  Checking metric $metric..."
        
        value=$(curl -s http://localhost:10200/api/v1/query?query=$metric | \
            python3 -c "import sys, json; d = json.load(sys.stdin); print('found' if d.get('data', {}).get('result') else 'missing')" 2>/dev/null || echo "error")
        
        if [ "$value" = "found" ]; then
            echo -e " ${GREEN}✓${NC}"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e " ${RED}✗${NC}"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
    done
}

echo "============================================"
echo "TEST 1: Service Discovery"
echo "============================================"
test_service_discovery

echo ""
echo "============================================"
echo "TEST 2: Service-to-Service Communication"
echo "============================================"
test_service_communication "backend-api" "redis" "/health" "200"
test_service_communication "backend-api" "postgresql" "/health" "200"
test_service_communication "backend-api" "mcp-claude-flow" "/status" "200"
test_service_communication "frontend" "backend-api" "/api/v1/health" "200"

echo ""
echo "============================================"
echo "TEST 3: Load Balancing"
echo "============================================"
test_load_balancing "mcp-claude-flow" 10

echo ""
echo "============================================"
echo "TEST 4: Circuit Breaker"
echo "============================================"
test_circuit_breaker "mcp-ruv-swarm"

echo ""
echo "============================================"
echo "TEST 5: Mesh Metrics"
echo "============================================"
test_mesh_metrics

echo ""
echo "============================================"
echo "TEST 6: DinD Bridge Integration"
echo "============================================"
echo "Testing DinD to Mesh bridge..."

# Test DinD bridge
response=$(curl -s http://localhost:10010/api/v1/mesh/dind/status 2>/dev/null || echo "{}")
if echo "$response" | grep -q "healthy"; then
    echo -e "  DinD bridge status: ${GREEN}✓${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    
    # Get MCP container count
    mcp_count=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('mcp_containers', 0))" 2>/dev/null || echo "0")
    echo "  MCP containers in DinD: $mcp_count"
else
    echo -e "  DinD bridge status: ${RED}✗${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

echo ""
echo "============================================"
echo "TEST SUMMARY"
echo "============================================"
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed! Mesh is fully operational.${NC}"
    exit 0
else
    echo -e "${YELLOW}Some tests failed. Please check the mesh configuration.${NC}"
    exit 1
fi