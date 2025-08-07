#!/bin/bash
# ARCHIVED COPY — moved from scripts/verify-service-mesh-health.sh
# See docs/decisions/2025-08-07-remove-service-mesh.md

#!/bin/bash
# Purpose: Verify service mesh health and provide detailed diagnostics
# Usage: ./verify-service-mesh-health.sh
# Requires: Kong, Consul, and services running
#
# DEPRECATION NOTICE: The Kong/Consul/RabbitMQ service-mesh stack is deprecated.
# See docs/decisions/2025-08-07-remove-service-mesh.md for context.
echo "[DEPRECATED] Service mesh validation is deprecated. Outputs may be stale."
echo "            See docs/decisions/2025-08-07-remove-service-mesh.md"

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Service Mesh Health Verification ===${NC}"
echo "Checking all components of the service mesh..."
echo ""

# Function to check upstream health
check_upstream_health() {
    local upstream=$1
    echo -e "\n${BLUE}Checking $upstream:${NC}"
    
    # Get upstream targets
    targets=$(curl -s "http://localhost:10007/upstreams/$upstream/targets" | jq -r '.data[]')
    
    if [ -z "$targets" ]; then
        echo -e "${RED}✗ No targets configured${NC}"
        return 1
    fi
    
    # Get health status
    health_data=$(curl -s "http://localhost:10007/upstreams/$upstream/health" 2>/dev/null || echo "{}")
    
    if [ "$health_data" = "{}" ]; then
        echo -e "${RED}✗ Health check failed${NC}"
        return 1
    fi
    
    # Parse health data
    total=$(echo "$health_data" | jq '.total // 0')
    healthy_count=$(echo "$health_data" | jq '[.data[] | select(.health == "HEALTHY")] | length // 0')
    
    # Show target details
    echo "$health_data" | jq -r '.data[] | "\(.target) - \(.health) (\(.weight))"' | while read line; do
        if echo "$line" | grep -q "HEALTHY"; then
            echo -e "  ${GREEN}✓${NC} $line"
        else
            echo -e "  ${RED}✗${NC} $line"
        fi
    done
    
    # Summary
    if [ "$healthy_count" -gt 0 ]; then
        echo -e "  ${GREEN}Summary: $healthy_count/$total targets healthy${NC}"
    else
        echo -e "  ${RED}Summary: 0/$total targets healthy${NC}"
    fi
}

echo -e "${BLUE}=== 1. Network Connectivity Check ===${NC}"
# Check if networks are properly connected
echo "Checking Docker networks..."
networks=$(docker network ls --format "{{.Name}}" | grep -E "(sutazai|service-mesh)")
for network in $networks; do
    container_count=$(docker network inspect "$network" | jq '.[0].Containers | length')
    echo -e "  ${GREEN}✓${NC} $network: $container_count containers connected"
done

echo -e "\n${BLUE}=== 2. Kong Admin API Check ===${NC}"
kong_status=$(curl -s http://localhost:10007/status)
if [ -n "$kong_status" ]; then
    echo -e "${GREEN}✓ Kong Admin API is responsive${NC}"
    echo "  Database: $(echo "$kong_status" | jq -r '.database.reachable')"
    echo "  Active connections: $(echo "$kong_status" | jq -r '.server.connections_active')"
else
    echo -e "${RED}✗ Kong Admin API is not responding${NC}"
fi

echo -e "\n${BLUE}=== 3. Consul Service Registry Check ===${NC}"
consul_services=$(curl -s http://localhost:10006/v1/agent/services | jq -r 'keys[]' | wc -l)
consul_healthy=$(curl -s http://localhost:10006/v1/health/state/passing | jq length)
echo -e "${GREEN}✓ Consul has $consul_services services registered${NC}"
echo -e "  ${GREEN}$consul_healthy services are healthy${NC}"

echo -e "\n${BLUE}=== 4. Kong Upstream Health Status ===${NC}"
# Get all upstreams
upstreams=$(curl -s http://localhost:10007/upstreams | jq -r '.data[].name' | sort)

# Critical upstreams to check
critical_upstreams=("backend-upstream" "frontend-upstream" "ollama-upstream" "redis-upstream" "postgres-upstream")

for upstream in "${critical_upstreams[@]}"; do
    if echo "$upstreams" | grep -q "^$upstream$"; then
        check_upstream_health "$upstream"
    else
        echo -e "\n${RED}✗ $upstream not found${NC}"
    fi
done

echo -e "\n${BLUE}=== 5. Service Accessibility Test ===${NC}"
# Test if services are accessible through Kong
test_service_route() {
    local path=$1
    local service=$2
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:10005$path" --connect-timeout 2 2>/dev/null || echo "000")
    
    if [ "$response" = "200" ] || [ "$response" = "404" ] || [ "$response" = "302" ]; then
        echo -e "  ${GREEN}✓${NC} $service ($path): HTTP $response"
    elif [ "$response" = "502" ] || [ "$response" = "503" ]; then
        echo -e "  ${YELLOW}⚠${NC}  $service ($path): HTTP $response (backend unavailable)"
    else
        echo -e "  ${RED}✗${NC} $service ($path): HTTP $response"
    fi
}

echo "Testing service routes through Kong proxy..."
test_service_route "/api/health" "Backend API"
test_service_route "/" "Frontend"
test_service_route "/prometheus/-/healthy" "Prometheus"
test_service_route "/ollama/api/version" "Ollama"
test_service_route "/health" "Health Monitor"

echo -e "\n${BLUE}=== 6. Load Balancing Verification ===${NC}"
# Check if any upstream has multiple healthy targets
lb_enabled=false
for upstream in $upstreams; do
    healthy_count=$(curl -s "http://localhost:10007/upstreams/$upstream/health" 2>/dev/null | jq '[.data[] | select(.health == "HEALTHY")] | length // 0')
    if [ "$healthy_count" -gt 1 ]; then
        echo -e "  ${GREEN}✓${NC} $upstream has $healthy_count healthy targets (load balancing active)"
        lb_enabled=true
    fi
done

if [ "$lb_enabled" = false ]; then
    echo -e "  ${YELLOW}⚠${NC}  No upstreams have multiple healthy targets for load balancing"
fi

echo -e "\n${BLUE}=== 7. Summary and Recommendations ===${NC}"
# Calculate overall health
total_upstreams=$(echo "$upstreams" | wc -l)
healthy_upstreams=0

for upstream in $upstreams; do
    if curl -s "http://localhost:10007/upstreams/$upstream/health" 2>/dev/null | jq -e '.data[] | select(.health == "HEALTHY")' >/dev/null 2>&1; then
        ((healthy_upstreams++))
    fi
done

health_percentage=$((healthy_upstreams * 100 / total_upstreams))

echo -e "\nOverall Service Mesh Health: ${health_percentage}%"
echo "  - Total upstreams: $total_upstreams"
echo "  - Healthy upstreams: $healthy_upstreams"
echo "  - Consul services: $consul_services"
echo "  - Kong routes: $(curl -s http://localhost:10007/routes | jq '.data | length')"

if [ "$health_percentage" -lt 50 ]; then
    echo -e "\n${RED}⚠️  Service mesh health is critical!${NC}"
    echo "Recommendations:"
    echo "  1. Check if all services are running: docker ps"
    echo "  2. Verify network connectivity between containers"
    echo "  3. Check service logs for errors"
    echo "  4. Run ./scripts/fix-service-mesh.sh to reconnect networks"
elif [ "$health_percentage" -lt 80 ]; then
    echo -e "\n${YELLOW}⚠${NC}  Service mesh health needs attention"
    echo "Recommendations:"
    echo "  1. Check unhealthy services and restart if needed"
    echo "  2. Verify service configurations"
    echo "  3. Monitor logs for intermittent issues"
else
    echo -e "\n${GREEN}✓ Service mesh is healthy!${NC}"
fi

echo -e "\n${BLUE}=== Verification Complete ===${NC}"

