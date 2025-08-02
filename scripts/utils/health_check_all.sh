#!/bin/bash

# SutazAI Comprehensive Health Check Script
# Verifies all services are running and responsive

echo "🔍 SutazAI Enterprise System Health Check"
echo "════════════════════════════════════════════════════════════"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILED_SERVICES=()
SUCCESSFUL_SERVICES=()

check_service() {
    local service_name="$1"
    local test_url="$2"
    local expected_response="$3"
    
    echo -n "Checking $service_name... "
    
    # Check if container is running first
    if ! docker-compose ps | grep -q "sutazai-$service_name.*Up"; then
        echo -e "${RED}❌ Container not running${NC}"
        FAILED_SERVICES+=("$service_name")
        return 1
    fi
    
    # Test the endpoint if provided
    if [[ -n "$test_url" ]]; then
        response=$(curl -s --max-time 10 "$test_url" 2>/dev/null)
        if [[ $? -eq 0 ]] && [[ -n "$response" ]]; then
            if [[ -z "$expected_response" ]] || echo "$response" | grep -q "$expected_response"; then
                echo -e "${GREEN}✅ Healthy${NC}"
                SUCCESSFUL_SERVICES+=("$service_name")
                return 0
            fi
        fi
    fi
    
    echo -e "${YELLOW}⚠️ Running but not responsive${NC}"
    FAILED_SERVICES+=("$service_name")
    return 1
}

# Check all services
echo "🔍 Core Infrastructure:"
check_service "postgres" "" ""
check_service "redis" "" ""
check_service "neo4j" "http://localhost:7474" ""

echo ""
echo "🔍 Vector Databases:"
check_service "chromadb" "http://localhost:8001/api/v1/heartbeat" ""
check_service "qdrant" "http://localhost:6333/collections" "ok"
check_service "faiss" "http://localhost:8002/health" "healthy"

echo ""
echo "🔍 AI Services:"
check_service "ollama" "http://localhost:11434/api/tags" ""
check_service "backend" "http://localhost:8000/health" "status"
check_service "frontend" "http://localhost:8501/healthz" ""

echo ""
echo "🔍 Monitoring:"
check_service "prometheus" "http://localhost:9090/-/healthy" ""
check_service "grafana" "http://localhost:3000/api/health" "ok"
check_service "loki" "http://localhost:3100/ready" ""
check_service "promtail" "" ""

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📊 Health Check Summary:"
echo -e "${GREEN}✅ Successful: ${#SUCCESSFUL_SERVICES[@]} services${NC}"
if [[ ${#SUCCESSFUL_SERVICES[@]} -gt 0 ]]; then
    echo "   ${SUCCESSFUL_SERVICES[*]}"
fi

if [[ ${#FAILED_SERVICES[@]} -gt 0 ]]; then
    echo -e "${RED}❌ Failed: ${#FAILED_SERVICES[@]} services${NC}"
    echo "   ${FAILED_SERVICES[*]}"
    echo ""
    echo "🔧 Troubleshooting commands:"
    for service in "${FAILED_SERVICES[@]}"; do
        echo "   docker-compose logs $service --tail=20"
    done
    exit 1
else
    echo -e "${GREEN}🎉 All services are healthy!${NC}"
    exit 0
fi