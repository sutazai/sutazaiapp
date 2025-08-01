#!/bin/bash
# SutazAI Complete Deployment Verification Script
# Verifies all services, containers, and integrations

set -e

echo "🚀 SutazAI Complete Deployment Verification"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check service health
check_service() {
    local service_name="$1"
    local url="$2"
    local timeout="${3:-5}"
    
    echo -n "Checking $service_name... "
    
    if curl -s --max-time "$timeout" "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ HEALTHY${NC}"
        return 0
    else
        echo -e "${RED}❌ UNHEALTHY${NC}"
        return 1
    fi
}

# Function to check Docker container
check_container() {
    local container_name="$1"
    echo -n "Checking container $container_name... "
    
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "$container_name.*Up"; then
        echo -e "${GREEN}✅ RUNNING${NC}"
        return 0
    else
        echo -e "${RED}❌ NOT RUNNING${NC}"
        return 1
    fi
}

# Function to test API endpoint
test_api_endpoint() {
    local endpoint="$1"
    local method="${2:-GET}"
    local data="$3"
    
    echo -n "Testing API $method $endpoint... "
    
    local curl_cmd="curl -s --max-time 10 -w '%{http_code}'"
    
    if [ "$method" = "POST" ]; then
        curl_cmd="$curl_cmd -X POST -H 'Content-Type: application/json'"
        if [ -n "$data" ]; then
            curl_cmd="$curl_cmd -d '$data'"
        fi
    fi
    
    local response=$(eval "$curl_cmd http://localhost:8000$endpoint")
    local http_code="${response: -3}"
    
    if [ "$http_code" = "200" ]; then
        echo -e "${GREEN}✅ $http_code${NC}"
        return 0
    else
        echo -e "${RED}❌ $http_code${NC}"
        return 1
    fi
}

echo ""
echo "🐳 DOCKER CONTAINER STATUS"
echo "------------------------"

# Core infrastructure containers
declare -a CORE_CONTAINERS=(
    "sutazai-postgres"
    "sutazai-redis" 
    "sutazai-neo4j"
    "sutazai-chromadb"
    "sutazai-qdrant"
    "sutazai-ollama"
    "sutazai-backend-agi"
    "sutazai-frontend-agi"
)

core_healthy=0
for container in "${CORE_CONTAINERS[@]}"; do
    if check_container "$container"; then
        ((core_healthy++))
    fi
done

echo ""
echo "📡 SERVICE HEALTH CHECK"
echo "---------------------"

# Service health endpoints
declare -A SERVICES=(
    ["Backend API"]="http://localhost:8000/health"
    ["Frontend App"]="http://localhost:8501"
    ["LangFlow"]="http://localhost:8090"
    ["FlowiseAI"]="http://localhost:8099"
    ["BigAGI"]="http://localhost:8106"
    ["Dify"]="http://localhost:8107"
    ["n8n"]="http://localhost:5678"
    ["Ollama"]="http://localhost:11434/api/tags"
    ["ChromaDB"]="http://localhost:8001/api/v1/heartbeat"
    ["Qdrant"]="http://localhost:6333/health"
    ["Neo4j"]="http://localhost:7474"
)

services_healthy=0
total_services=${#SERVICES[@]}

for service in "${!SERVICES[@]}"; do
    if check_service "$service" "${SERVICES[$service]}"; then
        ((services_healthy++))
    fi
done

echo ""
echo "🔗 API ENDPOINT TESTING"
echo "----------------------"

# Critical API endpoints
declare -a API_TESTS=(
    "/health GET"
    "/agents GET"
    "/models GET"
    "/metrics GET"
    "/api/v1/system/status GET"
)

api_healthy=0
total_apis=${#API_TESTS[@]}

for api_test in "${API_TESTS[@]}"; do
    read -r endpoint method <<< "$api_test"
    if test_api_endpoint "$endpoint" "$method"; then
        ((api_healthy++))
    fi
done

# Test POST endpoints with data
echo -n "Testing API POST /simple-chat... "
if curl -s --max-time 30 -X POST -H "Content-Type: application/json" \
   -d '{"message":"Test message"}' \
   http://localhost:8000/simple-chat | grep -q "response"; then
    echo -e "${GREEN}✅ 200${NC}"
    ((api_healthy++))
else
    echo -e "${RED}❌ FAILED${NC}"
fi
((total_apis++))

echo ""
echo "🧪 INTEGRATION TESTING"
echo "---------------------"

# Run Python integration tests if available
if [ -f "/opt/sutazaiapp/scripts/test_frontend_integration.py" ]; then
    echo "Running comprehensive integration tests..."
    cd /opt/sutazaiapp
    python3 scripts/test_frontend_integration.py
else
    echo "⚠️  Integration test script not found"
fi

echo ""
echo "📊 DEPLOYMENT SUMMARY"
echo "===================="

# Calculate health scores
core_score=$((core_healthy * 100 / ${#CORE_CONTAINERS[@]}))
services_score=$((services_healthy * 100 / total_services))
api_score=$((api_healthy * 100 / total_apis))
overall_score=$(((core_score + services_score + api_score) / 3))

echo "🐳 Core Containers:    $core_healthy/${#CORE_CONTAINERS[@]} ($core_score%)"
echo "📡 Services Health:    $services_healthy/$total_services ($services_score%)"
echo "🔗 API Endpoints:      $api_healthy/$total_apis ($api_score%)"
echo "📈 Overall Health:     $overall_score%"

echo ""
echo "🌐 ACCESS POINTS"
echo "==============="
echo "Frontend:          http://localhost:8501"
echo "Backend API:       http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "LangFlow:          http://localhost:8090"
echo "FlowiseAI:         http://localhost:8099"
echo "BigAGI:            http://localhost:8106"
echo "Dify:              http://localhost:8107"
echo "n8n:               http://localhost:5678"
echo "Neo4j Browser:     http://localhost:7474"

echo ""
if [ $overall_score -ge 80 ]; then
    echo -e "${GREEN}🎉 EXCELLENT! SutazAI system is fully operational!${NC}"
    exit 0
elif [ $overall_score -ge 60 ]; then
    echo -e "${YELLOW}⚠️  GOOD! Most components working, some may need attention.${NC}"
    exit 0
else
    echo -e "${RED}🚨 CRITICAL! Multiple components need immediate attention.${NC}"
    echo ""
    echo "🔧 TROUBLESHOOTING STEPS:"
    echo "1. Check Docker containers: docker ps -a"
    echo "2. Check container logs: docker logs <container_name>"
    echo "3. Restart failed services: docker-compose restart <service>"
    echo "4. Full system restart: docker-compose down && docker-compose up -d"
    exit 1
fi