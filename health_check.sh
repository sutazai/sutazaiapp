#!/bin/bash
# SutazAI System Health Check Script
# Optimized for minimal deployment

echo "=== SutazAI System Health Check ==="
echo "Timestamp: $(date)"
echo

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check function
check_service() {
    local service_name=$1
    local url=$2
    local expected_code=${3:-200}
    
    if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_code"; then
        echo -e "${GREEN}✓${NC} $service_name is healthy"
        return 0
    else
        echo -e "${RED}✗${NC} $service_name is unhealthy"
        return 1
    fi
}

echo "=== Core Services ==="

# PostgreSQL
if docker exec sutazai-postgres-minimal pg_isready -U sutazai >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} PostgreSQL is healthy"
else
    echo -e "${RED}✗${NC} PostgreSQL is unhealthy"
fi

# Redis
if docker exec sutazai-redis-minimal redis-cli ping >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Redis is healthy"
else
    echo -e "${RED}✗${NC} Redis is unhealthy"
fi

# Ollama
check_service "Ollama" "http://localhost:10104/api/tags"

# Backend
check_service "Backend API" "http://localhost:8000/health"
check_service "Backend API v1" "http://localhost:8000/api/v1/health"

# Frontend
check_service "Frontend" "http://localhost:8501"

echo
echo "=== AI Agents ==="

# Check agent containers
agents=("sutazai-senior-ai-engineer" "sutazai-infrastructure-devops-manager" "sutazai-testing-qa-validator")

for agent in "${agents[@]}"; do
    if docker ps --format "{{.Names}}" | grep -q "$agent"; then
        echo -e "${GREEN}✓${NC} $agent is running"
    else
        echo -e "${RED}✗${NC} $agent is not running"
    fi
done

echo
echo "=== Resource Usage ==="

# Get resource usage
echo "Container Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | grep sutazai

echo
echo "=== System Summary ==="

# Count healthy services
total_services=8
healthy_count=0

# Recheck all services for summary
docker exec sutazai-postgres-minimal pg_isready -U sutazai >/dev/null 2>&1 && ((healthy_count++))
docker exec sutazai-redis-minimal redis-cli ping >/dev/null 2>&1 && ((healthy_count++))
curl -s -o /dev/null "http://localhost:10104/api/tags" && ((healthy_count++))
curl -s -o /dev/null "http://localhost:8000/health" && ((healthy_count++))
curl -s -o /dev/null "http://localhost:8501" && ((healthy_count++))

for agent in "${agents[@]}"; do
    docker ps --format "{{.Names}}" | grep -q "$agent" && ((healthy_count++))
done

if [ $healthy_count -eq $total_services ]; then
    echo -e "${GREEN}System Status: ALL SERVICES HEALTHY ($healthy_count/$total_services)${NC}"
elif [ $healthy_count -gt $((total_services / 2)) ]; then
    echo -e "${YELLOW}System Status: MOSTLY HEALTHY ($healthy_count/$total_services)${NC}"
else
    echo -e "${RED}System Status: UNHEALTHY ($healthy_count/$total_services)${NC}"
fi

echo
echo "=== Access URLs ==="
echo "• Frontend: http://localhost:8501"
echo "• Backend API: http://localhost:8000"
echo "• API Documentation: http://localhost:8000/docs"
echo "• Ollama API: http://localhost:10104"

echo
echo "=== Model Test ==="
echo "Testing tinyllama model..."
response=$(curl -s -X POST http://localhost:10104/api/generate -d '{"model": "tinyllama", "prompt": "Hello", "stream": false}' | jq -r .response 2>/dev/null)
if [ -n "$response" ] && [ "$response" != "null" ]; then
    echo -e "${GREEN}✓${NC} GPT-OSS model is working"
    echo "  Response preview: ${response:0:50}..."
else
    echo -e "${RED}✗${NC} GPT-OSS model is not responding"
fi

echo
echo "=== Health Check Complete ==="