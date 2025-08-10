#!/bin/bash
# SutazAI System Status Check Script

set -e

# Colors

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

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================="
echo "SutazAI System Status Check"
echo "========================================="
echo ""

# Function to check service
check_service() {
    local name=$1
    local url=$2
    local expected=$3
    
    printf "%-20s" "$name:"
    
    if response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null); then
        if [ "$response" = "$expected" ]; then
            echo -e "${GREEN}✓ Running${NC}"
            return 0
        else
            echo -e "${RED}✗ Error (HTTP $response)${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ Not accessible${NC}"
        return 1
    fi
}

# Function to check API endpoint
check_api() {
    local name=$1
    local url=$2
    
    printf "  %-18s" "$name:"
    
    if response=$(curl -s "$url" 2>/dev/null); then
        if echo "$response" | grep -q "status" || echo "$response" | grep -q "models"; then
            echo -e "${GREEN}✓ OK${NC}"
        else
            echo -e "${YELLOW}⚠ Unexpected response${NC}"
        fi
    else
        echo -e "${RED}✗ Failed${NC}"
    fi
}

# Check Core Services
echo -e "${BLUE}Core Services:${NC}"
check_service "PostgreSQL" "http://localhost:5432" "000"
check_service "Redis" "http://localhost:6379" "000"
check_service "Backend API" "http://localhost:8000/health" "200"
check_service "Frontend UI" "http://localhost:8501" "200"

echo ""

# Check Vector Databases
echo -e "${BLUE}Vector Databases:${NC}"
check_service "ChromaDB" "http://localhost:8001/api/v1/heartbeat" "200"
check_service "Qdrant" "http://localhost:6333/health" "200"

echo ""

# Check AI Services
echo -e "${BLUE}AI Services:${NC}"
check_service "Ollama" "http://localhost:10104/api/tags" "200"

echo ""

# Check Monitoring
echo -e "${BLUE}Monitoring:${NC}"
check_service "Prometheus" "http://localhost:9090/-/healthy" "200"
check_service "Grafana" "http://localhost:3000/api/health" "200"

echo ""

# Check Backend APIs
echo -e "${BLUE}Backend APIs:${NC}"
check_api "System Status" "http://localhost:8000/api/v1/system/status"
check_api "Models API" "http://localhost:8000/api/v1/models/status"
check_api "Vectors API" "http://localhost:8000/api/v1/vectors/stats"
check_api "Chat API" "http://localhost:8000/api/v1/chat"

echo ""

# Check Docker Containers
echo -e "${BLUE}Docker Containers:${NC}"
running_containers=$(docker ps --filter "name=sutazai" --format "table {{.Names}}\t{{.Status}}" | tail -n +2)
while IFS= read -r line; do
    if echo "$line" | grep -q "healthy\|Up"; then
        echo -e "  ${GREEN}✓${NC} $line"
    else
        echo -e "  ${RED}✗${NC} $line"
    fi
done <<< "$running_containers"

echo ""

# System Resources
echo -e "${BLUE}System Resources:${NC}"
echo "  CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "  Memory: $(free -h | awk '/^Mem:/ {print $3 " / " $2}')"
echo "  Disk: $(df -h / | awk 'NR==2 {print $3 " / " $2 " (" $5 " used)"}')"

echo ""
echo "========================================="
echo -e "${GREEN}System Check Complete${NC}"
echo "========================================="

# Summary
total=$(docker ps --filter "name=sutazai" -q | wc -l)
healthy=$(docker ps --filter "name=sutazai" --format "{{.Status}}" | grep -c "healthy\|Up" || true)

if [ "$healthy" -eq "$total" ] && [ "$total" -gt 0 ]; then
    echo -e "${GREEN}All $total services are running!${NC}"
else
    echo -e "${YELLOW}$healthy of $total services are healthy${NC}"
fi

echo ""
echo "Access points:"
echo "- Frontend: http://localhost:8501"
echo "- Backend API: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"