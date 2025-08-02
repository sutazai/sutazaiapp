#\!/bin/bash
# Complete System Verification Script for SutazAI

echo "==============================================="
echo "SutazAI Complete System Verification"
echo "==============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Function to check service
check_service() {
    local service_name=$1
    local port=$2
    local url=$3
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    echo -n "Checking $service_name... "
    
    # Check if container is running
    if docker ps | grep -q "$service_name"; then
        # Check if port is accessible
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port$url" | grep -qE "200|301|302"; then
            echo -e "${GREEN}✓ Running${NC} (port $port)"
            PASSED_CHECKS=$((PASSED_CHECKS + 1))
        else
            echo -e "${YELLOW}⚠ Running but not accessible on port $port${NC}"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
        fi
    else
        echo -e "${RED}✗ Not running${NC}"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
}

# Function to count agents
count_agents() {
    local category=$1
    local pattern=$2
    
    local count=$(docker ps --format "{{.Names}}" | grep -E "$pattern" | wc -l)
    echo "$count"
}

echo "1. Core Services Check"
echo "----------------------"
check_service "sutazai-postgres" 5432 ""
check_service "sutazai-redis" 6379 ""
check_service "sutazai-ollama" 11434 ""
check_service "sutazai-backend" 8000 "/docs"
check_service "sutazai-frontend" 8501 ""
echo ""

echo "2. Vector Databases Check"
echo "-------------------------"
check_service "sutazai-chromadb" 8000 ""
check_service "sutazai-qdrant" 6333 ""
check_service "sutazai-neo4j" 7474 ""
echo ""

echo "3. Monitoring Stack Check"
echo "-------------------------"
check_service "sutazai-prometheus" 9090 ""
check_service "sutazai-grafana" 3000 ""
check_service "sutazai-loki" 3100 ""
echo ""

echo "4. Workflow Engines Check"
echo "-------------------------"
check_service "sutazai-n8n" 5678 ""
check_service "sutazai-jupyter" 8888 ""
# Check if workflow engines deployed
for engine in langflow flowise dify-api dify-web; do
    check_service "sutazai-$engine" 0 ""
done
echo ""

echo "5. AI Agents Summary"
echo "--------------------"
TOTAL_AGENTS=$(docker ps --format "{{.Names}}" | grep -E "agent|developer|engineer|specialist|coordinator|manager|optimizer|architect|improver|debugger|gpt|ai|crewai|aider|letta|devika|babyagi" | wc -l)

echo "Total AI Agents Running: $TOTAL_AGENTS"
echo ""
echo "By Category:"
echo "- Task Automation: $(count_agents "Task" "autogpt|agentgpt|crewai|agi|autonomous|task|coordinator|babyagi|letta")"
echo "- Code Generation: $(count_agents "Code" "code|developer|engineer|aider|gpt-engineer|devin|devika")"
echo "- Data Analysis: $(count_agents "Data" "data|analysis|pipeline|analyst")"
echo "- ML/AI: $(count_agents "ML" "model|training|learning|neural|quantum|federated")"
echo "- Infrastructure: $(count_agents "Infra" "infrastructure|devops|deployment|docker|kubernetes")"
echo "- Security: $(count_agents "Security" "security|pentest|semgrep|kali|shellgpt")"
echo "- Specialized: $(count_agents "Special" "special|manager|optimizer|architect|improver")"
echo ""

echo "6. System Resources"
echo "-------------------"
# Get system stats
CPU_USAGE=$(docker stats --no-stream --format "table {{.CPUPerc}}" | grep -v CPU | awk '{sum+=$1} END {print sum}')
MEM_USAGE=$(free -h | awk '/^Mem:/ {print $3 "/" $2}')
DISK_USAGE=$(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 ")"}')
CONTAINERS=$(docker ps -q | wc -l)

echo "CPU Usage (all containers): ${CPU_USAGE}%"
echo "Memory Usage: $MEM_USAGE"
echo "Disk Usage: $DISK_USAGE"
echo "Total Containers: $CONTAINERS"
echo ""

echo "7. API Health Check"
echo "-------------------"
# Check backend API
echo -n "Backend API: "
API_HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null | jq -r '.status' 2>/dev/null || echo "offline")
if [ "$API_HEALTH" = "healthy" ] || [ "$API_HEALTH" = "ok" ]; then
    echo -e "${GREEN}✓ Healthy${NC}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo -e "${RED}✗ $API_HEALTH${NC}"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

# Check agents endpoint
echo -n "Agents API: "
AGENTS_COUNT=$(curl -s http://localhost:8000/agents 2>/dev/null | jq '.agents | length' 2>/dev/null || echo "0")
if [ "$AGENTS_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ $AGENTS_COUNT agents registered${NC}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo -e "${YELLOW}⚠ No agents registered in API${NC}"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
echo ""

echo "8. Recent Agent Deployments"
echo "---------------------------"
# Show recently created containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}" | grep -E "agent|developer|engineer" | head -10
echo ""

echo "==============================================="
echo "VERIFICATION SUMMARY"
echo "==============================================="
echo "Total Checks: $TOTAL_CHECKS"
echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
echo ""

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}✅ SYSTEM FULLY OPERATIONAL\!${NC}"
    echo "All components are running successfully."
else
    echo -e "${YELLOW}⚠️  SYSTEM PARTIALLY OPERATIONAL${NC}"
    echo "Some components may need attention."
fi

echo ""
echo "Access Points:"
echo "- Frontend: http://localhost:8501"
echo "- API Docs: http://localhost:8000/docs"
echo "- Grafana: http://localhost:3000"
echo "- n8n: http://localhost:5678"
echo ""
echo "Total AI Agents Deployed: $TOTAL_AGENTS"
echo "==============================================="
