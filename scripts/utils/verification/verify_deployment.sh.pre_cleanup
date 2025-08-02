#!/bin/bash
# SutazAI AGI/ASI Deployment Verification Script

set -e

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}===================================================${NC}"
echo -e "${GREEN}🔍 SutazAI Deployment Verification${NC}"
echo -e "${GREEN}===================================================${NC}"

# Function to check if a service is running
check_service() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_status"; then
        echo -e "  ✅ $service_name"
        return 0
    else
        echo -e "  ❌ $service_name"
        return 1
    fi
}

# Check Docker containers
echo -e "\n${YELLOW}📦 Checking Docker Containers...${NC}"
RUNNING_CONTAINERS=$(docker ps --format "table {{.Names}}" | grep sutazai | wc -l || echo 0)
TOTAL_EXPECTED=30
echo -e "  Running: $RUNNING_CONTAINERS/$TOTAL_EXPECTED expected"

# Check core services
echo -e "\n${YELLOW}🏗️ Checking Core Services...${NC}"
check_service "Backend API" "http://localhost:8000/health"
check_service "Frontend UI" "http://localhost:8501" "200\|302"
check_service "Ollama" "http://localhost:11434/api/tags"
check_service "LiteLLM Proxy" "http://localhost:4000/health"
check_service "Service Hub" "http://localhost:8114/health"

# Check Ollama models
echo -e "\n${YELLOW}🧠 Checking Ollama Models...${NC}"
MODEL_COUNT=$(curl -s http://localhost:11434/api/tags | jq '.models | length' 2>/dev/null || echo 0)
echo -e "  Models available: $MODEL_COUNT"
if [ "$MODEL_COUNT" -gt 0 ]; then
    echo -e "  First 3 models:"
    curl -s http://localhost:11434/api/tags | jq -r '.models[0:3][].name' 2>/dev/null | sed 's/^/    - /'
fi

# Check AI Agents (sample)
echo -e "\n${YELLOW}🤖 Checking AI Agents (sample)...${NC}"
check_service "AutoGPT" "http://localhost:8080/health"
check_service "CrewAI" "http://localhost:8096/health"
check_service "BigAGI" "http://localhost:8106" "200\|302"
check_service "Dify" "http://localhost:8107" "200\|302"

# Check vector databases
echo -e "\n${YELLOW}📊 Checking Vector Databases...${NC}"
check_service "ChromaDB" "http://localhost:8001/api/v1/heartbeat"
check_service "Qdrant" "http://localhost:6333/healthz"

# Test LiteLLM proxy
echo -e "\n${YELLOW}🔄 Testing LiteLLM Proxy...${NC}"
LITELLM_TEST=$(curl -s -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10
  }' | jq -r '.choices[0].message.content' 2>/dev/null || echo "Failed")

if [ "$LITELLM_TEST" != "Failed" ]; then
    echo -e "  ✅ OpenAI API compatibility working"
else
    echo -e "  ❌ OpenAI API compatibility failed"
fi

# Quick summary
echo -e "\n${GREEN}===================================================${NC}"
echo -e "${GREEN}📊 Quick Summary${NC}"
echo -e "${GREEN}===================================================${NC}"

# Count healthy services
HEALTHY_COUNT=0
TOTAL_CHECKED=0

# Simple health check
for port in 8000 8501 11434 4000 8114 8080 8096 8001 6333; do
    TOTAL_CHECKED=$((TOTAL_CHECKED + 1))
    if nc -z localhost $port 2>/dev/null; then
        HEALTHY_COUNT=$((HEALTHY_COUNT + 1))
    fi
done

HEALTH_PERCENTAGE=$((HEALTHY_COUNT * 100 / TOTAL_CHECKED))

echo -e "  Healthy Services: $HEALTHY_COUNT/$TOTAL_CHECKED"
echo -e "  Health Percentage: $HEALTH_PERCENTAGE%"

if [ $HEALTH_PERCENTAGE -ge 80 ]; then
    echo -e "\n${GREEN}✅ System is HEALTHY!${NC}"
    echo -e "${GREEN}===================================================${NC}"
    
    echo -e "\n${YELLOW}📌 Key Access Points:${NC}"
    echo -e "  • Main App: http://localhost:8501"
    echo -e "  • Service Hub: http://localhost:8114"
    echo -e "  • BigAGI: http://localhost:8106"
    echo -e "  • Dify: http://localhost:8107"
    echo -e "  • LangFlow: http://localhost:8090"
    echo -e "  • n8n: http://localhost:5678"
    
    echo -e "\n${YELLOW}📖 Documentation:${NC}"
    echo -e "  • Ollama Config: ./docs/OLLAMA_AGENT_CONFIGURATION.md"
    echo -e "  • Deployment Summary: ./DEPLOYMENT_SUMMARY.md"
    
    echo -e "\n${YELLOW}🧪 Run full tests:${NC}"
    echo -e "  python test_complete_agi_system.py"
    
elif [ $HEALTH_PERCENTAGE -ge 60 ]; then
    echo -e "\n${YELLOW}⚠️ System is PARTIALLY operational${NC}"
    echo -e "${YELLOW}Some services may still be starting up...${NC}"
else
    echo -e "\n${RED}❌ System health is CRITICAL${NC}"
    echo -e "${RED}Check logs: docker-compose logs${NC}"
fi

echo -e "\n${GREEN}===================================================${NC}"