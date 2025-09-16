#!/bin/bash
# Test script for local LLM agents

echo "==========================================="
echo "Testing SutazAI Local LLM Agents"
echo "==========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test function
test_endpoint() {
    local name=$1
    local port=$2
    local endpoint=${3:-"/health"}
    
    echo -n "Testing $name (port $port)... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port$endpoint 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✅ HEALTHY${NC}"
        # Get detailed response
        curl -s http://localhost:$port$endpoint | jq -c 2>/dev/null || echo "Response received"
    elif [ "$response" = "000" ]; then
        echo -e "${YELLOW}⏳ STARTING${NC}"
    else
        echo -e "${RED}❌ ERROR (HTTP $response)${NC}"
    fi
}

echo "=== Health Check Results ==="
echo ""

# Test all agents
test_endpoint "CrewAI" 11401
test_endpoint "Aider" 11301
test_endpoint "Letta (MemGPT)" 11101
test_endpoint "GPT-Engineer" 11302
test_endpoint "FinRobot" 11601
test_endpoint "ShellGPT" 11701
test_endpoint "Documind" 11502
test_endpoint "LangChain" 11201

echo ""
echo "=== Testing Ollama Connection ==="
echo -n "Ollama Status: "
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✅ CONNECTED${NC}"
    echo -n "Available Model: "
    curl -s http://localhost:11434/api/tags | jq -r '.models[0].name' 2>/dev/null || echo "Unknown"
else
    echo -e "${RED}❌ NOT CONNECTED${NC}"
fi

echo ""
echo "=== Container Status ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "NAMES|sutazai-(crewai|aider|letta|gpt-engineer|finrobot|documind|shellgpt|langchain)"

echo ""
echo "==========================================="
echo "Test Complete"
echo "==========================================="