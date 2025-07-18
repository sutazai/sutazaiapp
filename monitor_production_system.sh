#!/bin/bash
# SutazAI Production System Monitor - Real-time Dashboard

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

while true; do
    clear
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}             SutazAI AGI/ASI Production System Monitor            ${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Time: $(date)${NC}"
    echo ""
    
    # Backend Health
    if curl -s http://localhost:8000/health | grep -q "healthy" 2>/dev/null; then
        echo -e "${GREEN}✅ Backend: Healthy${NC}"
    else
        echo -e "${RED}❌ Backend: Unhealthy${NC}"
    fi
    
    # Frontend Health
    if curl -s http://localhost:8501 | grep -q "Streamlit" 2>/dev/null; then
        echo -e "${GREEN}✅ Frontend: Accessible${NC}"
    else
        echo -e "${RED}❌ Frontend: Inaccessible${NC}"
    fi
    
    # System Metrics
    TOTAL_AGENTS=$(curl -s http://localhost:8000/api/system/complete_status | jq -r '.total_agents' 2>/dev/null || echo "0")
    ACTIVE_AGENTS=$(curl -s http://localhost:8000/api/system/complete_status | jq -r '.active_agents' 2>/dev/null || echo "0")
    
    echo -e "${BLUE}🤖 Total Agents: $TOTAL_AGENTS${NC}"
    echo -e "${GREEN}⚡ Active Agents: $ACTIVE_AGENTS${NC}"
    
    # Docker Containers Status
    echo ""
    echo -e "${BLUE}Docker Containers Status:${NC}"
    RUNNING_CONTAINERS=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep sutazai | wc -l)
    echo -e "${GREEN}🐳 Running Containers: $RUNNING_CONTAINERS${NC}"
    
    # Agent Health Status
    echo ""
    echo -e "${BLUE}Agent Health Status:${NC}"
    
    # LangChain
    if curl -s http://localhost:8084/health | grep -q "healthy" 2>/dev/null; then
        echo -e "${GREEN}✅ LangChain: Healthy${NC}"
    else
        echo -e "${RED}❌ LangChain: Unhealthy${NC}"
    fi
    
    # AutoGen
    if curl -s http://localhost:8085/health | grep -q "healthy" 2>/dev/null; then
        echo -e "${GREEN}✅ AutoGen: Healthy${NC}"
    else
        echo -e "${RED}❌ AutoGen: Unhealthy${NC}"
    fi
    
    # Browser Use
    if curl -s http://localhost:8088/health | grep -q "healthy" 2>/dev/null; then
        echo -e "${GREEN}✅ Browser Use: Healthy${NC}"
    else
        echo -e "${RED}❌ Browser Use: Unhealthy${NC}"
    fi
    
    # Mock Agents
    if curl -s http://localhost:8083/health | grep -q "healthy" 2>/dev/null; then
        echo -e "${GREEN}✅ Mock Agents: Healthy${NC}"
    else
        echo -e "${RED}❌ Mock Agents: Unhealthy${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Press Ctrl+C to exit monitoring${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    
    sleep 30
done
