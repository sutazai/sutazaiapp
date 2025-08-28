#!/bin/bash
# Complete AI Agent Deployment Script for SutazAI Platform
# Deploys 30+ agents in 4 phases with resource management

set -e

echo "============================================"
echo "SutazAI Complete Agent Deployment"
echo "============================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check available memory
check_memory() {
    local available=$(free -g | awk 'NR==2 {print $7}')
    echo "Available RAM: ${available}GB"
    if [ "$available" -lt 3 ]; then
        echo -e "${RED}WARNING: Low memory available!${NC}"
        return 1
    fi
    return 0
}

# Deploy a phase
deploy_phase() {
    local phase=$1
    local file=$2
    local agents=$3
    
    echo -e "${BLUE}=== Deploying Phase $phase: $agents ===${NC}"
    
    if [ ! -f "$file" ]; then
        echo -e "${RED}Error: $file not found${NC}"
        return 1
    fi
    
    echo "Checking memory before deployment..."
    if ! check_memory; then
        echo -e "${YELLOW}Skipping Phase $phase due to low memory${NC}"
        return 1
    fi
    
    echo "Starting Phase $phase deployment..."
    docker compose -f "$file" up -d
    
    echo "Waiting for containers to initialize..."
    sleep 30
    
    # Check health
    echo "Checking container health..."
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep sutazai- || true
    
    echo -e "${GREEN}Phase $phase deployment complete${NC}"
    echo ""
}

# Test agent endpoints
test_agents() {
    echo -e "${BLUE}=== Testing Agent Endpoints ===${NC}"
    
    # Phase 1 agents (already deployed)
    local phase1_ports="11401 11301 11101 11302 11601 11701 11502 11201"
    # Phase 2 agents
    local phase2_ports="11102 11103 11105 11106 11801 11203 11701 11702"
    # Phase 3 agents
    local phase3_ports="11107 11108 11104 11501 11202 11404 11802 11303"
    # Phase 4 agents
    local phase4_ports="11402 11403 11304 11901 11902 11903"
    
    for port in $phase1_ports $phase2_ports $phase3_ports $phase4_ports; do
        response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health 2>/dev/null || echo "000")
        if [ "$response" = "200" ]; then
            echo -e "Port $port: ${GREEN}✅ HEALTHY${NC}"
        elif [ "$response" = "000" ]; then
            echo -e "Port $port: ${YELLOW}⏳ NOT READY${NC}"
        else
            echo -e "Port $port: ${RED}❌ ERROR ($response)${NC}"
        fi
    done
}

# Main deployment
main() {
    echo "Starting complete agent deployment..."
    echo "This will deploy 30+ AI agents in 4 phases"
    echo ""
    
    # Check Ollama
    echo "Checking Ollama status..."
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Ollama is running${NC}"
        model=$(curl -s http://localhost:11434/api/tags | jq -r '.models[0].name' 2>/dev/null || echo "Unknown")
        echo "Available model: $model"
    else
        echo -e "${RED}❌ Ollama is not running!${NC}"
        echo "Please ensure Ollama is running on port 11434"
        exit 1
    fi
    echo ""
    
    # Phase 1 - Check existing deployment
    echo -e "${BLUE}=== Phase 1: Core Agents (Already Deployed) ===${NC}"
    echo "8 agents already running:"
    echo "- CrewAI, Aider, Letta, GPT-Engineer"
    echo "- FinRobot, ShellGPT, Documind, LangChain"
    docker ps --format "{{.Names}}" | grep -E "crewai|aider|letta|gpt-engineer|finrobot|documind|shellgpt|langchain" | wc -l
    echo ""
    
    # Phase 2 - Lightweight agents
    if [ "$1" != "--skip-phase2" ]; then
        deploy_phase 2 "docker-compose-phase2.yml" "8 Lightweight Agents"
    fi
    
    # Phase 3 - Medium agents
    if [ "$1" != "--skip-phase3" ]; then
        deploy_phase 3 "docker-compose-phase3.yml" "8 Medium Agents"
    fi
    
    # Phase 4 - Heavy/GPU agents (optional)
    if [ "$1" == "--include-heavy" ]; then
        echo -e "${YELLOW}Phase 4 requires significant resources (5.5GB RAM)${NC}"
        read -p "Deploy Phase 4 heavy/GPU agents? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            deploy_phase 4 "docker-compose-phase4.yml" "6 Heavy/GPU Agents"
        fi
    else
        echo -e "${YELLOW}Skipping Phase 4 (heavy/GPU agents) - use --include-heavy to deploy${NC}"
    fi
    
    echo ""
    echo "============================================"
    echo "Deployment Summary"
    echo "============================================"
    
    # Count running containers
    total=$(docker ps --format "{{.Names}}" | grep sutazai- | wc -l)
    echo -e "Total agents deployed: ${GREEN}$total${NC}"
    
    # Show resource usage
    echo ""
    echo "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep sutazai- | head -10 || true
    
    echo ""
    echo "Testing all agent endpoints..."
    test_agents
    
    echo ""
    echo -e "${GREEN}Deployment complete!${NC}"
    echo "Use 'docker ps' to check container status"
    echo "Use './test_local_llm_agents.sh' to test health endpoints"
}

# Handle script arguments
case "$1" in
    --help)
        echo "Usage: $0 [OPTIONS]"
        echo "Options:"
        echo "  --skip-phase2      Skip Phase 2 deployment"
        echo "  --skip-phase3      Skip Phase 3 deployment"
        echo "  --include-heavy    Include Phase 4 heavy/GPU agents"
        echo "  --test-only        Only test endpoints, no deployment"
        exit 0
        ;;
    --test-only)
        test_agents
        exit 0
        ;;
    *)
        main "$1"
        ;;
esac