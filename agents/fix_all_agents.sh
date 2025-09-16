#!/bin/bash

# SutazAI Platform - Fix All Agents Script
# This script fixes connectivity and deployment issues for all 16 AI agents

set -e

echo "================================================"
echo "SutazAI Platform - Fixing All AI Agents"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to agents directory
cd /opt/sutazaiapp/agents

# Function to check if a container is healthy
check_health() {
    local container=$1
    local port=$2
    local retries=5
    local wait_time=10
    
    echo -e "${YELLOW}Checking health of $container on port $port...${NC}"
    
    for i in $(seq 1 $retries); do
        if curl -f -s http://localhost:$port/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $container is healthy${NC}"
            return 0
        fi
        echo "  Attempt $i/$retries failed, waiting ${wait_time}s..."
        sleep $wait_time
    done
    
    echo -e "${RED}✗ $container failed health check${NC}"
    return 1
}

# Stop all existing agent containers
echo -e "${YELLOW}Stopping existing agent containers...${NC}"
docker compose -f docker-compose-local-llm.yml down || true
docker compose -f docker-compose-phase2.yml down || true
docker compose -f docker-compose-phase3.yml down || true
docker compose -f docker-compose-phase4.yml down || true

# Remove any stopped containers
docker rm -f $(docker ps -aq --filter "name=sutazai-letta" --filter "name=sutazai-autogpt" --filter "name=sutazai-crewai" --filter "name=sutazai-aider" --filter "name=sutazai-langchain" --filter "name=sutazai-localagi" --filter "name=sutazai-bigagi" --filter "name=sutazai-agentzero" --filter "name=sutazai-skyvern" --filter "name=sutazai-finrobot" --filter "name=sutazai-pentestgpt" --filter "name=sutazai-tabbyml" --filter "name=sutazai-shellgpt" --filter "name=sutazai-documind" --filter "name=sutazai-autogen" --filter "name=sutazai-gpt-engineer") 2>/dev/null || true

# Ensure Ollama is running
echo -e "${YELLOW}Checking Ollama service...${NC}"
if ! docker ps | grep -q sutazai-ollama; then
    echo -e "${YELLOW}Starting Ollama service...${NC}"
    docker compose -f docker-compose-local-llm.yml up -d ollama
    sleep 10
fi

# Ensure TinyLlama model is pulled
echo -e "${YELLOW}Ensuring TinyLlama model is available...${NC}"
docker exec sutazai-ollama ollama pull tinyllama || true

# Deploy agents in groups to manage resource usage
echo -e "${GREEN}Deploying Phase 1: Core Agents (Letta, CrewAI, Aider, LangChain, GPT-Engineer)${NC}"
docker compose -f docker-compose-local-llm.yml up -d letta crewai aider langchain gpt-engineer
sleep 15

echo -e "${GREEN}Deploying Phase 2: Lightweight Agents (AutoGPT, LocalAGI, BigAGI, AgentZero, AutoGen)${NC}"
docker compose -f docker-compose-phase2.yml up -d autogpt localagi bigagi agentzero autogen
sleep 15

echo -e "${GREEN}Deploying Phase 3: Specialized Agents (Skyvern, FinRobot, ShellGPT, Documind)${NC}"
docker compose -f docker-compose-phase2.yml up -d skyvern
docker compose -f docker-compose-local-llm.yml up -d finrobot shellgpt documind
sleep 15

echo -e "${GREEN}Deploying Phase 4: Security & Code Agents (PentestGPT, TabbyML)${NC}"
docker compose -f docker-compose-phase3.yml up -d pentestgpt
docker compose -f docker-compose-phase4.yml up -d tabbyml
sleep 15

# Health check all agents
echo ""
echo "================================================"
echo "Health Check Results:"
echo "================================================"

FAILED_AGENTS=()

# Check each agent
check_health "Letta (MemGPT)" 11401 || FAILED_AGENTS+=("Letta")
check_health "AutoGPT" 11402 || FAILED_AGENTS+=("AutoGPT")
check_health "CrewAI" 11403 || FAILED_AGENTS+=("CrewAI")
check_health "Aider" 11404 || FAILED_AGENTS+=("Aider")
check_health "LangChain" 11405 || FAILED_AGENTS+=("LangChain")
check_health "LocalAGI" 11406 || FAILED_AGENTS+=("LocalAGI")
check_health "BigAGI" 11407 || FAILED_AGENTS+=("BigAGI")
check_health "AgentZero" 11408 || FAILED_AGENTS+=("AgentZero")
check_health "Skyvern" 11409 || FAILED_AGENTS+=("Skyvern")
check_health "FinRobot" 11410 || FAILED_AGENTS+=("FinRobot")
check_health "PentestGPT" 11411 || FAILED_AGENTS+=("PentestGPT")
check_health "TabbyML" 11412 || FAILED_AGENTS+=("TabbyML")
check_health "ShellGPT" 11413 || FAILED_AGENTS+=("ShellGPT")
check_health "Documind" 11414 || FAILED_AGENTS+=("Documind")
check_health "AutoGen" 11415 || FAILED_AGENTS+=("AutoGen")
check_health "GPT-Engineer" 11416 || FAILED_AGENTS+=("GPT-Engineer")

echo ""
echo "================================================"
echo "Deployment Summary:"
echo "================================================"

if [ ${#FAILED_AGENTS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All 16 agents deployed successfully!${NC}"
else
    echo -e "${YELLOW}⚠ Some agents failed deployment:${NC}"
    for agent in "${FAILED_AGENTS[@]}"; do
        echo -e "${RED}  - $agent${NC}"
    done
    echo ""
    echo "To check logs for failed agents, use:"
    echo "docker logs sutazai-<agent-name>"
fi

# Show container status
echo ""
echo "Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep sutazai- | grep -E "(letta|autogpt|crewai|aider|langchain|localagi|bigagi|agentzero|skyvern|finrobot|pentestgpt|tabbyml|shellgpt|documind|autogen|gpt-engineer)"

echo ""
echo "================================================"
echo "Agent Ports Mapping:"
echo "================================================"
echo "Letta (MemGPT)  : http://localhost:11401"
echo "AutoGPT         : http://localhost:11402"
echo "CrewAI          : http://localhost:11403"
echo "Aider           : http://localhost:11404"
echo "LangChain       : http://localhost:11405"
echo "LocalAGI        : http://localhost:11406"
echo "BigAGI          : http://localhost:11407"
echo "AgentZero       : http://localhost:11408"
echo "Skyvern         : http://localhost:11409"
echo "FinRobot        : http://localhost:11410"
echo "PentestGPT      : http://localhost:11411"
echo "TabbyML         : http://localhost:11412"
echo "ShellGPT        : http://localhost:11413"
echo "Documind        : http://localhost:11414"
echo "AutoGen         : http://localhost:11415"
echo "GPT-Engineer    : http://localhost:11416"
echo "================================================"