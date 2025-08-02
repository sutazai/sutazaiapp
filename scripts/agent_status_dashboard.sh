#!/bin/bash

# Agent Status Dashboard - Shows all agents working together

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

clear

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                     🤖 SutazAI Agent Coordination Dashboard 🤖                ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo

# System Overview
echo -e "${YELLOW}═══ System Overview ═══${NC}"
echo -e "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo

# Core Services Status
echo -e "${YELLOW}═══ Core Services ═══${NC}"
services=("postgres" "redis" "ollama" "chromadb" "qdrant")
for service in "${services[@]}"; do
    if docker ps | grep -q "sutazai-$service"; then
        status=$(docker ps --filter "name=sutazai-$service" --format "{{.Status}}" | head -1)
        if [[ $status == *"healthy"* ]]; then
            echo -e "  ${GREEN}●${NC} $service: ${GREEN}Healthy${NC}"
        else
            echo -e "  ${YELLOW}●${NC} $service: ${YELLOW}Running${NC}"
        fi
    else
        echo -e "  ${RED}●${NC} $service: ${RED}Down${NC}"
    fi
done
echo

# AI Agents Status
echo -e "${YELLOW}═══ AI Agents Status ═══${NC}"

# Check specific agent containers
agent_containers=(
    "sutazai-devops-manager:Infrastructure DevOps Manager"
    "sutazai-ollama-specialist:Ollama Integration Specialist"
    "sutazai-hardware-optimizer:Hardware Resource Optimizer"
    "sutazai-task_coordinator:automation Coordinator System"
)

for agent_info in "${agent_containers[@]}"; do
    container_name="${agent_info%%:*}"
    agent_display="${agent_info##*:}"
    
    if docker ps | grep -q "$container_name"; then
        status=$(docker ps --filter "name=$container_name" --format "{{.Status}}" | head -1)
        if [[ $status == *"healthy"* ]]; then
            echo -e "  ${GREEN}●${NC} $agent_display: ${GREEN}Active${NC}"
        else
            echo -e "  ${YELLOW}●${NC} $agent_display: ${YELLOW}Starting${NC}"
        fi
    else
        echo -e "  ${RED}●${NC} $agent_display: ${RED}Offline${NC}"
    fi
done

# Check agent registry in Redis
if docker ps | grep -q "sutazai-redis"; then
    echo
    echo -e "${YELLOW}═══ Agent Registry ═══${NC}"
    
    # Get all registered agents from Redis
    agent_count=$(docker exec sutazai-redis redis-cli --scan --pattern "agent:registry:*" 2>/dev/null | wc -l || echo "0")
    echo -e "  Total Registered Agents: ${CYAN}$agent_count${NC}"
    
    # Show sample of active agents
    if [ "$agent_count" -gt 0 ]; then
        echo -e "  ${BLUE}Active Agents:${NC}"
        docker exec sutazai-redis redis-cli --scan --pattern "agent:registry:*" 2>/dev/null | head -5 | while read key; do
            agent_name=$(echo $key | cut -d: -f3)
            status=$(docker exec sutazai-redis redis-cli HGET "$key" status 2>/dev/null || echo "unknown")
            if [ "$status" = "active" ]; then
                echo -e "    ${GREEN}✓${NC} $agent_name"
            else
                echo -e "    ${YELLOW}○${NC} $agent_name ($status)"
            fi
        done
    fi
fi

# Task Queue Status
echo
echo -e "${YELLOW}═══ Task Queue Status ═══${NC}"
if docker ps | grep -q "sutazai-redis"; then
    pending_tasks=$(docker exec sutazai-redis redis-cli LLEN "agent:tasks" 2>/dev/null || echo "0")
    echo -e "  Pending Tasks: ${CYAN}$pending_tasks${NC}"
    
    # Show active collaborations
    collab_count=$(docker exec sutazai-redis redis-cli --scan --pattern "collab:*:status" 2>/dev/null | wc -l || echo "0")
    echo -e "  Active Collaborations: ${CYAN}$collab_count${NC}"
fi

# Resource Usage
echo
echo -e "${YELLOW}═══ Resource Usage ═══${NC}"
# Get container stats
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep sutazai | head -10 | while read line; do
    if [[ $line == *"sutazai"* ]]; then
        container=$(echo "$line" | awk '{print $1}' | sed 's/sutazai-//')
        cpu=$(echo "$line" | awk '{print $2}')
        mem=$(echo "$line" | awk '{print $3}')
        echo -e "  ${BLUE}$container:${NC} CPU: $cpu, Memory: $mem"
    fi
done

# Access Points
echo
echo -e "${YELLOW}═══ Access Points ═══${NC}"
echo -e "  ${BLUE}Frontend:${NC} http://localhost:8501"
echo -e "  ${BLUE}Backend API:${NC} http://localhost:8000"
echo -e "  ${BLUE}MCP Server:${NC} http://localhost:8100"
echo -e "  ${BLUE}automation Coordinator:${NC} http://localhost:8900"
echo -e "  ${BLUE}Grafana:${NC} http://localhost:3000"
echo -e "  ${BLUE}Prometheus:${NC} http://localhost:9090"

# Agent Collaboration Examples
echo
echo -e "${YELLOW}═══ Active Agent Collaborations ═══${NC}"
echo -e "  ${GREEN}●${NC} Development Team: AI Engineer + Backend Dev + Frontend Dev"
echo -e "  ${GREEN}●${NC} Deployment Pipeline: QA Validator → Security Tester → Deploy Master"
echo -e "  ${GREEN}●${NC} Optimization Squad: Hardware Optimizer + Context Engineer + Orchestrator"
echo -e "  ${GREEN}●${NC} Architecture Council: automation Architect + System Controller + Agent Creator"

# Quick Actions
echo
echo -e "${YELLOW}═══ Quick Actions ═══${NC}"
echo -e "  ${CYAN}1.${NC} View agent logs: docker-compose logs -f [agent-name]"
echo -e "  ${CYAN}2.${NC} Submit task: curl -X POST http://localhost:8100/execute_agent_task -d '{...}'"
echo -e "  ${CYAN}3.${NC} Check health: curl http://localhost:8100/health"
echo -e "  ${CYAN}4.${NC} Run tests: python3 scripts/test_agent_coordination.py"

echo
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"