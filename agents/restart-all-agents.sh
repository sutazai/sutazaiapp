#!/bin/bash

# SutazAI Agent Orchestration Restart Script
# Comprehensive restart with proper configuration

set -e

echo "================================================"
echo "SutazAI Agent Orchestration System Restart"
echo "================================================"

# Change to agents directory
cd /opt/sutazaiapp/agents

# Step 1: Stop all agent containers
echo ""
echo "Step 1: Stopping all agent containers..."
docker compose -f docker-compose-local-llm.yml down || true
docker compose -f docker-compose-phase2.yml down || true  
docker compose -f docker-compose-phase3.yml down || true
docker compose -f docker-compose-phase4.yml down || true

# Stop individual problem containers
docker stop sutazai-pentestgpt || true
docker rm sutazai-pentestgpt || true

# Step 2: Ensure Ollama is running with correct model
echo ""
echo "Step 2: Checking Ollama service..."
if ! docker ps | grep -q sutazai-ollama; then
    echo "Starting Ollama service..."
    docker compose -f docker-compose-local-llm.yml up -d ollama
    sleep 10
fi

# Pull TinyLlama model if not present
echo "Ensuring TinyLlama model is available..."
docker exec sutazai-ollama ollama pull tinyllama:latest || true

# Step 3: Start MCP Bridge if not running
echo ""
echo "Step 3: Checking MCP Bridge..."
if ! docker ps | grep -q sutazai-mcp-bridge; then
    echo "MCP Bridge not running. Please ensure it's started from the main docker compose."
    # Optionally start it here if needed
    cd /opt/sutazaiapp
    docker compose up -d mcp-bridge
    cd /opt/sutazaiapp/agents
    sleep 5
fi

# Step 4: Start agents in phases for resource management
echo ""
echo "Step 4: Starting Phase 1 agents (Core)..."
docker compose -f docker-compose-local-llm.yml up -d \
    crewai aider letta gpt-engineer finrobot shellgpt documind langchain

# Wait for phase 1 to stabilize
sleep 15

echo ""
echo "Step 5: Starting Phase 2 agents (Lightweight)..."
docker compose -f docker-compose-phase2.yml up -d \
    autogpt localagi agentzero bigagi semgrep autogen browseruse skyvern

# Wait for phase 2 to stabilize
sleep 15

echo ""
echo "Step 6: Starting Phase 3 agents (Specialized)..."
docker compose -f docker-compose-phase3.yml up -d pentestgpt

# Optional: Start TabbyML from Phase 4 if needed
# echo ""
# echo "Step 7: Starting Phase 4 agents (Heavy/GPU)..."
# docker compose -f docker-compose-phase4.yml up -d tabbyml

# Step 7: Wait for health checks
echo ""
echo "Step 7: Waiting for agents to become healthy..."
sleep 30

# Step 8: Check agent status
echo ""
echo "Step 8: Agent Status Report:"
echo "================================"

# Function to check agent health
check_agent() {
    local name=$1
    local port=$2
    
    if curl -s -f "http://localhost:${port}/health" > /dev/null 2>&1; then
        echo "✅ ${name}: HEALTHY (port ${port})"
    else
        echo "❌ ${name}: UNHEALTHY (port ${port})"
    fi
}

# Check all agents
check_agent "Letta" 11401
check_agent "AutoGPT" 11402
check_agent "CrewAI" 11403
check_agent "Aider" 11404
check_agent "LangChain" 11405
check_agent "LocalAGI" 11406
check_agent "BigAGI" 11407
check_agent "AgentZero" 11408
check_agent "Skyvern" 11409
check_agent "FinRobot" 11410
check_agent "PentestGPT" 11411
check_agent "ShellGPT" 11413
check_agent "Documind" 11414
check_agent "AutoGen" 11415
check_agent "GPT-Engineer" 11416
check_agent "Browser Use" 11703
check_agent "Semgrep" 11801

# Check Ollama
echo ""
echo "LLM Service Status:"
if curl -s "http://localhost:11435/api/tags" | grep -q "tinyllama"; then
    echo "✅ Ollama: RUNNING with TinyLlama model"
else
    echo "❌ Ollama: NOT RESPONDING or model missing"
fi

# Check MCP Bridge
if curl -s -f "http://localhost:11100/health" > /dev/null 2>&1; then
    echo "✅ MCP Bridge: HEALTHY"
else
    echo "❌ MCP Bridge: UNHEALTHY"
fi

# Step 9: Show container status
echo ""
echo "Container Status:"
echo "================================"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "sutazai-.*agent|sutazai-ollama|sutazai-mcp|NAMES" | head -20

# Step 10: Show any error logs
echo ""
echo "Recent Errors (if any):"
echo "================================"
for container in $(docker ps -a --format "{{.Names}}" | grep -E "sutazai-(letta|autogpt|crewai|aider|langchain|localagi|bigagi|agentzero|skyvern|finrobot|pentestgpt|shellgpt|documind|autogen|gpt-engineer|browseruse|semgrep)"); do
    errors=$(docker logs "$container" --tail 5 2>&1 | grep -E "ERROR|Error|Failed" | head -2)
    if [ ! -z "$errors" ]; then
        echo "$container:"
        echo "$errors"
        echo ""
    fi
done

echo ""
echo "================================================"
echo "Agent restart complete!"
echo "================================================"
echo ""
echo "To monitor agent logs:"
echo "  docker compose -f docker-compose-local-llm.yml logs -f [service-name]"
echo ""
echo "To check individual agent health:"
echo "  curl http://localhost:[port]/health"
echo ""
echo "To view MCP Bridge status:"
echo "  curl http://localhost:11100/agents"