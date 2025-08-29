#!/bin/bash

# Test Ollama connectivity from various services

echo "==================================================="
echo "Testing Ollama Connectivity"
echo "==================================================="

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Test from host
echo -e "\n1. Testing from host (port 11435):"
if curl -s http://localhost:11435/api/tags >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Host can reach Ollama on port 11435"
else
    echo -e "${RED}✗${NC} Host cannot reach Ollama on port 11435"
fi

# Test from backend container
echo -e "\n2. Testing from backend container:"
if docker exec sutazai-backend curl -s http://sutazai-ollama:11434/api/tags >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Backend can reach Ollama via container name"
else
    echo -e "${RED}✗${NC} Backend cannot reach Ollama"
fi

# Test from MCP bridge if running
echo -e "\n3. Testing from MCP bridge:"
if docker ps | grep -q sutazai-mcp-bridge; then
    if docker exec sutazai-mcp-bridge curl -s http://sutazai-ollama:11434/api/tags >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} MCP Bridge can reach Ollama"
    else
        echo -e "${RED}✗${NC} MCP Bridge cannot reach Ollama"
    fi
else
    echo "MCP Bridge not running, skipping..."
fi

# Check Ollama health status
echo -e "\n4. Checking Ollama health status:"
HEALTH_STATUS=$(docker inspect sutazai-ollama --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
if [ "$HEALTH_STATUS" = "healthy" ]; then
    echo -e "${GREEN}✓${NC} Ollama health check: $HEALTH_STATUS"
else
    echo -e "${RED}✗${NC} Ollama health check: $HEALTH_STATUS"
fi

# List available models
echo -e "\n5. Available Ollama models:"
docker exec sutazai-ollama ollama list 2>/dev/null || echo "Could not list models"

echo -e "\n==================================================="
