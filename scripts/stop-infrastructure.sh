#!/bin/bash

# SutazAI Platform - Infrastructure Shutdown Script
# Stops all services in reverse order

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================================"
echo "SutazAI Platform Infrastructure Shutdown"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Stop Frontend
echo "Stopping frontend services..."
cd "$PROJECT_ROOT"
if [ -f "docker-compose-frontend.yml" ]; then
    docker-compose -f docker-compose-frontend.yml down
else
    pkill -f "streamlit run" 2>/dev/null || true
fi

# Step 2: Stop AI Agents
echo "Stopping AI agents..."
cd "$PROJECT_ROOT/agents"
docker-compose -f docker-compose-phase2.yml down 2>/dev/null || true

# Step 3: Stop MCP Bridge and Ollama
echo "Stopping AI infrastructure..."
docker-compose -f docker-compose-network-fix.yml down 2>/dev/null || true

# Step 4: Stop Backend
echo "Stopping backend services..."
cd "$PROJECT_ROOT"
if [ -f "docker-compose-backend.yml" ]; then
    docker-compose -f docker-compose-backend.yml down
else
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
fi

# Step 5: Stop Vector Databases
echo "Stopping vector databases..."
if [ -f "$PROJECT_ROOT/docker-compose-vectors.yml" ]; then
    docker-compose -f docker-compose-vectors.yml down
fi

# Step 6: Stop Core Infrastructure
echo "Stopping core infrastructure..."
docker-compose -f docker-compose-core.yml down

echo ""
echo "================================================"
echo -e "${GREEN}All services stopped successfully!${NC}"
echo "================================================"

# Show any remaining containers
remaining=$(docker ps -q 2>/dev/null)
if [ ! -z "$remaining" ]; then
    echo ""
    echo -e "${YELLOW}Warning: Some containers are still running:${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}"
fi