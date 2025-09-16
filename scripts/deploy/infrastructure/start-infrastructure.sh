#!/bin/bash

# SutazAI Platform - Infrastructure Startup Script
# Starts all services in the correct order with health checks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================================"
echo "SutazAI Platform Infrastructure Startup"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a service is healthy
check_service() {
    local service=$1
    local port=$2
    local endpoint=$3
    local max_retries=30
    local retry_count=0
    
    echo -n "Checking $service on port $port..."
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -s -f "http://localhost:$port$endpoint" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        retry_count=$((retry_count + 1))
    done
    
    echo -e " ${RED}✗${NC}"
    return 1
}

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    
    echo -n "Waiting for $service on port $port..."
    while ! nc -z localhost $port 2>/dev/null; do
        echo -n "."
        sleep 1
    done
    echo -e " ${GREEN}✓${NC}"
}

# Step 1: Start Core Infrastructure
echo "========================================="
echo "Phase 1: Core Infrastructure"
echo "========================================="
cd "$PROJECT_ROOT"

echo "Starting core services..."
docker-compose -f docker-compose-core.yml up -d

# Wait for core services
wait_for_service "PostgreSQL" 10000
wait_for_service "Redis" 10001
wait_for_service "RabbitMQ" 10004
wait_for_service "Neo4j" 10003
wait_for_service "Consul" 10006
wait_for_service "Kong" 10008

echo ""
echo "Core infrastructure started successfully!"
echo ""

# Step 2: Start Vector Databases
echo "========================================="
echo "Phase 2: Vector Databases"
echo "========================================="

if [ -f "$PROJECT_ROOT/docker-compose-vectors.yml" ]; then
    echo "Starting vector databases..."
    docker-compose -f docker-compose-vectors.yml up -d
    
    wait_for_service "ChromaDB" 10100
    wait_for_service "Qdrant" 10101
    wait_for_service "FAISS" 10103
    
    echo "Vector databases started successfully!"
else
    echo -e "${YELLOW}Vector databases compose file not found, skipping...${NC}"
fi

echo ""

# Step 3: Start Backend Services
echo "========================================="
echo "Phase 3: Backend Services"
echo "========================================="

if [ -f "$PROJECT_ROOT/docker-compose-backend.yml" ]; then
    echo "Starting backend services..."
    docker-compose -f docker-compose-backend.yml up -d
    
    wait_for_service "Backend API" 10200
    echo "Backend services started successfully!"
else
    echo "Starting backend manually..."
    cd "$PROJECT_ROOT/backend"
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
    nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 10200 --reload > backend.log 2>&1 &
    echo "Backend started in background (PID: $!)"
fi

echo ""

# Step 4: Start Ollama and MCP Bridge
echo "========================================="
echo "Phase 4: AI Infrastructure"
echo "========================================="

echo "Starting Ollama and MCP Bridge..."
cd "$PROJECT_ROOT/agents"
docker-compose -f docker-compose-network-fix.yml up -d

wait_for_service "Ollama" 11434
wait_for_service "MCP Bridge" 11100

echo "AI infrastructure started successfully!"
echo ""

# Step 5: Register Services with Consul
echo "========================================="
echo "Phase 5: Service Registration"
echo "========================================="

if [ -f "$SCRIPT_DIR/register-consul-services.sh" ]; then
    echo "Registering services with Consul..."
    bash "$SCRIPT_DIR/register-consul-services.sh"
else
    echo -e "${YELLOW}Consul registration script not found, skipping...${NC}"
fi

echo ""

# Step 6: Configure Kong API Gateway
echo "========================================="
echo "Phase 6: API Gateway Configuration"
echo "========================================="

if [ -f "$SCRIPT_DIR/configure-kong-routes.sh" ]; then
    echo "Configuring Kong API Gateway..."
    bash "$SCRIPT_DIR/configure-kong-routes.sh"
else
    echo -e "${YELLOW}Kong configuration script not found, skipping...${NC}"
fi

echo ""

# Step 7: Start AI Agents
echo "========================================="
echo "Phase 7: AI Agents"
echo "========================================="

echo "Starting AI agents..."
cd "$PROJECT_ROOT/agents"
docker-compose -f docker-compose-phase2.yml up -d

echo "Waiting for agents to initialize..."
sleep 10

echo ""

# Step 8: Start Frontend
echo "========================================="
echo "Phase 8: Frontend"
echo "========================================="

if [ -f "$PROJECT_ROOT/docker-compose-frontend.yml" ]; then
    echo "Starting frontend services..."
    cd "$PROJECT_ROOT"
    docker-compose -f docker-compose-frontend.yml up -d
    wait_for_service "Frontend" 11000
else
    echo "Starting frontend manually..."
    cd "$PROJECT_ROOT/frontend"
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
    nohup streamlit run app/main.py --server.port 11000 --server.address 0.0.0.0 > streamlit.log 2>&1 &
    echo "Frontend started in background (PID: $!)"
fi

echo ""
echo "========================================="
echo "Infrastructure Health Check"
echo "========================================="

# Perform health checks
echo ""
echo "Service Health Status:"
echo "----------------------"

check_service "PostgreSQL" 10000 "/" || true
check_service "Redis" 10001 "/" || true
check_service "RabbitMQ Management" 10005 "/" || true
check_service "Neo4j" 10002 "/" || true
check_service "Consul" 10006 "/v1/status/leader" || true
check_service "Kong Gateway" 10008 "/" || true
check_service "Kong Admin" 10009 "/status" || true
check_service "Backend API" 10200 "/health" || true
check_service "MCP Bridge" 11100 "/health" || true
check_service "Ollama" 11434 "/api/tags" || true
check_service "Frontend" 11000 "/" || true

echo ""
echo "========================================="
echo "Infrastructure Status Summary"
echo "========================================="

# Show running containers
echo ""
echo "Running Docker Containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "================================================"
echo -e "${GREEN}Infrastructure startup complete!${NC}"
echo "================================================"
echo ""
echo "Access Points:"
echo "- Frontend: http://localhost:11000"
echo "- Backend API: http://localhost:10200"
echo "- Kong Gateway: http://localhost:10008"
echo "- Kong Admin: http://localhost:10009"
echo "- Consul UI: http://localhost:10006"
echo "- RabbitMQ Management: http://localhost:10005"
echo "- Neo4j Browser: http://localhost:10002"
echo "- MCP Bridge: http://localhost:11100"
echo ""
echo "To stop all services, run: $SCRIPT_DIR/stop-infrastructure.sh"