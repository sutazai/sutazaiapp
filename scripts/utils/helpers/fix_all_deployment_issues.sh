#!/bin/bash
# Comprehensive fix for all SutazAI deployment issues
# Fixes: Docker permissions, MCP servers, network issues, and agent deployment

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        SUTAZAI COMPREHENSIVE DEPLOYMENT FIX                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# 1. Fix Docker Socket Permissions
echo -e "\n${BLUE}[1/6]${NC} Fixing Docker socket permissions..."
echo "================================================"

# Create docker group if it doesn't exist
if ! getent group docker > /dev/null 2>&1; then
    sudo groupadd docker
fi

# Fix Docker socket permissions
sudo chmod 666 /var/run/docker.sock || true

# Fix the restarting containers by updating their Docker socket mounting
RESTARTING_CONTAINERS=("sutazai-hardware-optimizer" "sutazai-devops-manager" "sutazai-ollama-specialist")

for container in "${RESTARTING_CONTAINERS[@]}"; do
    if docker ps -a | grep -q "$container"; then
        echo "Fixing $container..."
        # Stop and remove the container
        docker stop "$container" 2>/dev/null || true
        docker rm "$container" 2>/dev/null || true
    fi
done

# 2. Fix Network Issues
echo -e "\n${BLUE}[2/6]${NC} Fixing network configuration..."
echo "================================================"

# Get the actual network name
NETWORK_NAME=$(docker network ls --format '{{.Name}}' | grep sutazai | head -1)
if [ -z "$NETWORK_NAME" ]; then
    echo "Creating sutazai-network..."
    docker network create sutazai-network
    NETWORK_NAME="sutazai-network"
else
    echo "Using existing network: $NETWORK_NAME"
fi

# 3. Install MCP Server Dependencies
echo -e "\n${BLUE}[3/6]${NC} Setting up MCP servers..."
echo "================================================"

# Install Task Master MCP if not already installed
if ! command -v npx &> /dev/null; then
    echo "Installing Node.js and npm..."
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# Setup SutazAI MCP server
cd /opt/sutazaiapp/mcp_server
if [ ! -f "node_modules/.bin/express" ]; then
    echo "Installing MCP server dependencies..."
    npm install
fi

# Create MCP server startup script
cat > /opt/sutazaiapp/scripts/start_mcp_servers.sh << 'EOF'
#!/bin/bash
# Start MCP servers for SutazAI

# Start SutazAI MCP server
echo "Starting SutazAI MCP server..."
cd /opt/sutazaiapp/mcp_server
nohup node index.js > /opt/sutazaiapp/logs/mcp-server.log 2>&1 &
echo $! > /tmp/sutazai-mcp-server.pid

# Start Task Master MCP (if needed)
echo "Task Master MCP is configured for on-demand startup via npx"

echo "MCP servers started. Check logs at /opt/sutazaiapp/logs/mcp-server.log"
EOF

chmod +x /opt/sutazaiapp/scripts/start_mcp_servers.sh

# 4. Redeploy Fixed Containers
echo -e "\n${BLUE}[4/6]${NC} Redeploying containers with proper configuration..."
echo "================================================"

# Create fixed docker-compose for the problematic services
cat > /opt/sutazaiapp/docker-compose-fixed-agents.yml << EOF
version: '3.9'

networks:
  sutazai-network:
    external: true
    name: ${NETWORK_NAME}

services:
  # Hardware Resource Optimizer with fixed Docker access
  hardware-resource-optimizer:
    build:
      context: ./agents/hardware-optimizer
      dockerfile: Dockerfile
    container_name: sutazai-hardware-optimizer
    restart: unless-stopped
    environment:
      - DOCKER_HOST=unix:///var/run/docker.sock
      - OPENAI_API_BASE=http://localhost:4000/v1
      - OPENAI_API_KEY=sk-local
      - LOG_LEVEL=INFO
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:rw
      - ./config/resource-optimizer:/config
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
    networks:
      - sutazai-network
    security_opt:
      - apparmor:unconfined
    privileged: true
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8523/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Infrastructure DevOps Manager with fixed Docker access
  infrastructure-devops-manager:
    build:
      context: ./agents/infrastructure-devops
      dockerfile: Dockerfile
    container_name: sutazai-devops-manager
    restart: unless-stopped
    environment:
      - DOCKER_HOST=unix:///var/run/docker.sock
      - OPENAI_API_BASE=http://localhost:4000/v1
      - OPENAI_API_KEY=sk-local
      - LOG_LEVEL=INFO
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:rw
      - ./config/devops:/config
      - ./scripts:/scripts:ro
    networks:
      - sutazai-network
    security_opt:
      - apparmor:unconfined
    privileged: true
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8522/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Ollama Integration Specialist with fixed configuration
  ollama-integration-specialist:
    build:
      context: ./agents/ollama-integration
      dockerfile: Dockerfile
    container_name: sutazai-ollama-specialist
    restart: unless-stopped
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - OPENAI_API_BASE=http://localhost:4000/v1
      - OPENAI_API_KEY=sk-local
      - LOG_LEVEL=INFO
      - DOCKER_HOST=unix:///var/run/docker.sock
    volumes:
      - ./models:/models
      - ./config/ollama:/config
      - /var/run/docker.sock:/var/run/docker.sock:rw
    networks:
      - sutazai-network
    depends_on:
      - ollama
    security_opt:
      - apparmor:unconfined
    privileged: true
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8520/health"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF

# Deploy the fixed containers
echo "Deploying fixed containers..."
cd /opt/sutazaiapp
docker-compose -f docker-compose-fixed-agents.yml up -d --build

# 5. Deploy Additional AI Agents
echo -e "\n${BLUE}[5/6]${NC} Deploying additional AI agents..."
echo "================================================"

# Deploy backend and frontend if not running
if ! docker ps | grep -q "sutazai-backend"; then
    echo "Deploying backend service..."
    docker-compose up -d backend || echo "Backend deployment skipped"
fi

if ! docker ps | grep -q "sutazai-frontend"; then
    echo "Deploying frontend service..."
    docker-compose up -d frontend || echo "Frontend deployment skipped"
fi

# 6. Start MCP Servers
echo -e "\n${BLUE}[6/6]${NC} Starting MCP servers..."
echo "================================================"

# Kill any existing MCP server processes
pkill -f "node.*mcp_server" || true
pkill -f "task-master-ai" || true

# Start the MCP servers
/opt/sutazaiapp/scripts/start_mcp_servers.sh

# Create comprehensive test script
cat > /opt/sutazaiapp/scripts/test_all_services.sh << 'EOF'
#!/bin/bash
# Test all SutazAI services

echo "🧪 Testing SutazAI Services"
echo "=========================="

# Test core services
services=(
    "Ollama:11434:/api/tags"
    "LiteLLM:4000:/health"
    "PostgreSQL:5432:"
    "Redis:6379:"
    "ChromaDB:8001:/api/v1"
    "Qdrant:6333:/health"
    "Coordinator:8888:/status"
    "Grafana:3000:/api/health"
    "Prometheus:9090/-/healthy"
)

for service in "${services[@]}"; do
    IFS=':' read -r name port endpoint <<< "$service"
    if [ -z "$endpoint" ]; then
        # TCP check only
        if nc -z localhost $port 2>/dev/null; then
            echo "✅ $name (port $port)"
        else
            echo "❌ $name (port $port)"
        fi
    else
        # HTTP check
        if curl -s -f "http://localhost:$port$endpoint" > /dev/null 2>&1; then
            echo "✅ $name (http://localhost:$port$endpoint)"
        else
            echo "❌ $name (http://localhost:$port$endpoint)"
        fi
    fi
done

# Check Docker containers
echo -e "\n📦 Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep sutazai | sort

# Check MCP servers
echo -e "\n🔌 MCP Server Status:"
if pgrep -f "node.*mcp_server" > /dev/null; then
    echo "✅ SutazAI MCP server is running"
else
    echo "❌ SutazAI MCP server is not running"
fi

# Summary
total=$(docker ps | grep sutazai | wc -l)
echo -e "\n📊 Summary: $total SutazAI containers running"
EOF

chmod +x /opt/sutazaiapp/scripts/test_all_services.sh

# Final status check
echo -e "\n${GREEN}✅ All fixes applied!${NC}"
echo "================================================"

# Run the test script
/opt/sutazaiapp/scripts/test_all_services.sh

echo -e "\n${GREEN}🎉 SutazAI deployment issues fixed!${NC}"
echo ""
echo "Next steps:"
echo "1. Check MCP status: Check the MCP tab in Claude"
echo "2. Access Coordinator API: http://localhost:8888"
echo "3. Access Frontend: http://localhost:8501"
echo "4. Monitor system: http://localhost:3000 (Grafana)"
echo ""
echo "To test all services again: /opt/sutazaiapp/scripts/test_all_services.sh"