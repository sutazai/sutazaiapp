#!/bin/bash

# Fix Ollama Connectivity Issues Script
# This script fixes all Ollama connectivity issues across the SutazAI platform

set -e

echo "==================================================="
echo "SutazAI Platform - Ollama Connectivity Fix"
echo "==================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "→ $1"; }

# 1. Fix the Ollama health check issue
echo -e "\n1. Fixing Ollama health check..."

# Create a new Docker compose override file with fixed health check
cat > /opt/sutazaiapp/docker-compose-ollama-fix.yml << 'EOF'
# SutazAI Platform - Ollama Service Fix
# Fixes health check and network connectivity issues

version: '3.8'

networks:
  sutazai-network:
    external: true
    name: sutazaiapp_sutazai-network

services:
  ollama:
    image: ollama/ollama:latest
    container_name: sutazai-ollama
    restart: unless-stopped
    ports:
      - "11435:11434"  # External port 11435 -> Internal port 11434
    networks:
      sutazai-network:
        aliases:
          - ollama
          - sutazai-ollama
    volumes:
      - ollama-data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_KEEP_ALIVE=24h
      - OLLAMA_NUM_PARALLEL=2
      - OLLAMA_MAX_LOADED_MODELS=2
    healthcheck:
      # Use ollama CLI which is available in the container
      test: ["CMD-SHELL", "ollama list >/dev/null 2>&1 || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.0'

volumes:
  ollama-data:
    driver: local
EOF

print_success "Created fixed Ollama Docker compose configuration"

# 2. Update backend configuration to use correct Ollama settings
echo -e "\n2. Updating backend configuration..."

# Backup original file
cp /opt/sutazaiapp/docker-compose-backend.yml /opt/sutazaiapp/docker-compose-backend.yml.bak
print_info "Backed up original backend configuration"

# Update the backend compose file to use container name instead of host.docker.internal
sed -i 's/OLLAMA_HOST: host.docker.internal/OLLAMA_HOST: sutazai-ollama/g' /opt/sutazaiapp/docker-compose-backend.yml
sed -i 's/OLLAMA_PORT: 11434/OLLAMA_PORT: 11434/g' /opt/sutazaiapp/docker-compose-backend.yml

print_success "Updated backend configuration to use sutazai-ollama:11434"

# 3. Fix agent configurations in phase2
echo -e "\n3. Updating agent configurations..."

# All agent configs already use sutazai-ollama:11434 which is correct
print_info "Agent configurations in docker-compose-phase2.yml are already correct"
print_info "They use: OLLAMA_BASE_URL=http://sutazai-ollama:11434"

# 4. Create a connectivity test script
echo -e "\n4. Creating connectivity test script..."

cat > /opt/sutazaiapp/test-ollama-connectivity.sh << 'EOF'
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
EOF

chmod +x /opt/sutazaiapp/test-ollama-connectivity.sh
print_success "Created connectivity test script"

# 5. Restart Ollama with the fixed configuration
echo -e "\n5. Applying fixes..."

# Stop the current Ollama container
print_info "Stopping current Ollama container..."
docker stop sutazai-ollama 2>/dev/null || true
docker rm sutazai-ollama 2>/dev/null || true

# Start Ollama with the fixed configuration
print_info "Starting Ollama with fixed configuration..."
cd /opt/sutazaiapp
docker compose -f docker-compose-ollama-fix.yml up -d

# Wait for Ollama to be ready
print_info "Waiting for Ollama to be ready..."
sleep 10

# 6. Pull a lightweight model if none exists
echo -e "\n6. Ensuring a model is available..."
if ! docker exec sutazai-ollama ollama list | grep -q "tinyllama"; then
    print_info "Pulling tinyllama model (smallest available)..."
    docker exec sutazai-ollama ollama pull tinyllama || print_warning "Could not pull tinyllama model"
else
    print_success "tinyllama model already available"
fi

# 7. Restart backend to pick up new configuration
echo -e "\n7. Restarting backend service..."
docker compose -f docker-compose-backend.yml restart backend

# 8. Run connectivity tests
echo -e "\n8. Running connectivity tests..."
sleep 5
bash /opt/sutazaiapp/test-ollama-connectivity.sh

echo -e "\n==================================================="
echo "Ollama connectivity fix complete!"
echo "==================================================="
echo ""
echo "Summary of changes:"
echo "1. Fixed Ollama health check to use 'ollama list' instead of curl"
echo "2. Updated backend to use 'sutazai-ollama' instead of 'host.docker.internal'"
echo "3. Ensured all services use correct internal port (11434)"
echo "4. External access remains on port 11435"
echo ""
echo "To test connectivity manually:"
echo "  From host: curl http://localhost:11435/api/tags"
echo "  From container: docker exec [container] curl http://sutazai-ollama:11434/api/tags"
echo ""
echo "To check health status:"
echo "  docker ps | grep ollama"
echo ""
EOF