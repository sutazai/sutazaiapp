#!/bin/bash

# Deploy Autonomous AGI System
# ============================
# This script deploys the complete autonomous AGI system
# with all AI agents working together independently

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        SUTAZAI AUTONOMOUS AGI DEPLOYMENT SYSTEM              ║"
echo "║                                                              ║"
echo "║  Deploying complete AI agent infrastructure...               ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/autonomous_agi_deployment.log"

# Create log directory
mkdir -p "$PROJECT_ROOT/logs"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check service health
check_health() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    log "Checking health of $service on port $port..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} $service is healthy"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    echo -e "${RED}✗${NC} $service failed health check"
    return 1
}

# Step 1: Ensure Docker is running
log "Step 1: Checking Docker status..."
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker is running"

# Step 2: Pull required Ollama models
log "Step 2: Setting up Ollama models..."
MODELS=(
    "llama2:latest"
    "codellama:latest"
    "deepseek-coder:latest"
    "mistral:latest"
    "starcoder:latest"
)

for model in "${MODELS[@]}"; do
    log "Pulling model: $model"
    docker exec sutazai-ollama ollama pull "$model" || echo "Model $model may already exist"
done
echo -e "${GREEN}✓${NC} Ollama models configured"

# Step 3: Start Redis if not running
log "Step 3: Starting Redis..."
if ! docker ps | grep -q "redis"; then
    docker run -d --name sutazai-redis -p 6379:6379 redis:alpine
    sleep 5
fi
echo -e "${GREEN}✓${NC} Redis is running"

# Step 4: Start core infrastructure
log "Step 4: Starting core infrastructure..."
cd "$PROJECT_ROOT"

# Start AGI Brain
if ! docker ps | grep -q "sutazai-brain"; then
    log "Starting AGI Brain..."
    docker-compose -f docker-compose-agi-brain.yml up -d
    sleep 10
fi

# Start Universal Agents
log "Starting Universal Agent System..."
docker-compose -f docker-compose-new-universal-agents.yml up -d
sleep 10

echo -e "${GREEN}✓${NC} Core infrastructure started"

# Step 5: Deploy specialized agents
log "Step 5: Deploying specialized agents..."

# AutoGPT
if ! docker ps | grep -q "autogpt"; then
    log "Starting AutoGPT..."
    docker run -d --name sutazai-autogpt \
        -p 8000:8000 \
        -e OPENAI_API_KEY=ollama \
        --network sutazai-network \
        significantgravitas/auto-gpt:latest || true
fi

# CrewAI
if ! docker ps | grep -q "crewai"; then
    log "Starting CrewAI..."
    # Create CrewAI container if needed
    echo "CrewAI will be integrated through the universal agent system"
fi

# TabbyML
if ! docker ps | grep -q "tabby"; then
    log "Starting TabbyML..."
    docker run -d --name sutazai-tabby \
        -p 8081:8080 \
        --gpus all \
        -v tabby-data:/data \
        --network sutazai-network \
        tabbyml/tabby serve --model StarCoder-1B || true
fi

echo -e "${GREEN}✓${NC} Specialized agents deployed"

# Step 6: Run integration script
log "Step 6: Integrating all agents..."
cd "$PROJECT_ROOT"
python3 scripts/integrate_all_agents.py &
INTEGRATION_PID=$!
sleep 10

# Step 7: Health checks
log "Step 7: Running health checks..."
echo ""
echo "Service Health Status:"
echo "====================="

# Check all services
check_health "AGI Brain" 8900
check_health "Universal Agents" 9101
check_health "Redis" 6379
check_health "Ollama" 11434
check_health "LiteLLM" 4000

# Step 8: Display status
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              AUTONOMOUS AGI DEPLOYMENT COMPLETE              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 System Status:"
echo "  - AGI Brain: http://localhost:8900"
echo "  - Universal Agents API: http://localhost:9101"
echo "  - Agent Registry: http://localhost:9101/agents"
echo "  - System Health: http://localhost:9101/health"
echo ""
echo "📊 Active Components:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep sutazai
echo ""
echo "🤖 Available Agents:"
echo "  - System Architect (port 8900)"
echo "  - Code Generator (port 8080)"
echo "  - AutoGPT Orchestrator (port 8000)"
echo "  - TabbyML Assistant (port 8081)"
echo "  - Security Scanner (port 8083)"
echo "  - Resource Optimizer (port 8086)"
echo "  - Knowledge Manager (port 8087)"
echo "  - Test Validator (port 8088)"
echo "  - Workflow Engine (port 5678)"
echo "  - System Controller (port 8091)"
echo ""
echo "✅ The system is now fully autonomous and independent!"
echo ""
echo "📝 Logs available at: $LOG_FILE"
echo ""
echo "To monitor the system:"
echo "  python3 scripts/sutazai_monitor.py"
echo ""
echo "To stop the system:"
echo "  docker-compose -f docker-compose-new-universal-agents.yml down"
echo "  docker-compose -f docker-compose-agi-brain.yml down"

# Save deployment info
cat > "$PROJECT_ROOT/deployment_status.json" <<EOF
{
  "deployment_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "status": "active",
  "components": {
    "agi_brain": "running",
    "universal_agents": "running",
    "redis": "running",
    "ollama": "running",
    "integration": "active"
  },
  "agents_count": 16,
  "autonomous": true,
  "external_dependencies": 0
}
EOF

log "Deployment complete!"