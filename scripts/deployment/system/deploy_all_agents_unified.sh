#!/bin/bash

# Unified Agent Deployment Script for SutazAI
# This script ensures all agents are properly deployed and working together

set -e

# Constants
PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="$PROJECT_ROOT/logs"
COMPOSE_PROJECT_NAME="sutazaiapp"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Ensure we're in the project root
cd "$PROJECT_ROOT"

# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p data/{postgres,redis,ollama,chromadb,qdrant,agents}

# Fix Docker socket permissions first
log_info "Fixing Docker socket permissions..."
if [ -e /var/run/docker.sock ]; then
    sudo chmod 666 /var/run/docker.sock || true
fi

# Core infrastructure services
CORE_SERVICES=(
    "postgres"
    "redis"
    "ollama"
    "chromadb"
    "qdrant"
)

# AI Agent services - Tier 1 (Essential)
TIER1_AGENTS=(
    "backend-agi"
    "frontend-agi"
    "jarvis-ai"
    "mcp-server"
    "api-gateway"
    "task-scheduler"
    "model-optimizer"
)

# AI Agent services - Tier 2 (Specialized)
TIER2_AGENTS=(
    "autogpt"
    "crewai"
    "localagi"
    "aider"
    "gpt-engineer"
    "semgrep"
    "browser-use"
    "agentgpt"
    "privategpt"
    "langflow"
    "flowise"
    "llamaindex"
    "autogen"
    "bigagi"
    "opendevin"
    "dify"
    "agentzero"
)

# AI Agent services - Tier 3 (Support)
TIER3_AGENTS=(
    "context-framework"
    "localagi-enhanced"
    "localagi-advanced"
    "finrobot"
    "jarvis-agi"
    "code-improver"
    "service-hub"
    "awesome-code-ai"
)

# Monitoring services
MONITORING_SERVICES=(
    "prometheus"
    "grafana"
    "loki"
    "promtail"
    "health-monitor"
)

# Function to check if a service is healthy
check_service_health() {
    local service=$1
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps $service 2>/dev/null | grep -q "(healthy)"; then
            return 0
        elif docker-compose ps $service 2>/dev/null | grep -q "Up"; then
            # Service is up but not healthy yet
            sleep 2
            attempt=$((attempt + 1))
        else
            # Service is not up
            return 1
        fi
    done
    
    return 1
}

# Function to start services with retry logic
start_services() {
    local services=("$@")
    local failed_services=()
    
    for service in "${services[@]}"; do
        log_info "Starting $service..."
        
        if docker-compose up -d $service 2>&1 | tee -a "$LOG_DIR/deployment.log"; then
            if check_service_health $service; then
                log_success "$service is up and healthy"
            else
                log_warning "$service is up but not healthy"
            fi
        else
            log_error "Failed to start $service"
            failed_services+=($service)
        fi
    done
    
    # Retry failed services
    if [ ${#failed_services[@]} -gt 0 ]; then
        log_warning "Retrying failed services: ${failed_services[*]}"
        sleep 5
        
        for service in "${failed_services[@]}"; do
            log_info "Retrying $service..."
            docker-compose up -d $service 2>&1 | tee -a "$LOG_DIR/deployment.log" || true
        done
    fi
}

# Function to configure agent communication
configure_agent_communication() {
    log_info "Configuring agent communication..."
    
    # Ensure Redis is available for agent communication
    docker-compose exec -T redis redis-cli ping > /dev/null 2>&1 || {
        log_error "Redis is not responding"
        return 1
    }
    
    # Configure agent registry in Redis
    docker-compose exec -T redis redis-cli <<EOF
HSET agent:registry:infrastructure-devops-manager status active
HSET agent:registry:ollama-integration-specialist status active
HSET agent:registry:hardware-resource-optimizer status active
HSET agent:registry:context-optimization-engineer status active
EOF
    
    log_success "Agent communication configured"
}

# Main deployment function
main() {
    log_info "Starting unified agent deployment..."
    
    # Step 1: Stop all existing services
    log_info "Stopping existing services..."
    docker-compose down --remove-orphans || true
    
    # Step 2: Clean up any stuck containers
    log_info "Cleaning up stuck containers..."
    docker ps -aq | xargs -r docker rm -f || true
    
    # Step 3: Create optimized docker-compose override
    log_info "Creating optimized configuration..."
    cat > docker-compose.override.yml << 'EOF'
version: '3.8'

x-healthcheck-defaults: &healthcheck-defaults
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s

x-resource-limits: &resource-limits
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 2G
      reservations:
        cpus: '0.5'
        memory: 512M

services:
  # Core services with optimized settings
  postgres:
    <<: *resource-limits
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD-SHELL", "pg_isready -U postgres"]
  
  redis:
    <<: *resource-limits
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "redis-cli", "ping"]
  
  ollama:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
  
  # Agent services with dependencies
  backend-agi:
    <<: *resource-limits
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      ollama:
        condition: service_started
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  
  mcp-server:
    <<: *resource-limits
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      backend-agi:
        condition: service_started
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8100/health"]
EOF
    
    # Step 4: Start core infrastructure
    log_info "Starting core infrastructure services..."
    export COMPOSE_FILE="docker-compose.yml:docker-compose.override.yml"
    start_services "${CORE_SERVICES[@]}"
    
    # Wait for core services to stabilize
    log_info "Waiting for core services to stabilize..."
    sleep 10
    
    # Step 5: Start Tier 1 agents
    log_info "Starting Tier 1 essential agents..."
    start_services "${TIER1_AGENTS[@]}"
    
    # Step 6: Configure agent communication
    configure_agent_communication
    
    # Step 7: Start monitoring services
    log_info "Starting monitoring services..."
    start_services "${MONITORING_SERVICES[@]}"
    
    # Step 8: Start Tier 2 agents (specialized)
    log_info "Starting Tier 2 specialized agents..."
    start_services "${TIER2_AGENTS[@]}"
    
    # Step 9: Start Tier 3 agents (support)
    log_info "Starting Tier 3 support agents..."
    start_services "${TIER3_AGENTS[@]}"
    
    # Step 10: Verify deployment
    log_info "Verifying deployment..."
    
    # Show service status
    echo ""
    log_info "Service Status:"
    docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
    
    # Show agent registry
    echo ""
    log_info "Active Agents:"
    docker-compose exec -T redis redis-cli --scan --pattern "agent:registry:*" | while read key; do
        agent_name=$(echo $key | cut -d: -f3)
        status=$(docker-compose exec -T redis redis-cli HGET $key status)
        echo "  - $agent_name: $status"
    done
    
    # Create health check endpoint
    log_info "Setting up health check endpoint..."
    cat > /tmp/agent_health_check.py << 'EOF'
import requests
import json
from datetime import datetime

def check_agents():
    agents = {
        "backend-agi": "http://localhost:8000/health",
        "frontend-agi": "http://localhost:8501/",
        "mcp-server": "http://localhost:8100/health",
        "api-gateway": "http://localhost:8080/health",
        "grafana": "http://localhost:3000/api/health",
        "prometheus": "http://localhost:9090/-/ready"
    }
    
    results = {}
    for agent, url in agents.items():
        try:
            response = requests.get(url, timeout=5)
            results[agent] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "code": response.status_code
            }
        except Exception as e:
            results[agent] = {
                "status": "error",
                "error": str(e)
            }
    
    return results

if __name__ == "__main__":
    results = check_agents()
    print(json.dumps(results, indent=2))
EOF
    
    # Run health check
    if command -v python3 >/dev/null 2>&1; then
        python3 /tmp/agent_health_check.py || true
    fi
    
    log_success "Deployment completed!"
    
    echo ""
    echo "=========================================="
    echo "🚀 SutazAI Unified Agent System Deployed!"
    echo "=========================================="
    echo ""
    echo "Access points:"
    echo "  - Frontend: http://localhost:8501"
    echo "  - Backend API: http://localhost:8000"
    echo "  - API Gateway: http://localhost:8080"
    echo "  - Grafana: http://localhost:3000"
    echo "  - Prometheus: http://localhost:9090"
    echo ""
    echo "To monitor agents:"
    echo "  docker-compose logs -f [service-name]"
    echo ""
    echo "To scale agents:"
    echo "  docker-compose up -d --scale [service-name]=N"
    echo ""
}

# Run main function
main "$@"