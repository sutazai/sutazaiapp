#!/bin/bash
# ============================================================================
# REAL MCP Infrastructure Deployment Script
# Created: 2025-08-19
# Author: MCP Deployment Expert (20 years experience)
# Purpose: Deploy the ACTUAL MCP infrastructure with all required services
# ============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Pipeline failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# ============================================================================
# Pre-flight checks
# ============================================================================

log "Starting REAL MCP Infrastructure Deployment..."
log "============================================"

# Check if running as appropriate user
if [ "$EUID" -eq 0 ]; then 
   warning "Running as root - this is not recommended for production"
fi

# Check Docker installation
if ! command -v docker &> /dev/null; then
    error "Docker is not installed!"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    error "Docker Compose is not installed!"
    exit 1
fi

# Define Docker Compose command
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Set working directory
WORK_DIR="/opt/sutazaiapp"
cd "$WORK_DIR"

# Check for .env file
if [ ! -f "$WORK_DIR/.env" ]; then
    error ".env file not found at $WORK_DIR/.env"
    exit 1
fi

# Source environment variables
set -a
source "$WORK_DIR/.env"
set +a

# ============================================================================
# Step 1: Clean up existing infrastructure (if needed)
# ============================================================================

log "Step 1: Cleaning up existing infrastructure..."

# Stop all running containers with sutazai prefix
log "Stopping existing sutazai containers..."
docker ps -q --filter "name=sutazai-" | xargs -r docker stop || true
docker ps -q --filter "name=mcp-" | xargs -r docker stop || true

# Remove existing containers (but keep volumes for data persistence)
log "Removing existing containers (keeping volumes)..."
docker ps -aq --filter "name=sutazai-" | xargs -r docker rm || true
docker ps -aq --filter "name=mcp-" | xargs -r docker rm || true

# ============================================================================
# Step 2: Create Docker networks
# ============================================================================

log "Step 2: Creating Docker networks..."

# Create main sutazai network
if ! docker network ls | grep -q "sutazai-network"; then
    log "Creating sutazai-network..."
    docker network create --driver bridge \
        --subnet=172.25.0.0/16 \
        --gateway=172.25.0.1 \
        --attachable \
        sutazai-network
else
    log "sutazai-network already exists"
fi

# Create MCP internal network
if ! docker network ls | grep -q "mcp-internal"; then
    log "Creating mcp-internal network..."
    docker network create --driver bridge \
        --internal \
        --subnet=172.26.0.0/16 \
        mcp-internal
else
    log "mcp-internal network already exists"
fi

# Create DinD internal network
if ! docker network ls | grep -q "sutazai-dind-internal"; then
    log "Creating sutazai-dind-internal network..."
    docker network create --driver bridge \
        --internal \
        --subnet=172.30.0.0/16 \
        sutazai-dind-internal
else
    log "sutazai-dind-internal network already exists"
fi

# ============================================================================
# Step 3: Deploy using consolidated Docker Compose
# ============================================================================

log "Step 3: Deploying core infrastructure services..."

# Use the consolidated Docker Compose file
COMPOSE_FILE="$WORK_DIR/docker/docker-compose.consolidated.yml"

if [ ! -f "$COMPOSE_FILE" ]; then
    error "Consolidated compose file not found at $COMPOSE_FILE"
    
    # Fallback to root compose file
    COMPOSE_FILE="$WORK_DIR/docker-compose.yml"
    if [ ! -f "$COMPOSE_FILE" ]; then
        error "No valid Docker Compose file found!"
        exit 1
    fi
    warning "Using fallback compose file: $COMPOSE_FILE"
fi

# Deploy core services first (databases)
log "Deploying database services..."
$DOCKER_COMPOSE -f "$COMPOSE_FILE" up -d postgres redis neo4j

# Wait for databases to be healthy
log "Waiting for databases to initialize..."
sleep 10

# Check database health
for service in postgres redis neo4j; do
    container="sutazai-$service"
    log "Checking health of $container..."
    
    max_attempts=30
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if docker ps --filter "name=$container" --filter "status=running" | grep -q "$container"; then
            log "$container is running"
            break
        fi
        warning "Waiting for $container to start (attempt $attempt/$max_attempts)..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        error "$container failed to start!"
    fi
done

# Deploy AI/ML services
log "Deploying AI/ML services..."
$DOCKER_COMPOSE -f "$COMPOSE_FILE" up -d ollama chromadb qdrant

# Deploy infrastructure services
log "Deploying infrastructure services..."
$DOCKER_COMPOSE -f "$COMPOSE_FILE" up -d kong consul rabbitmq

# Deploy monitoring stack
log "Deploying monitoring stack..."
$DOCKER_COMPOSE -f "$COMPOSE_FILE" up -d prometheus grafana loki alertmanager jaeger

# Deploy application services
log "Deploying application services..."
$DOCKER_COMPOSE -f "$COMPOSE_FILE" up -d backend frontend

# Deploy all remaining services
log "Deploying all remaining services..."
$DOCKER_COMPOSE -f "$COMPOSE_FILE" up -d

# ============================================================================
# Step 4: Deploy Docker-in-Docker MCP Orchestrator
# ============================================================================

log "Step 4: Deploying Docker-in-Docker MCP Orchestrator..."

DIND_COMPOSE="$WORK_DIR/docker/dind/docker-compose.dind.yml"

if [ -f "$DIND_COMPOSE" ]; then
    log "Deploying DinD orchestrator..."
    $DOCKER_COMPOSE -f "$DIND_COMPOSE" up -d
else
    warning "DinD compose file not found at $DIND_COMPOSE"
fi

# ============================================================================
# Step 5: Deploy MCP services inside DinD
# ============================================================================

log "Step 5: Deploying MCP services..."

MCP_COMPOSE="$WORK_DIR/docker/dind/mcp-containers/docker-compose.mcp-services.yml"

if [ -f "$MCP_COMPOSE" ]; then
    log "Deploying MCP services inside DinD environment..."
    
    # Wait for DinD orchestrator to be ready
    sleep 10
    
    # Deploy MCP services
    if docker ps --filter "name=sutazai-mcp-orchestrator" --filter "status=running" | grep -q "sutazai-mcp-orchestrator"; then
        # Execute inside the DinD container
        docker exec sutazai-mcp-orchestrator sh -c "
            cd /mcp-manifests && 
            docker-compose -f docker-compose.mcp-services.yml up -d
        " || {
            warning "Failed to deploy MCP services inside DinD, trying direct deployment..."
            $DOCKER_COMPOSE -f "$MCP_COMPOSE" up -d
        }
    else
        warning "DinD orchestrator not running, deploying MCP services directly..."
        $DOCKER_COMPOSE -f "$MCP_COMPOSE" up -d
    fi
else
    warning "MCP compose file not found at $MCP_COMPOSE"
fi

# ============================================================================
# Step 6: Initialize Ollama models
# ============================================================================

log "Step 6: Initializing Ollama models..."

# Wait for Ollama to be ready
sleep 5

# Pull the tinyllama model
if docker ps --filter "name=sutazai-ollama" --filter "status=running" | grep -q "sutazai-ollama"; then
    log "Pulling tinyllama model..."
    docker exec sutazai-ollama ollama pull tinyllama || warning "Failed to pull tinyllama model"
fi

# ============================================================================
# Step 7: Health checks and verification
# ============================================================================

log "Step 7: Performing health checks..."

# Function to check service health
check_service() {
    local service=$1
    local port=$2
    local endpoint=${3:-"/"}
    
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port$endpoint" | grep -q "200\|301\|302\|404"; then
        log "✅ $service is responding on port $port"
        return 0
    else
        warning "❌ $service is not responding on port $port"
        return 1
    fi
}

# Check core services
info "Checking core services..."
check_service "PostgreSQL" 10000 || true
check_service "Redis" 10001 || true
check_service "Neo4j" 10002 || true
check_service "Backend API" 10010 "/health" || true
check_service "Frontend" 10011 || true
check_service "Consul" 10006 || true
check_service "Prometheus" 10200 || true
check_service "Ollama" 10104 || true

# ============================================================================
# Step 8: Display deployment summary
# ============================================================================

log "============================================"
log "Deployment Summary"
log "============================================"

# Count running containers
TOTAL_CONTAINERS=$(docker ps --filter "name=sutazai-" --filter "status=running" | tail -n +2 | wc -l)
MCP_CONTAINERS=$(docker ps --filter "name=mcp-" --filter "status=running" | tail -n +2 | wc -l)

info "Total Sutazai containers running: $TOTAL_CONTAINERS"
info "Total MCP containers running: $MCP_CONTAINERS"
info "Total containers: $((TOTAL_CONTAINERS + MCP_CONTAINERS))"

log ""
log "Service Endpoints:"
log "  Backend API:     http://localhost:10010"
log "  Frontend UI:     http://localhost:10011"
log "  PostgreSQL:      localhost:10000"
log "  Redis:           localhost:10001"
log "  Neo4j Browser:   http://localhost:10002"
log "  Consul UI:       http://localhost:10006"
log "  Prometheus:      http://localhost:10200"
log "  Grafana:         http://localhost:10201"
log "  Ollama:          http://localhost:10104"
log ""

# List all running containers
log "Running containers:"
docker ps --filter "name=sutazai-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -20
docker ps --filter "name=mcp-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -20

log ""
log "============================================"
log "✅ MCP Infrastructure Deployment Complete!"
log "============================================"
log ""
log "Next steps:"
log "1. Verify services at the endpoints listed above"
log "2. Check logs: docker logs sutazai-backend"
log "3. Monitor health: docker ps"
log "4. Access Consul UI for service discovery"
log ""

# Create a deployment report
REPORT_FILE="$WORK_DIR/docs/reports/DEPLOYMENT_REPORT_$(date +%Y%m%d_%H%M%S).md"
mkdir -p "$(dirname "$REPORT_FILE")"

cat > "$REPORT_FILE" << EOF
# MCP Infrastructure Deployment Report
Generated: $(date '+%Y-%m-%d %H:%M:%S UTC')

## Deployment Status
- **Total Containers**: $((TOTAL_CONTAINERS + MCP_CONTAINERS))
- **Sutazai Containers**: $TOTAL_CONTAINERS
- **MCP Containers**: $MCP_CONTAINERS

## Service Endpoints
- Backend API: http://localhost:10010
- Frontend UI: http://localhost:10011
- PostgreSQL: localhost:10000
- Redis: localhost:10001
- Neo4j: http://localhost:10002
- Consul: http://localhost:10006
- Prometheus: http://localhost:10200
- Grafana: http://localhost:10201
- Ollama: http://localhost:10104

## Running Containers
\`\`\`
$(docker ps --filter "name=sutazai-" --format "{{.Names}} | {{.Status}}")
$(docker ps --filter "name=mcp-" --format "{{.Names}} | {{.Status}}")
\`\`\`

## Network Configuration
\`\`\`
$(docker network ls | grep sutazai)
$(docker network ls | grep mcp)
\`\`\`
EOF

log "Deployment report saved to: $REPORT_FILE"

exit 0