#!/bin/bash

# Development Environment Deployment Script
# Purpose: Deploy SutazAI development stack with reduced resources and debug features

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/deploy-dev-$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${1}" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "${RED}ERROR: $1${NC}"
    exit 1
}

# Create logs directory
mkdir -p "${PROJECT_ROOT}/logs"

log "${BLUE}SutazAI Development Environment Deployment${NC}"
log "${BLUE}===========================================${NC}"
log "Started at: $(date)"
log "Project root: $PROJECT_ROOT"
log "Log file: $LOG_FILE"

# Change to project root
cd "$PROJECT_ROOT"

# Validate docker-compose files
log "${YELLOW}Validating Docker Compose configurations...${NC}"
if ! docker-compose -f docker/docker-compose.yml config >/dev/null 2>&1; then
    error_exit "Main docker-compose.yml is invalid"
fi

if ! docker-compose -f docker/docker-compose.dev.yml config >/dev/null 2>&1; then
    error_exit "Development docker-compose.dev.yml is invalid"
fi

log "${GREEN}✓ Docker Compose configurations are valid${NC}"

# Create network if it doesn't exist
log "${YELLOW}Creating Docker network...${NC}"
if ! docker network inspect sutazai-network >/dev/null 2>&1; then
    docker network create sutazai-network
    log "${GREEN}✓ Created sutazai-network${NC}"
else
    log "${GREEN}✓ Network sutazai-network already exists${NC}"
fi

# Pull latest images
log "${YELLOW}Pulling latest Docker images...${NC}"
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml pull || {
    log "${YELLOW}Warning: Some images could not be pulled. Continuing with local images.${NC}"
}

# Build custom images
log "${YELLOW}Building custom Docker images...${NC}"
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml build

# Stop existing containers
log "${YELLOW}Stopping existing containers...${NC}"
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml down || true

# Start development environment
log "${YELLOW}Starting development environment...${NC}"
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

# Wait for services to be healthy
log "${YELLOW}Waiting for services to be healthy...${NC}"
sleep 30

# Validate service health
log "${YELLOW}Checking service health...${NC}"
FAILED_SERVICES=()

# Core services health check
SERVICES=("sutazai-postgres" "sutazai-redis" "sutazai-backend" "sutazai-frontend")

for service in "${SERVICES[@]}"; do
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep "$service" | grep -q "healthy\|Up"; then
        log "${GREEN}✓ $service is running${NC}"
    else
        log "${RED}✗ $service is not healthy${NC}"
        FAILED_SERVICES+=("$service")
    fi
done

# Show service URLs
log "${BLUE}Development Environment URLs:${NC}"
log "Backend API: http://localhost:10010"
log "Frontend UI: http://localhost:10011"
log "PostgreSQL: localhost:10000"
log "Redis: localhost:10001"
log "Prometheus: http://localhost:10200"
log "Grafana: http://localhost:10201 (admin/dev)"

# Show running containers
log "${YELLOW}Running containers:${NC}"
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml ps

# Final status
if [ ${#FAILED_SERVICES[@]} -eq 0 ]; then
    log "${GREEN}Development environment deployed successfully!${NC}"
    log "${GREEN}All core services are running and healthy.${NC}"
    
    # Show logs for debugging
    log "${YELLOW}Showing recent logs for debugging:${NC}"
    docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml logs --tail=10
    
    exit 0
else
    log "${RED}Deployment completed with issues!${NC}"
    log "${RED}Failed services: ${FAILED_SERVICES[*]}${NC}"
    
    # Show logs for failed services
    for service in "${FAILED_SERVICES[@]}"; do
        log "${YELLOW}Logs for $service:${NC}"
        docker logs "$service" --tail=20 2>&1 || true
    done
    
    exit 1
fi