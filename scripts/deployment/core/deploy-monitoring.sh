#!/bin/bash

# Monitoring Stack Deployment Script
# Purpose: Deploy complete observability infrastructure for SutazAI

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/deploy-monitoring-$(date +%Y%m%d_%H%M%S).log"

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

log "${BLUE}SutazAI Monitoring Stack Deployment${NC}"
log "${BLUE}===================================${NC}"
log "Started at: $(date)"
log "Project root: $PROJECT_ROOT"
log "Log file: $LOG_FILE"

# Change to project root
cd "$PROJECT_ROOT"

# Validate docker-compose files
log "${YELLOW}Validating monitoring configuration...${NC}"
if ! docker-compose -f docker/docker-compose.monitoring.yml config >/dev/null 2>&1; then
    error_exit "Monitoring docker-compose.monitoring.yml is invalid"
fi

log "${GREEN}✓ Monitoring configuration is valid${NC}"

# Create network if it doesn't exist
log "${YELLOW}Creating Docker network...${NC}"
if ! docker network inspect sutazai-network >/dev/null 2>&1; then
    docker network create sutazai-network
    log "${GREEN}✓ Created sutazai-network${NC}"
else
    log "${GREEN}✓ Network sutazai-network already exists${NC}"
fi

# Create monitoring directories
log "${YELLOW}Creating monitoring directories...${NC}"
mkdir -p monitoring/{prometheus,grafana,loki,alertmanager,blackbox}
mkdir -p monitoring/grafana/{provisioning/{datasources,dashboards},dashboards}
mkdir -p monitoring/prometheus/{rules,targets}

# Pull latest monitoring images
log "${YELLOW}Pulling latest monitoring images...${NC}"
docker-compose -f docker/docker-compose.monitoring.yml pull || {
    log "${YELLOW}Warning: Some images could not be pulled. Continuing with local images.${NC}"
}

# Build custom monitoring images
log "${YELLOW}Building custom monitoring images...${NC}"
docker-compose -f docker/docker-compose.monitoring.yml build

# Stop existing monitoring containers
log "${YELLOW}Stopping existing monitoring containers...${NC}"
docker-compose -f docker/docker-compose.monitoring.yml down || true

# Start monitoring stack
log "${YELLOW}Starting monitoring stack...${NC}"
docker-compose -f docker/docker-compose.monitoring.yml up -d

# Wait for services to initialize
log "${YELLOW}Waiting for monitoring services to initialize...${NC}"
sleep 45

# Validate monitoring service health
log "${YELLOW}Checking monitoring service health...${NC}"
FAILED_SERVICES=()

# Monitoring services health check
MONITORING_SERVICES=(
    "sutazai-prometheus:10200"
    "sutazai-grafana:10201"
    "sutazai-loki:10202"
    "sutazai-alertmanager:10203"
    "sutazai-blackbox-exporter:10204"
    "sutazai-node-exporter:10205"
    "sutazai-cadvisor:10206"
    "sutazai-jaeger:10210"
)

for service_port in "${MONITORING_SERVICES[@]}"; do
    service=$(echo "$service_port" | cut -d: -f1)
    port=$(echo "$service_port" | cut -d: -f2)
    
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep "$service" | grep -q "healthy\|Up"; then
        # Additional HTTP health check for web services
        if curl -s -f "http://localhost:$port" >/dev/null 2>&1 || curl -s -f "http://localhost:$port/health" >/dev/null 2>&1; then
            log "${GREEN}✓ $service is running and accessible on port $port${NC}"
        else
            log "${YELLOW}△ $service is running but not yet accessible on port $port${NC}"
        fi
    else
        log "${RED}✗ $service is not healthy${NC}"
        FAILED_SERVICES+=("$service")
    fi
done

# Show monitoring URLs
log "${BLUE}Monitoring Stack URLs:${NC}"
log "Prometheus: http://localhost:10200"
log "Grafana: http://localhost:10201 (admin/admin)"
log "Loki: http://localhost:10202"
log "AlertManager: http://localhost:10203"
log "Blackbox Exporter: http://localhost:10204"
log "Node Exporter: http://localhost:10205"
log "cAdvisor: http://localhost:10206"
log "Jaeger UI: http://localhost:10210"

# Configure Grafana dashboards if accessible
log "${YELLOW}Configuring Grafana dashboards...${NC}"
sleep 10
if curl -s -f "http://localhost:10201" >/dev/null 2>&1; then
    log "${GREEN}✓ Grafana is accessible, dashboards should auto-provision${NC}"
else
    log "${YELLOW}△ Grafana not yet accessible, dashboards will provision on startup${NC}"
fi

# Validate Prometheus targets
log "${YELLOW}Checking Prometheus targets...${NC}"
sleep 5
if curl -s "http://localhost:10200/api/v1/targets" | grep -q "\"health\":\"up\""; then
    log "${GREEN}✓ Prometheus has healthy targets${NC}"
else
    log "${YELLOW}△ Some Prometheus targets may still be initializing${NC}"
fi

# Show running monitoring containers
log "${YELLOW}Running monitoring containers:${NC}"
docker-compose -f docker/docker-compose.monitoring.yml ps

# Show resource usage
log "${YELLOW}Monitoring stack resource usage:${NC}"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" $(docker-compose -f docker/docker-compose.monitoring.yml ps -q) 2>/dev/null || true

# Final status
if [ ${#FAILED_SERVICES[@]} -eq 0 ]; then
    log "${GREEN}Monitoring stack deployed successfully!${NC}"
    log "${GREEN}All monitoring services are running and healthy.${NC}"
    
    # Show recent logs
    log "${YELLOW}Showing recent monitoring logs:${NC}"
    docker-compose -f docker/docker-compose.monitoring.yml logs --tail=5
    
    log "${BLUE}Next Steps:${NC}"
    log "1. Visit Grafana at http://localhost:10201 (admin/admin)"
    log "2. Import dashboards from monitoring/grafana/dashboards/"
    log "3. Configure alerting in AlertManager"
    log "4. Set up log shipping with Promtail"
    
    exit 0
else
    log "${RED}Monitoring deployment completed with issues!${NC}"
    log "${RED}Failed services: ${FAILED_SERVICES[*]}${NC}"
    
    # Show logs for failed services
    for service in "${FAILED_SERVICES[@]}"; do
        log "${YELLOW}Logs for $service:${NC}"
        docker logs "$service" --tail=20 2>&1 || true
    done
    
    log "${YELLOW}Troubleshooting tips:${NC}"
    log "1. Check Docker resources (memory/CPU)"
    log "2. Verify configuration files exist"
    log "3. Check network connectivity"
    log "4. Review service logs above"
    
    exit 1
fi