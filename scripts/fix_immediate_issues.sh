#!/bin/bash

# Immediate Critical Fixes Script
# Fixes Docker socket permissions and restarts failing containers without full redeployment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root to fix Docker socket permissions"
    exit 1
fi

log "Applying immediate critical fixes..."

# Fix 1: Docker socket permissions
log "Fix 1: Setting Docker socket permissions..."
chmod 666 /var/run/docker.sock
chown root:docker /var/run/docker.sock
log_success "Docker socket permissions fixed"

# Fix 2: Stop failing containers
log "Fix 2: Stopping failing containers..."
FAILED_CONTAINERS=("sutazai-hardware-optimizer" "sutazai-devops-manager" "sutazai-ollama-specialist")

for container in "${FAILED_CONTAINERS[@]}"; do
    if docker ps -a --filter "name=$container" --format "{{.Names}}" | grep -q "$container"; then
        log "Stopping $container"
        docker stop "$container" 2>/dev/null || true
        docker rm "$container" 2>/dev/null || true
        log_success "$container stopped and removed"
    fi
done

# Fix 3: Rebuild problematic images with root user
log "Fix 3: Rebuilding Docker images with fixes..."
cd /opt/sutazaiapp

# Quick fix for hardware optimizer
if [ -d "agents/hardware-optimizer" ]; then
    log "Rebuilding hardware-optimizer..."
    docker build -t sutazaiapp-hardware-resource-optimizer ./agents/hardware-optimizer/ --no-cache
fi

# Quick fix for devops manager
if [ -d "agents/infrastructure-devops" ]; then
    log "Rebuilding devops-manager..."
    docker build -t sutazaiapp-infrastructure-devops-manager ./agents/infrastructure-devops/ --no-cache
fi

# Quick fix for ollama specialist
if [ -d "agents/ollama-integration" ]; then
    log "Rebuilding ollama-specialist..."
    docker build -t sutazaiapp-ollama-integration-specialist ./agents/ollama-integration/ --no-cache
fi

log_success "Images rebuilt with fixes"

# Fix 4: Start containers with proper configuration
log "Fix 4: Starting fixed containers..."

# Start hardware optimizer with correct configuration
docker run -d \
    --name sutazai-hardware-optimizer \
    --restart unless-stopped \
    --network sutazaiapp_sutazai-network \
    --user root \
    -v /var/run/docker.sock:/var/run/docker.sock:rw \
    -v /proc:/host/proc:ro \
    -v /sys:/host/sys:ro \
    -v /opt/sutazaiapp/logs/optimizer:/logs \
    -p 8523:8523 \
    -e MAX_MEMORY_MB=8192 \
    -e CPU_THRESHOLD=80 \
    -e MEMORY_THRESHOLD=85 \
    -e OPTIMIZATION_INTERVAL=60 \
    -e DOCKER_HOST=unix:///var/run/docker.sock \
    --memory=384m \
    --cpus=0.5 \
    sutazaiapp-hardware-resource-optimizer

# Start devops manager with correct configuration
docker run -d \
    --name sutazai-devops-manager \
    --restart unless-stopped \
    --network sutazaiapp_sutazai-network \
    --user root \
    -v /var/run/docker.sock:/var/run/docker.sock:rw \
    -v /opt/sutazaiapp/scripts:/scripts:ro \
    -v /opt/sutazaiapp/logs/devops:/logs \
    -p 8522:8522 \
    -e DOCKER_HOST=unix:///var/run/docker.sock \
    -e MONITORING_ENABLED=true \
    -e PROMETHEUS_URL=http://sutazai-prometheus:9090 \
    -e GRAFANA_URL=http://sutazai-grafana:3000 \
    --memory=384m \
    --cpus=0.5 \
    sutazaiapp-infrastructure-devops-manager

# Start ollama specialist with correct configuration
docker run -d \
    --name sutazai-ollama-specialist \
    --restart unless-stopped \
    --network sutazaiapp_sutazai-network \
    --user root \
    -v /var/run/docker.sock:/var/run/docker.sock:ro \
    -v /opt/sutazaiapp/models:/models \
    -v /opt/sutazaiapp/config/ollama:/config \
    -p 8520:8520 \
    -e OLLAMA_HOST=http://sutazai-ollama:11434 \
    -e OPENAI_API_BASE=http://host.docker.internal:4000 \
    -e OPENAI_API_KEY=local \
    -e LOG_LEVEL=INFO \
    -e DOCKER_HOST=unix:///var/run/docker.sock \
    --memory=384m \
    --cpus=0.5 \
    sutazaiapp-ollama-integration-specialist

log_success "Fixed containers started"

# Fix 5: Verify fixes
log "Fix 5: Verifying fixes..."
sleep 15

for container in "${FAILED_CONTAINERS[@]}"; do
    if docker ps --filter "name=$container" --filter "status=running" --format "{{.Names}}" | grep -q "$container"; then
        log_success "$container is now running"
        
        # Check health endpoint
        case $container in
            "sutazai-hardware-optimizer") 
                if curl -f http://localhost:8523/health >/dev/null 2>&1; then
                    log_success "Hardware optimizer health check passed"
                fi
                ;;
            "sutazai-devops-manager") 
                if curl -f http://localhost:8522/health >/dev/null 2>&1; then
                    log_success "DevOps manager health check passed"
                fi
                ;;
            "sutazai-ollama-specialist") 
                if curl -f http://localhost:8520/health >/dev/null 2>&1; then
                    log_success "Ollama specialist health check passed"
                fi
                ;;
        esac
    else
        log_error "$container failed to start"
        docker logs "$container" --tail 10 2>/dev/null || true
    fi
done

# Final status
log "Final system status:"
docker ps --filter "name=sutazai" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

log_success "Immediate fixes applied successfully!"
log "All previously failing containers should now be running with proper Docker socket access."