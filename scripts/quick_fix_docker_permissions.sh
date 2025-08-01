#!/bin/bash

# Quick Fix for Docker Socket Permissions
# Immediate resolution for containers restarting due to Docker socket permission issues

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[WARNING] $1${NC}"; }
error() { echo -e "${RED}[ERROR] $1${NC}"; }
success() { echo -e "${GREEN}[SUCCESS] $1${NC}"; }

# Get Docker group ID from host
DOCKER_GID=$(stat -c %g /var/run/docker.sock)
log "Host Docker group ID: $DOCKER_GID"

# Container names
CONTAINERS=(
    "sutazai-hardware-optimizer"
    "sutazai-devops-manager" 
    "sutazai-ollama-specialist"
)

# Stop problematic containers
log "Stopping problematic containers..."
for container in "${CONTAINERS[@]}"; do
    if docker ps -q -f name="$container" | grep -q .; then
        log "Stopping $container..."
        docker stop "$container" || warn "Failed to stop $container"
    fi
done

# Create quick fix compose override
log "Creating quick fix configuration..."
cat > /opt/sutazaiapp/docker-compose.quick-fix.yml << EOF
version: '3.8'

services:
  infrastructure-devops-manager:
    user: "1000:$DOCKER_GID"
    environment:
      - DOCKER_HOST=unix:///var/run/docker.sock
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:rw
      - ./scripts:/scripts:ro
      - ./logs/devops:/logs

  hardware-resource-optimizer:
    user: "1000:$DOCKER_GID" 
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:rw
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - ./logs/optimizer:/logs

  ollama-integration-specialist:
    user: "1000:$DOCKER_GID"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:rw
      - ./models:/models
      - ./config/ollama:/config
EOF

# Create log directories
log "Creating log directories..."
mkdir -p /opt/sutazaiapp/logs/{devops,optimizer}

# Start containers with quick fix
log "Starting containers with corrected permissions..."
cd /opt/sutazaiapp
docker-compose -f docker-compose-agents-tier1.yml -f docker-compose.quick-fix.yml up -d "${CONTAINERS[@]}" || {
    error "Failed to start containers with quick fix"
    exit 1
}

# Wait and verify
log "Waiting for containers to start..."
sleep 15

# Check container status
log "Verifying container status..."
for container in "${CONTAINERS[@]}"; do
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep "$container" | grep -q "Up"; then
        success "$container is running"
        
        # Test Docker access
        if docker exec "$container" docker ps >/dev/null 2>&1; then
            success "$container has Docker socket access"
        else
            warn "$container: Docker socket access test failed"
        fi
    else
        warn "$container is not running properly"
        docker logs --tail 10 "$container" 2>/dev/null || true
    fi
done

success "Quick fix applied! Containers should now have proper Docker socket access."
log "For permanent fix with enhanced security, run: ./scripts/fix_docker_socket_permissions.sh"