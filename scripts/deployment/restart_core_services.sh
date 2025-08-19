#!/bin/bash
#
# Core Services Restart Script
# Purpose: Restart core services to get proper container names
# Created: 2025-08-18 by devops-engineer
#
# This script restarts core services that need proper container names

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Core Services Restart Script ==="
echo "Project Root: $PROJECT_ROOT"
echo "Started: $(date)"
echo

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to safely restart a service
restart_service() {
    local service=$1
    local compose_file="$PROJECT_ROOT/docker/docker-compose.consolidated.yml"
    
    log "Restarting service: $service"
    cd "$PROJECT_ROOT/docker"
    
    # Stop the service
    if docker-compose -f docker-compose.consolidated.yml stop "$service" 2>/dev/null; then
        log "✅ Stopped $service"
    else
        log "⚠️  Service $service was not running"
    fi
    
    # Remove the service container
    if docker-compose -f docker-compose.consolidated.yml rm -f "$service" 2>/dev/null; then
        log "✅ Removed $service container"
    else
        log "⚠️  No container to remove for $service"
    fi
    
    # Start the service
    if docker-compose -f docker-compose.consolidated.yml up -d "$service"; then
        log "✅ Started $service with proper container name"
        
        # Wait for service to be healthy
        sleep 5
        container_name="sutazai-$service"
        if docker ps --filter "name=$container_name" --format "{{.Names}}" | grep -q "$container_name"; then
            log "✅ Container $container_name is running"
        else
            log "⚠️  Container $container_name not found"
        fi
    else
        log "❌ Failed to start $service"
        return 1
    fi
}

# Function to clean up MCP containers with random names
cleanup_mcp_containers() {
    log "=== Cleaning up MCP containers with random names ==="
    
    # Get containers with random names from MCP images
    local mcp_containers=$(docker ps --filter "ancestor=mcp/duckduckgo" --filter "ancestor=mcp/fetch" --filter "ancestor=mcp/sequentialthinking" --format "{{.ID}} {{.Names}}" | grep -E "^[a-f0-9]+ [a-z]+_[a-z]+")
    
    if [ -n "$mcp_containers" ]; then
        log "Found MCP containers with random names:"
        echo "$mcp_containers"
        
        # Stop and remove them
        echo "$mcp_containers" | while read -r container_id container_name; do
            log "Stopping MCP container: $container_name ($container_id)"
            docker stop "$container_id" 2>/dev/null || true
            docker rm "$container_id" 2>/dev/null || true
        done
    else
        log "No MCP containers with random names found"
    fi
}

# Main execution
main() {
    log "=== Phase 1: Core Service Identification ==="
    
    # Core services that need to be restarted for proper naming
    core_services=("backend" "frontend")
    
    # Check which services are currently running with wrong names
    log "Checking current container status..."
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" | grep -E "(backend|frontend|postgres|redis)" || true
    
    log ""
    log "=== Phase 2: MCP Container Cleanup ==="
    cleanup_mcp_containers
    
    log ""
    log "=== Phase 3: Core Service Restart ==="
    
    # Only restart services that actually need fixing
    for service in "${core_services[@]}"; do
        # Check if service exists in docker-compose
        if grep -q "^[[:space:]]*${service}:" "$PROJECT_ROOT/docker/docker-compose.consolidated.yml"; then
            restart_service "$service"
        else
            log "⚠️  Service $service not found in docker-compose.consolidated.yml"
        fi
    done
    
    log ""
    log "=== Phase 4: DNS Resolution Validation ==="
    
    # Wait for services to stabilize
    sleep 10
    
    # Test DNS resolution from frontend to backend and databases
    if docker exec sutazai-frontend ping -c 1 sutazai-postgres >/dev/null 2>&1; then
        log "✅ DNS resolution working: frontend -> sutazai-postgres"
    else
        log "❌ DNS resolution failed: frontend -> sutazai-postgres"
    fi
    
    if docker exec sutazai-frontend ping -c 1 sutazai-redis >/dev/null 2>&1; then
        log "✅ DNS resolution working: frontend -> sutazai-redis"
    else
        log "❌ DNS resolution failed: frontend -> sutazai-redis"
    fi
    
    log ""
    log "=== Phase 5: Final Status Check ==="
    
    echo "=== FINAL CONTAINER STATUS (Sutazai services) ==="
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" | grep "sutazai-" | head -15
    
    echo ""
    echo "=== REMAINING RANDOM NAME CONTAINERS ==="
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" | grep -E "^[a-z]+_[a-z]+" | head -10 || echo "No containers with random names found"
    
    log ""
    log "=== Completion ==="
    log "Core services restarted with proper container names."
}

# Execute main function
main "$@"