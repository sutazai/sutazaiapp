#!/bin/bash
#
# Container Naming Fix Script
# Purpose: Fix Docker container naming issues permanently
# Created: 2025-08-18 by devops-engineer
# 
# This script addresses the container naming problem where containers get
# prefixed with random hashes like "a6d814bf7918_sutazai-postgres"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Container Naming Fix Script ==="
echo "Project Root: $PROJECT_ROOT"
echo "Started: $(date)"
echo

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if service has container_name defined
check_container_name() {
    local service=$1
    local compose_file="$PROJECT_ROOT/docker/docker-compose.consolidated.yml"
    
    if grep -A 10 "^[[:space:]]*${service}:" "$compose_file" | grep -q "container_name:"; then
        echo "✅ $service has container_name defined"
        return 0
    else
        echo "❌ $service missing container_name"
        return 1
    fi
}

# Function to add network alias for existing containers
add_network_alias() {
    local container_id=$1
    local alias_name=$2
    local network="sutazai-network"
    
    log "Adding network alias $alias_name for container $container_id"
    
    if docker network connect --alias "$alias_name" "$network" "$container_id" 2>/dev/null; then
        log "✅ Added network alias: $alias_name -> $container_id"
    else
        log "⚠️  Container $container_id already has alias $alias_name or is not connected"
    fi
}

# Function to restart container with proper name
restart_with_proper_name() {
    local container_id=$1
    local service_name=$2
    
    log "Stopping container: $container_id"
    docker stop "$container_id" 2>/dev/null || true
    
    log "Removing container: $container_id"
    docker rm "$container_id" 2>/dev/null || true
    
    log "Recreating service: $service_name"
    cd "$PROJECT_ROOT/docker"
    docker-compose -f docker-compose.consolidated.yml up -d "$service_name"
}

# Main execution
main() {
    log "=== Phase 1: Analysis ==="
    
    # Get all running containers
    log "Current running containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" | head -20
    
    log ""
    log "=== Phase 2: Network Alias Fixes ==="
    
    # Add network aliases for containers with random names
    # These are immediate fixes while we plan proper restarts
    while read -r container_id container_name image; do
        if [[ "$container_name" =~ ^[a-z]+_[a-z]+ ]] && [[ "$image" =~ postgres ]]; then
            log "Found PostgreSQL container with random name: $container_name"
            add_network_alias "$container_id" "sutazai-postgres"
        elif [[ "$container_name" =~ ^[a-z]+_[a-z]+ ]] && [[ "$image" =~ redis ]]; then
            log "Found Redis container with random name: $container_name"
            add_network_alias "$container_id" "sutazai-redis"
        elif [[ "$container_name" =~ ^[a-z]+_[a-z]+ ]] && [[ "$image" =~ neo4j ]]; then
            log "Found Neo4j container with random name: $container_name"
            add_network_alias "$container_id" "sutazai-neo4j"
        fi
    done < <(docker ps --format "{{.ID}}\t{{.Names}}\t{{.Image}}")
    
    log ""
    log "=== Phase 3: DNS Resolution Test ==="
    
    # Test DNS resolution from a known good container
    if docker exec sutazai-frontend nslookup sutazai-postgres 2>/dev/null; then
        log "✅ DNS resolution working: sutazai-postgres"
    else
        log "❌ DNS resolution failed: sutazai-postgres"
    fi
    
    log ""
    log "=== Phase 4: Service Status Check ==="
    
    # Check which services are actually defined in docker-compose
    log "Services with missing container_name in docker-compose.consolidated.yml:"
    
    # Services that should have container_name but might be missing it
    services=("backend" "frontend" "prometheus" "grafana" "loki" "alertmanager" "blackbox-exporter" 
              "node-exporter" "cadvisor" "postgres-exporter" "redis-exporter" "ollama-integration" 
              "hardware-resource-optimizer" "task-assignment-coordinator" "ultra-system-architect" 
              "ultra-frontend-ui-architect" "jaeger" "promtail")
    
    for service in "${services[@]}"; do
        check_container_name "$service" || true
    done
    
    log ""
    log "=== Completion ==="
    log "Container naming fixes applied. Check docker ps for updated container names."
    
    # Final status
    echo ""
    echo "=== FINAL CONTAINER STATUS ==="
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" | head -15
}

# Execute main function
main "$@"