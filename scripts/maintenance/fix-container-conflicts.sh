#!/bin/bash

# Fix Container Naming Conflicts
# Purpose: Resolve conflicts between standalone containers and Docker Compose managed containers
# Usage: ./fix-container-conflicts.sh [--dry-run] [--force]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
DRY_RUN=false
FORCE=false
BACKUP_PREFIX="sutazai-backup-$(date +%Y%m%d_%H%M%S)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--force]"
            exit 1
            ;;
    esac
done

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

# Check if running as root or with Docker permissions
check_docker_permissions() {
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon not accessible. Please ensure Docker is running and you have permissions."
        exit 1
    fi
}

# Backup container data volumes if they exist
backup_container_data() {
    local container_name="$1"
    
    if docker container inspect "$container_name" >/dev/null 2>&1; then
        log "Backing up data for container: $container_name"
        
        # Get volume mounts
        local volumes
        volumes=$(docker inspect "$container_name" --format '{{range .Mounts}}{{if eq .Type "volume"}}{{.Name}}:{{.Destination}} {{end}}{{end}}')
        
        if [[ -n "$volumes" ]]; then
            log "Found volumes for $container_name: $volumes"
            
            if [[ "$DRY_RUN" == "false" ]]; then
                # Create backup container to copy data
                docker run --rm \
                    --volumes-from "$container_name" \
                    -v "$PWD/backup:/backup" \
                    alpine:latest \
                    sh -c "tar czf /backup/${container_name}-$(date +%Y%m%d_%H%M%S).tar.gz /data /var/lib/postgresql/data /var/lib/redis /var/lib/neo4j/data 2>/dev/null || true"
            fi
        fi
    fi
}

# Stop and remove conflicting containers
remove_conflicting_containers() {
    local containers=(
        "sutazai-postgres"
        "sutazai-redis"
        "sutazai-neo4j"
        "sutazai-chromadb"
        "sutazai-qdrant"
        "sutazai-ollama"
        "sutazai-backend"
        "sutazai-frontend"
        "sutazai-prometheus"
        "sutazai-grafana"
    )
    
    log "Checking for conflicting containers..."
    
    for container in "${containers[@]}"; do
        if docker container inspect "$container" >/dev/null 2>&1; then
            local status
            status=$(docker inspect --format='{{.State.Status}}' "$container")
            
            log "Found container $container with status: $status"
            
            if [[ "$status" == "running" ]]; then
                log "Stopping running container: $container"
                if [[ "$DRY_RUN" == "false" ]]; then
                    docker stop "$container" || true
                fi
            fi
            
            log "Removing container: $container"
            if [[ "$DRY_RUN" == "false" ]]; then
                docker rm "$container" || true
            fi
        else
            log "Container $container not found - OK"
        fi
    done
}

# Clean up orphaned networks
cleanup_networks() {
    log "Cleaning up Docker networks..."
    
    # Remove sutazai-network if it exists and is not in use
    if docker network inspect sutazai-network >/dev/null 2>&1; then
        log "Found existing sutazai-network"
        if [[ "$DRY_RUN" == "false" ]]; then
            # Try to remove, but don't fail if containers are still using it
            docker network rm sutazai-network 2>/dev/null || log "Network in use - will be recreated by compose"
        fi
    fi
    
    # Prune unused networks
    if [[ "$DRY_RUN" == "false" ]]; then
        docker network prune -f
    fi
}

# Verify Docker Compose configuration
verify_compose_config() {
    log "Verifying Docker Compose configuration..."
    
    cd "$PROJECT_ROOT"
    
    if [[ ! -f "docker-compose.yml" ]]; then
        error "docker-compose.yml not found in $PROJECT_ROOT"
        exit 1
    fi
    
    # Validate compose file
    if ! docker-compose config >/dev/null 2>&1; then
        error "Docker Compose configuration is invalid"
        exit 1
    fi
    
    log "Docker Compose configuration is valid"
}

# Deploy core services with Docker Compose
deploy_core_services() {
    log "Deploying core services with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Core services that must be deployed first
    local core_services=(
        "postgres"
        "redis"
        "neo4j"
        "chromadb"
        "qdrant"
        "faiss"
        "ollama"
    )
    
    # Deploy core services one by one to handle dependencies
    for service in "${core_services[@]}"; do
        log "Deploying service: $service"
        if [[ "$DRY_RUN" == "false" ]]; then
            docker-compose up -d "$service"
            
            # Wait for health check to pass if defined
            local max_wait=60
            local wait_time=0
            while [[ $wait_time -lt $max_wait ]]; do
                if docker-compose ps "$service" | grep -q "healthy\|Up"; then
                    log "Service $service is ready"
                    break
                fi
                log "Waiting for $service to be ready... ($wait_time/${max_wait}s)"
                sleep 5
                wait_time=$((wait_time + 5))
            done
        fi
    done
}

# Deploy application services
deploy_app_services() {
    log "Deploying application services..."
    
    cd "$PROJECT_ROOT"
    
    local app_services=(
        "backend"
        "frontend"
    )
    
    for service in "${app_services[@]}"; do
        log "Deploying service: $service"
        if [[ "$DRY_RUN" == "false" ]]; then
            docker-compose up -d "$service"
            
            # Wait for service to be ready
            local max_wait=120
            local wait_time=0
            while [[ $wait_time -lt $max_wait ]]; do
                if docker-compose ps "$service" | grep -q "healthy\|Up"; then
                    log "Service $service is ready"
                    break
                fi
                log "Waiting for $service to be ready... ($wait_time/${max_wait}s)"
                sleep 10
                wait_time=$((wait_time + 10))
            done
        fi
    done
}

# Deploy monitoring services
deploy_monitoring_services() {
    log "Deploying monitoring services..."
    
    cd "$PROJECT_ROOT"
    
    local monitoring_services=(
        "prometheus"
        "grafana"
        "loki"
        "alertmanager"
    )
    
    for service in "${monitoring_services[@]}"; do
        log "Deploying monitoring service: $service"
        if [[ "$DRY_RUN" == "false" ]]; then
            docker-compose up -d "$service" || log "Warning: Failed to deploy $service"
        fi
    done
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    cd "$PROJECT_ROOT"
    
    # Check service status
    log "Service Status:"
    docker-compose ps
    
    # Check health of core services
    local core_services=("postgres" "redis" "neo4j" "chromadb" "qdrant" "ollama" "backend")
    
    for service in "${core_services[@]}"; do
        local container_name="sutazai-$service"
        if docker container inspect "$container_name" >/dev/null 2>&1; then
            local status
            status=$(docker inspect --format='{{.State.Status}}' "$container_name")
            local health
            health=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "no-healthcheck")
            
            log "✓ $service: $status ($health)"
        else
            log "✗ $service: not found"
        fi
    done
}

# Create deployment summary
create_summary() {
    log "Creating deployment summary..."
    
    local summary_file="$PROJECT_ROOT/logs/container-conflict-resolution-$(date +%Y%m%d_%H%M%S).log"
    
    {
        echo "Container Conflict Resolution Summary"
        echo "Timestamp: $(date)"
        echo "====================================="
        echo
        echo "Resolved Conflicts:"
        echo "- Removed standalone containers conflicting with Docker Compose"
        echo "- Cleaned up orphaned networks"
        echo "- Deployed services using Docker Compose"
        echo
        echo "Services Status:"
        docker-compose ps
        echo
        echo "Container Status:"
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep sutazai
    } > "$summary_file"
    
    log "Summary saved to: $summary_file"
}

# Main execution
main() {
    log "Starting container conflict resolution..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN MODE - No changes will be made"
    fi
    
    if [[ "$FORCE" != "true" ]]; then
        echo "This script will stop and remove existing containers to resolve conflicts."
        echo "Data in named volumes will be preserved, but container state will be lost."
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Operation cancelled by user"
            exit 0
        fi
    fi
    
    # Execute steps
    check_docker_permissions
    verify_compose_config
    remove_conflicting_containers
    cleanup_networks
    deploy_core_services
    deploy_app_services
    deploy_monitoring_services
    verify_deployment
    
    if [[ "$DRY_RUN" == "false" ]]; then
        create_summary
    fi
    
    log "Container conflict resolution completed successfully!"
    log "You can now use 'docker-compose up -d' to manage all services"
}

# Execute main function
main "$@"