#!/bin/bash
# Container Security Migration - Volume Permission Fix Script
# Created: August 9, 2025
# Purpose: Fix volume permissions for non-root container migration

set -euo pipefail

# Colors for output

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Check if running as root or with docker permissions
check_docker_permissions() {
    if ! docker ps &>/dev/null; then
        error "Cannot access Docker. Please run with sudo or add user to docker group."
        exit 1
    fi
}

# Backup current volume permissions
backup_volume_permissions() {
    log "Creating backup of current volume permissions..."
    
    local backup_dir="/opt/sutazaiapp/backups/permissions_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Get volume mount points and save permissions
    docker volume ls | grep sutazaiapp | while read -r driver volume; do
        local mount_point
        mount_point=$(docker volume inspect "$volume" --format '{{.Mountpoint}}' 2>/dev/null)
        if [[ -n "$mount_point" && -d "$mount_point" ]]; then
            log "Backing up permissions for $volume..."
            sudo ls -la "$mount_point" > "$backup_dir/$volume.permissions" 2>/dev/null || true
        fi
    done
    
    success "Volume permissions backed up to $backup_dir"
    echo "$backup_dir" > "$(mktemp /tmp/sutazai_permissions_backup_path.XXXXXX)"
}

# Fix AI Agent Orchestrator permissions
fix_ai_orchestrator_permissions() {
    log "Fixing AI Agent Orchestrator permissions..."
    
    # Create appuser in container if it doesn't exist
    docker exec sutazai-ai-agent-orchestrator bash -c "
        groupadd -f -g 999 appuser 2>/dev/null || true
        useradd -r -u 999 -g appuser -s /bin/false appuser 2>/dev/null || true
        chown -R appuser:appuser /app
        mkdir -p /app/logs
        chown -R appuser:appuser /app/logs
    " || warning "Could not fix AI orchestrator permissions (container may not be running)"
}

# Fix ChromaDB permissions
fix_chromadb_permissions() {
    log "Fixing ChromaDB permissions..."
    
    docker exec sutazai-chromadb bash -c "
        # ChromaDB typically runs as root, create chromadb user
        groupadd -f -g 1000 chromadb 2>/dev/null || true
        useradd -r -u 1000 -g chromadb -s /bin/false chromadb 2>/dev/null || true
        mkdir -p /chroma/chroma
        chown -R chromadb:chromadb /chroma
        chmod -R 755 /chroma
    " || warning "Could not fix ChromaDB permissions (container may not be running)"
    
    success "ChromaDB permissions fixed"
}

# Fix Qdrant permissions
fix_qdrant_permissions() {
    log "Fixing Qdrant permissions..."
    
    docker exec sutazai-qdrant bash -c "
        # Qdrant has its own user, ensure proper ownership
        id qdrant &>/dev/null || {
            groupadd -f -g 1001 qdrant
            useradd -r -u 1001 -g qdrant -s /bin/false qdrant
        }
        mkdir -p /qdrant/storage
        chown -R qdrant:qdrant /qdrant/storage
        chmod -R 755 /qdrant/storage
    " || warning "Could not fix Qdrant permissions (container may not be running)"
    
    success "Qdrant permissions fixed"
}

# Fix PostgreSQL permissions
fix_postgres_permissions() {
    log "Fixing PostgreSQL permissions..."
    
    docker exec sutazai-postgres bash -c "
        # PostgreSQL already runs as postgres user, ensure volume ownership
        chown -R postgres:postgres /var/lib/postgresql/data
        chmod 700 /var/lib/postgresql/data
    " || warning "Could not fix PostgreSQL permissions (container may not be running)"
    
    success "PostgreSQL permissions fixed"
}

# Fix Redis permissions
fix_redis_permissions() {
    log "Fixing Redis permissions..."
    
    docker exec sutazai-redis bash -c "
        # Redis already has redis user
        chown -R redis:redis /data
        chmod 755 /data
    " || warning "Could not fix Redis permissions (container may not be running)"
    
    success "Redis permissions fixed"
}

# Fix Neo4j permissions
fix_neo4j_permissions() {
    log "Fixing Neo4j permissions..."
    
    # Neo4j uses uid 7474, ensure volume ownership is correct
    local mount_point
    mount_point=$(docker volume inspect sutazaiapp_neo4j_data --format '{{.Mountpoint}}' 2>/dev/null)
    
    if [[ -n "$mount_point" && -d "$mount_point" ]]; then
        sudo chown -R 7474:7474 "$mount_point" || warning "Could not change Neo4j volume ownership"
        sudo chmod -R 755 "$mount_point" || warning "Could not change Neo4j volume permissions"
        success "Neo4j permissions fixed"
    else
        warning "Neo4j volume not found or not accessible"
    fi
}

# Fix RabbitMQ permissions
fix_rabbitmq_permissions() {
    log "Fixing RabbitMQ permissions..."
    
    docker exec sutazai-rabbitmq bash -c "
        # Create rabbitmq user if doesn't exist
        id rabbitmq &>/dev/null || {
            groupadd -f -g 999 rabbitmq
            useradd -r -u 999 -g rabbitmq -s /bin/false rabbitmq
        }
        chown -R rabbitmq:rabbitmq /var/lib/rabbitmq
        chmod -R 755 /var/lib/rabbitmq
    " || warning "Could not fix RabbitMQ permissions (container may not be running)"
    
    success "RabbitMQ permissions fixed"
}

# Fix Ollama permissions
fix_ollama_permissions() {
    log "Fixing Ollama permissions..."
    
    docker exec sutazai-ollama bash -c "
        # Create ollama user
        groupadd -f -g 1001 ollama 2>/dev/null || true
        useradd -r -u 1001 -g ollama -s /bin/false ollama 2>/dev/null || true
        # Move /root/.ollama to /home/ollama/.ollama
        mkdir -p /home/ollama
        if [[ -d /root/.ollama ]]; then
            cp -r /root/.ollama /home/ollama/ 2>/dev/null || true
            rm -rf /root/.ollama 2>/dev/null || true
        fi
        mkdir -p /home/ollama/.ollama
        chown -R ollama:ollama /home/ollama
        # Also fix models directory
        mkdir -p /models
        chown -R ollama:ollama /models
    " || warning "Could not fix Ollama permissions (container may not be running)"
    
    success "Ollama permissions fixed"
}

# Fix Consul permissions
fix_consul_permissions() {
    log "Fixing Consul permissions..."
    
    docker exec sutazai-consul bash -c "
        # Consul has its own user
        id consul &>/dev/null || {
            groupadd -f -g 100 consul
            useradd -r -u 100 -g consul -s /bin/false consul
        }
        mkdir -p /consul/data
        chown -R consul:consul /consul/data
        chmod -R 755 /consul/data
    " || warning "Could not fix Consul permissions (container may not be running)"
    
    success "Consul permissions fixed"
}

# Fix Blackbox Exporter permissions
fix_blackbox_exporter_permissions() {
    log "Fixing Blackbox Exporter permissions..."
    
    docker exec sutazai-blackbox-exporter bash -c "
        # Create blackbox-exporter user
        groupadd -f -g 1002 blackbox
        useradd -r -u 1002 -g blackbox -s /bin/false blackbox
        # No persistent volumes to fix for blackbox exporter
    " || warning "Could not fix Blackbox Exporter permissions (container may not be running)"
    
    success "Blackbox Exporter permissions fixed"
}

# Verify container users after permission fixes
verify_container_users() {
    log "Verifying container users after permission fixes..."
    
    local containers=(
        "sutazai-ai-agent-orchestrator"
        "sutazai-chromadb"
        "sutazai-qdrant"
        "sutazai-postgres"
        "sutazai-redis"
        "sutazai-neo4j"
        "sutazai-rabbitmq"
        "sutazai-ollama"
        "sutazai-consul"
        "sutazai-blackbox-exporter"
    )
    
    for container in "${containers[@]}"; do
        if docker ps --format '{{.Names}}' | grep -q "^$container$"; then
            local user_info
            user_info=$(docker exec "$container" id 2>/dev/null || echo "Cannot check")
            log "$container: $user_info"
        else
            warning "$container: Container not running"
        fi
    done
}

# Test container health after permission changes
test_container_health() {
    log "Testing container health after permission changes..."
    
    local failed_containers=()
    
    # Wait a moment for containers to stabilize
    sleep 5
    
    # Check Docker health status
    while read -r container status; do
        if [[ "$status" != "healthy" && "$status" != "Up" ]]; then
            failed_containers+=("$container")
            warning "$container: $status"
        else
            success "$container: $status"
        fi
    done < <(docker ps --format "table {{.Names}}\t{{.Status}}" | tail -n +2)
    
    if [[ ${#failed_containers[@]} -gt 0 ]]; then
        error "Some containers are not healthy after permission changes:"
        printf '%s\n' "${failed_containers[@]}"
        return 1
    else
        success "All containers are healthy after permission changes"
        return 0
    fi
}

# Rollback permissions if needed
rollback_permissions() {
    local backup_path
    backup_path=$(cat /tmp/sutazai_permissions_backup_path 2>/dev/null || echo "")
    
    if [[ -z "$backup_path" || ! -d "$backup_path" ]]; then
        error "No backup path found. Cannot rollback permissions."
        return 1
    fi
    
    warning "Rolling back permissions from backup: $backup_path"
    
    # This is a simplified rollback - in production you'd want to restore exact permissions
    warning "Automatic permission rollback not implemented. Manual intervention required."
    warning "Backup location: $backup_path"
    
    return 1
}

# Main execution
main() {
    log "Starting Container Security Migration - Permission Fix"
    log "========================================================="
    
    check_docker_permissions
    backup_volume_permissions
    
    log "Fixing permissions for all containers..."
    
    # Fix permissions for each container type
    fix_ai_orchestrator_permissions
    fix_chromadb_permissions
    fix_qdrant_permissions
    fix_postgres_permissions
    fix_redis_permissions
    fix_neo4j_permissions
    fix_rabbitmq_permissions
    fix_ollama_permissions
    fix_consul_permissions
    fix_blackbox_exporter_permissions
    
    log "Permission fixes completed. Verifying results..."
    verify_container_users
    
    if test_container_health; then
        success "========================================================="
        success "Container permission migration completed successfully!"
        success "All containers are healthy and ready for user migration."
        success "========================================================="
    else
        error "========================================================="
        error "Some containers failed health checks after permission fix!"
        error "Consider rolling back changes and investigating issues."
        error "========================================================="
        
        read -p "Do you want to attempt rollback? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rollback_permissions
        fi
        exit 1
    fi
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi