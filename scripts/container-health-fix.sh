#!/bin/bash
# SutazAI Container Health Fix Script
# Fixes common container health issues and optimizes resource allocation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/opt/sutazaiapp/logs/container-health-fix.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "[$TIMESTAMP] $1" | tee -a "$LOG_FILE"
}

log "Starting SutazAI Container Health Fix..."

# Function to check if container is running
is_container_running() {
    local container_name="$1"
    docker ps --format "table {{.Names}}" | grep -q "^$container_name$"
}

# Function to check container health
get_container_health() {
    local container_name="$1"
    docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "no-health-check"
}

# Function to fix container health check timeouts
fix_health_check_timeouts() {
    log "Fixing health check timeouts..."
    
    # Update docker-compose files to increase health check timeouts
    local compose_files=(
        "/opt/sutazaiapp/docker-compose.yml"
        "/opt/sutazaiapp/docker-compose.phase3-auxiliary.yml"
    )
    
    for compose_file in "${compose_files[@]}"; do
        if [[ -f "$compose_file" ]]; then
            log "Updating health check timeouts in $compose_file"
            
            # Backup original file
            cp "$compose_file" "${compose_file}.bak.$(date +%s)"
            
            # Update timeout values using sed
            sed -i 's/timeout: 10s/timeout: 30s/g' "$compose_file"
            sed -i 's/interval: 30s/interval: 60s/g' "$compose_file"
            sed -i 's/retries: 3/retries: 5/g' "$compose_file"
            sed -i 's/start_period: 30s/start_period: 60s/g' "$compose_file"
            sed -i 's/start_period: 40s/start_period: 90s/g' "$compose_file"
        fi
    done
}

# Function to clean up Docker resources
cleanup_docker_resources() {
    log "Cleaning up Docker resources..."
    
    # Remove unused images
    log "Removing unused Docker images..."
    REMOVED_IMAGES=$(docker image prune -f --filter "dangling=true" | grep "Total reclaimed space" || echo "No images removed")
    log "Images cleanup: $REMOVED_IMAGES"
    
    # Remove stopped containers
    log "Removing stopped containers..."
    REMOVED_CONTAINERS=$(docker container prune -f | grep "Total reclaimed space" || echo "No containers removed")
    log "Containers cleanup: $REMOVED_CONTAINERS"
    
    # Remove unused volumes (be careful with this)
    log "Removing unused volumes..."
    REMOVED_VOLUMES=$(docker volume prune -f | grep "Total reclaimed space" || echo "No volumes removed")
    log "Volumes cleanup: $REMOVED_VOLUMES"
    
    # Remove unused networks
    log "Removing unused networks..."
    docker network prune -f >/dev/null 2>&1
    log "Networks cleanup completed"
}

# Function to restart unhealthy containers
restart_unhealthy_containers() {
    log "Checking and restarting unhealthy containers..."
    
    # Get list of containers with issues
    local unhealthy_containers=()
    local restarting_containers=()
    
    # Find unhealthy containers
    while IFS= read -r container_name; do
        if [[ -n "$container_name" ]]; then
            health_status=$(get_container_health "$container_name")
            container_status=$(docker ps -a --format "table {{.Names}}\t{{.Status}}" | grep "$container_name" | awk '{print $2}')
            
            if [[ "$health_status" == "unhealthy" ]]; then
                unhealthy_containers+=("$container_name")
            elif [[ "$container_status" == "Restarting" ]]; then
                restarting_containers+=("$container_name")
            fi
        fi
    done < <(docker ps -a --format "{{.Names}}" | grep "sutazai-")
    
    log "Found ${#unhealthy_containers[@]} unhealthy containers"
    log "Found ${#restarting_containers[@]} restarting containers"
    
    # Restart unhealthy containers
    for container in "${unhealthy_containers[@]}"; do
        log "Restarting unhealthy container: $container"
        docker restart "$container" >/dev/null 2>&1 || log "Failed to restart $container"
        sleep 5
    done
    
    # Stop and restart containers stuck in restarting state
    for container in "${restarting_containers[@]}"; do
        log "Stopping and restarting stuck container: $container"
        docker stop "$container" >/dev/null 2>&1 || true
        sleep 2
        docker start "$container" >/dev/null 2>&1 || log "Failed to start $container"
        sleep 5
    done
}

# Function to optimize container resource limits
optimize_container_resources() {
    log "Optimizing container resource allocation..."
    
    # Create optimized docker-compose override
    cat > "/opt/sutazaiapp/docker-compose.override.yml" << 'EOF'
version: '3.8'

# Resource optimization overrides
x-agent-optimized: &agent-optimized
  deploy:
    resources:
      limits:
        cpus: '0.3'
        memory: 384M
      reservations:
        cpus: '0.1'
        memory: 128M
  restart: unless-stopped
  healthcheck:
    interval: 60s
    timeout: 30s
    retries: 5
    start_period: 90s

services:
  # Core services optimization
  postgres:
    deploy:
      resources:
        limits:
          cpus: '1.5'
          memory: 1.5G
        reservations:
          cpus: '0.5'
          memory: 512M
  
  neo4j:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 3G
        reservations:
          cpus: '1'
          memory: 1G
  
  ollama:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
        reservations:
          cpus: '4'
          memory: 8G
  
  redis:
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
EOF

    log "Created resource optimization override file"
}

# Function to check system resource usage
check_system_resources() {
    log "Checking system resource usage..."
    
    # Memory usage
    local memory_info=$(free -h | grep "Mem:")
    log "Memory: $memory_info"
    
    # CPU load
    local load_avg=$(uptime | awk -F'load average:' '{print $2}')
    log "Load average:$load_avg"
    
    # Disk usage
    local disk_usage=$(df -h / | tail -1)
    log "Disk: $disk_usage"
    
    # Docker stats
    local container_count=$(docker ps | wc -l)
    local running_containers=$((container_count - 1))
    log "Running containers: $running_containers"
    
    # Health status summary
    local healthy_count=0
    local unhealthy_count=0
    local total_count=0
    
    while IFS= read -r container_name; do
        if [[ -n "$container_name" ]]; then
            health_status=$(get_container_health "$container_name")
            total_count=$((total_count + 1))
            
            if [[ "$health_status" == "healthy" ]]; then
                healthy_count=$((healthy_count + 1))
            elif [[ "$health_status" == "unhealthy" ]]; then
                unhealthy_count=$((unhealthy_count + 1))
            fi
        fi
    done < <(docker ps --format "{{.Names}}" | grep "sutazai-")
    
    local health_rate=0
    if [[ $total_count -gt 0 ]]; then
        health_rate=$((healthy_count * 100 / total_count))
    fi
    
    log "Container health: $healthy_count/$total_count healthy ($health_rate%)"
}

# Function to apply fixes based on specific issues
apply_targeted_fixes() {
    log "Applying targeted fixes..."
    
    # Fix Ollama connection issues
    if ! curl -s http://localhost:10104/api/tags >/dev/null 2>&1; then
        log "Ollama API not responding, restarting..."
        docker restart sutazai-ollama >/dev/null 2>&1 || true
        sleep 30
    fi
    
    # Fix Redis connection issues
    if ! docker exec sutazai-redis redis-cli ping >/dev/null 2>&1; then
        log "Redis not responding, restarting..."
        docker restart sutazai-redis >/dev/null 2>&1 || true
        sleep 10
    fi
    
    # Fix Neo4j connection issues
    if ! docker exec sutazai-neo4j cypher-shell "RETURN 1" >/dev/null 2>&1; then
        log "Neo4j not responding, restarting..."
        docker restart sutazai-neo4j >/dev/null 2>&1 || true
        sleep 30
    fi
    
    # Fix network issues
    log "Recreating Docker network if needed..."
    if ! docker network inspect sutazai-network >/dev/null 2>&1; then
        log "Recreating sutazai-network..."
        docker network create sutazai-network >/dev/null 2>&1 || true
    fi
}

# Function to wait for services to be healthy
wait_for_services() {
    log "Waiting for services to become healthy..."
    
    local max_wait=300  # 5 minutes
    local wait_time=0
    local check_interval=15
    
    while [[ $wait_time -lt $max_wait ]]; do
        local healthy_count=0
        local total_count=0
        
        while IFS= read -r container_name; do
            if [[ -n "$container_name" ]]; then
                total_count=$((total_count + 1))
                health_status=$(get_container_health "$container_name")
                
                if [[ "$health_status" == "healthy" ]] || [[ "$health_status" == "no-health-check" ]]; then
                    healthy_count=$((healthy_count + 1))
                fi
            fi
        done < <(docker ps --format "{{.Names}}" | grep "sutazai-")
        
        local health_rate=0
        if [[ $total_count -gt 0 ]]; then
            health_rate=$((healthy_count * 100 / total_count))
        fi
        
        log "Health check: $healthy_count/$total_count containers healthy ($health_rate%)"
        
        if [[ $health_rate -ge 80 ]]; then
            log "Acceptable health rate achieved!"
            break
        fi
        
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
}

# Main execution
main() {
    log "=== SutazAI Container Health Fix Started ==="
    
    # Check initial system state
    check_system_resources
    
    # Apply fixes
    fix_health_check_timeouts
    cleanup_docker_resources
    optimize_container_resources
    apply_targeted_fixes
    restart_unhealthy_containers
    
    # Wait for services to stabilize
    wait_for_services
    
    # Final system check
    log "=== Final System Status ==="
    check_system_resources
    
    log "=== Container Health Fix Completed ==="
    log "Check log file for details: $LOG_FILE"
}

# Parse command line arguments
case "${1:-all}" in
    "cleanup")
        cleanup_docker_resources
        ;;
    "restart")
        restart_unhealthy_containers
        ;;
    "optimize")
        optimize_container_resources
        ;;
    "check")
        check_system_resources
        ;;
    "all"|*)
        main
        ;;
esac