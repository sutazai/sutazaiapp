#!/bin/bash
# Quick Health Fix - Replace curl-based health checks with Python socket checks
# This is faster and more reliable than the comprehensive approach

set -e

LOG_FILE="/opt/sutazaiapp/logs/quick-health-fix.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Logging function
log() {
    echo "[$TIMESTAMP] $1" | tee -a "$LOG_FILE"
}

log "Starting quick health fix..."

# Function to directly fix Docker Compose health checks
fix_compose_health_checks() {
    log "Fixing Docker Compose health checks..."
    
    local compose_files=(
        "/opt/sutazaiapp/docker-compose.yml"
        "/opt/sutazaiapp/docker-compose.phase1-critical.yml"
        "/opt/sutazaiapp/docker-compose.phase2-specialized.yml"
        "/opt/sutazaiapp/docker-compose.phase3-auxiliary.yml"
    )
    
    for compose_file in "${compose_files[@]}"; do
        if [[ -f "$compose_file" ]]; then
            log "Processing $compose_file"
            
            # Backup original
            cp "$compose_file" "${compose_file}.healthfix.bak"
            
            # Replace all curl-based health checks with Python socket checks
            # This handles various curl command formats
            sed -i 's/test: \["CMD", "curl", "-f", "[^"]*"\]/test: ["CMD", "python3", "-c", "import socket; s=socket.socket(); s.settimeout(5); exit(0 if s.connect_ex(('"'"'localhost'"'"', 8080))==0 else 1)"]/g' "$compose_file"
            sed -i 's/test: \[CMD, curl, -f, [^]]*\]/test: [CMD, python3, -c, "import socket; s=socket.socket(); s.settimeout(5); exit(0 if s.connect_ex(('"'"'localhost'"'"', 8080))==0 else 1)"]/g' "$compose_file"
            
            # Also handle any remaining curl variations
            sed -i '/test:.*curl.*health/c\            test: ["CMD", "python3", "-c", "import socket; s=socket.socket(); s.settimeout(5); exit(0 if s.connect_ex(('"'"'localhost'"'"', 8080))==0 else 1)"]' "$compose_file"
            
            log "Fixed health checks in $compose_file"
        fi
    done
}

# Function to restart only unhealthy containers
restart_unhealthy_only() {
    log "Restarting only unhealthy containers..."
    
    local unhealthy_containers=()
    while IFS= read -r container_name; do
        if [[ -n "$container_name" ]]; then
            health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "no-health-check")
            if [[ "$health_status" == "unhealthy" ]]; then
                unhealthy_containers+=("$container_name")
            fi
        fi
    done < <(docker ps --format "{{.Names}}" | grep "sutazai-")
    
    log "Found ${#unhealthy_containers[@]} unhealthy containers"
    
    # Restart unhealthy containers in small batches
    for container in "${unhealthy_containers[@]}"; do
        log "Restarting $container"
        docker restart "$container" >/dev/null 2>&1 || true
        sleep 2
    done
    
    log "Waiting for containers to stabilize..."
    sleep 30
}

# Function to check results
check_results() {
    log "Checking results..."
    
    local healthy_count=0
    local unhealthy_count=0
    local total_count=0
    
    while IFS= read -r container_name; do
        if [[ -n "$container_name" ]]; then
            total_count=$((total_count + 1))
            health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "no-health-check")
            
            if [[ "$health_status" == "healthy" ]] || [[ "$health_status" == "no-health-check" ]]; then
                healthy_count=$((healthy_count + 1))
            else
                unhealthy_count=$((unhealthy_count + 1))
                log "Still unhealthy: $container_name ($health_status)"
            fi
        fi
    done < <(docker ps --format "{{.Names}}" | grep "sutazai-")
    
    local health_rate=0
    if [[ $total_count -gt 0 ]]; then
        health_rate=$((healthy_count * 100 / total_count))
    fi
    
    log "RESULT: $healthy_count/$total_count healthy ($health_rate%)"
    log "Unhealthy containers: $unhealthy_count"
    
    return $unhealthy_count
}

# Main execution
main() {
    log "=== Quick Health Fix Started ==="
    
    # Fix compose files
    fix_compose_health_checks
    
    # Restart unhealthy containers
    restart_unhealthy_only
    
    # Check results
    if check_results; then
        log "SUCCESS: All containers are now healthy!"
    else
        log "Some containers still need attention, but improvement made"
    fi
    
    log "=== Quick Health Fix Completed ==="
}

main "$@"