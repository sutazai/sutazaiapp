#!/bin/bash

# Strict error handling
set -euo pipefail

# Container Health Monitoring Script


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

LOG_FILE="/opt/sutazaiapp/logs/health-monitor.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

monitor_health() {
    local max_wait=600  # 10 minutes
    local wait_time=0
    local check_interval=15
    
    log "Starting container health monitoring..."
    
    while [[ $wait_time -lt $max_wait ]]; do
        local healthy_count=0
        local unhealthy_count=0
        local starting_count=0
        local total_count=0
        local no_health_count=0
        
        local unhealthy_containers=()
        local starting_containers=()
        
        # Check each SutazAI container
        while IFS=$'\t' read -r name status; do
            if [[ -n "$name" ]] && [[ "$name" =~ ^sutazai- ]]; then
                total_count=$((total_count + 1))
                
                # Extract health status from status field
                if [[ "$status" =~ "healthy" ]]; then
                    healthy_count=$((healthy_count + 1))
                elif [[ "$status" =~ "unhealthy" ]]; then
                    unhealthy_count=$((unhealthy_count + 1))
                    unhealthy_containers+=("$name")
                elif [[ "$status" =~ "starting" ]]; then
                    starting_count=$((starting_count + 1))
                    starting_containers+=("$name")
                else
                    no_health_count=$((no_health_count + 1))
                fi
            fi
        done < <(docker ps --format "{{.Names}}\t{{.Status}}")
        
        local health_rate=0
        local functional_containers=$((healthy_count + no_health_count))
        if [[ $total_count -gt 0 ]]; then
            health_rate=$((functional_containers * 100 / total_count))
        fi
        
        log "Health Status: $healthy_count healthy, $starting_count starting, $unhealthy_count unhealthy, $no_health_count no-health-check ($total_count total)"
        log "Functional Rate: $functional_containers/$total_count containers functional ($health_rate%)"
        
        if [[ ${#unhealthy_containers[@]} -gt 0 ]]; then
            log "Unhealthy: ${unhealthy_containers[*]}"
        fi
        
        if [[ ${#starting_containers[@]} -gt 0 ]] && [[ ${#starting_containers[@]} -lt 5 ]]; then
            log "Starting: ${starting_containers[*]}"
        fi
        
        # Consider success if we have 80% functional containers (healthy + no-health-check)
        if [[ $health_rate -ge 80 ]]; then
            log "SUCCESS: Target health rate achieved! ($health_rate%)"
            break
        fi
        
        # Also restart any containers that have been unhealthy for too long
        if [[ $wait_time -gt 300 ]] && [[ ${#unhealthy_containers[@]} -gt 0 ]]; then
            log "Restarting long-term unhealthy containers..."
            for container in "${unhealthy_containers[@]}"; do
                log "Restarting: $container"
                docker restart "$container" >/dev/null 2>&1 || log "Failed to restart $container"
            done
        fi
        
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    # Final summary
    log "=== Final Health Summary ==="
    log "Total monitoring time: $wait_time seconds"
    
    # Get final counts
    local final_healthy=0
    local final_unhealthy=0
    local final_total=0
    local final_functional=0
    
    while IFS=$'\t' read -r name status; do
        if [[ -n "$name" ]] && [[ "$name" =~ ^sutazai- ]]; then
            final_total=$((final_total + 1))
            
            if [[ "$status" =~ "healthy" ]]; then
                final_healthy=$((final_healthy + 1))
                final_functional=$((final_functional + 1))
            elif [[ "$status" =~ "unhealthy" ]]; then
                final_unhealthy=$((final_unhealthy + 1))
            else
                # No health check containers are considered functional if running
                if [[ "$status" =~ "Up" ]]; then
                    final_functional=$((final_functional + 1))
                fi
            fi
        fi
    done < <(docker ps --format "{{.Names}}\t{{.Status}}")
    
    local final_rate=0
    if [[ $final_total -gt 0 ]]; then
        final_rate=$((final_functional * 100 / final_total))
    fi
    
    log "Final Result: $final_functional/$final_total containers functional ($final_rate%)"
    log "Healthy containers: $final_healthy"
    log "Unhealthy containers: $final_unhealthy"
    
    if [[ $final_rate -ge 80 ]]; then
        log "SUCCESS: Container health fix completed successfully!"
        return 0
    else
        log "WARNING: Container health is still below target (80%)"
        return 1
    fi
}

# Execute monitoring
monitor_health "$@"