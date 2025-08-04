#!/bin/bash
# Container Auto-Healer Service

HEAL_LOG="/opt/sutazaiapp/logs/auto-healer.log"

log_heal() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$HEAL_LOG"
}

heal_container() {
    local container_name="$1"
    local health_status="$2"
    
    log_heal "Healing container: $container_name (status: $health_status)"
    
    case "$health_status" in
        "unhealthy")
            log_heal "Restarting unhealthy container: $container_name"
            docker restart "$container_name" 2>/dev/null || log_heal "Failed to restart $container_name"
            ;;
        "restarting")
            # If container has been restarting too long, force restart
            local restart_count=$(docker inspect "$container_name" --format='{{.RestartCount}}' 2>/dev/null || echo "0")
            if [[ $restart_count -gt 5 ]]; then
                log_heal "Force stopping and starting stuck container: $container_name"
                docker stop "$container_name" 2>/dev/null || true
                sleep 5
                docker start "$container_name" 2>/dev/null || log_heal "Failed to start $container_name"
            fi
            ;;
        "exited")
            log_heal "Starting exited container: $container_name"
            docker start "$container_name" 2>/dev/null || log_heal "Failed to start $container_name"
            ;;
    esac
}

# Main healing loop
while true; do
    # Check all SutazAI containers
    while IFS=$'\t' read -r name status health; do
        if [[ -n "$name" ]] && [[ "$name" =~ ^sutazai- ]]; then
            case "$health" in
                "unhealthy"|"starting")
                    heal_container "$name" "$health"
                    ;;
            esac
            
            # Check for containers stuck in restarting state
            if [[ "$status" =~ "Restarting" ]]; then
                heal_container "$name" "restarting"
            fi
            
            # Check for exited containers
            if [[ "$status" =~ "Exited" ]]; then
                heal_container "$name" "exited"
            fi
        fi
    done < <(docker ps -a --format "{{.Names}}\t{{.Status}}\t{{.State}}" 2>/dev/null)
    
    # Wait before next check
    sleep 30
done
