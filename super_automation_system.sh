#!/bin/bash
set -euo pipefail

# Global Configuration
BASE_DIR=$(pwd)
LOG_DIR="/var/log/sutazai"
CONFIG_DIR="$BASE_DIR/config"
BACKUP_DIR="$BASE_DIR/backups"
MAX_RETRIES=3
RETRY_DELAY=30
MONITOR_INTERVAL=300 # 5 minutes

# Initialize logging
setup_logging() {
    mkdir -p "$LOG_DIR"
    exec > >(tee -a "$LOG_DIR/super_automation.log") 2>&1
    echo "📝 Super Automation System initialized at $(date)"
}

# Error handling
handle_error() {
    local exit_code=$?
    local error_message=$1
    
    echo "❌ Error: $error_message" | tee -a "$LOG_DIR/super_automation.log"
    echo "🔄 Initiating recovery..." | tee -a "$LOG_DIR/super_automation.log"
    comprehensive_recovery
    echo "📝 Check the full log at: $LOG_DIR/super_automation.log"
    exit $exit_code
}

# Comprehensive recovery
comprehensive_recovery() {
    echo "🔄 Starting comprehensive recovery..." | tee -a "$LOG_DIR/super_automation.log"
    
    # Stop all services
    docker-compose -f docker-compose.yml down --remove-orphans || true
    docker-compose -f docker-compose-super.yml down --remove-orphans || true
    docker-compose -f docker-compose-ai.yml down --remove-orphans || true
    
    # Clean up resources
    docker system prune -f
    docker volume prune -f
    docker network prune -f
    
    # Restore from backup if available
    if [ -d "$BACKUP_DIR/latest" ]; then
        echo "🔧 Restoring from backup..." | tee -a "$LOG_DIR/super_automation.log"
        rsync -a "$BACKUP_DIR/latest/" "$BASE_DIR/"
    fi
    
    echo "✅ Recovery completed" | tee -a "$LOG_DIR/super_automation.log"
}

# Automated backup
create_backup() {
    echo "💾 Creating system backup..." | tee -a "$LOG_DIR/super_automation.log"
    local timestamp=$(date +%Y%m%d%H%M%S)
    local backup_dir="$BACKUP_DIR/$timestamp"
    
    mkdir -p "$backup_dir"
    
    # Backup configurations
    rsync -a "$CONFIG_DIR/" "$backup_dir/config/"
    
    # Backup databases
    docker exec sutazai-postgres pg_dumpall -U postgres > "$backup_dir/postgres.sql"
    docker exec sutazai-redis redis-cli save && cp "$BASE_DIR/redis_data/dump.rdb" "$backup_dir/redis.rdb"
    
    # Update latest backup link
    ln -sfn "$backup_dir" "$BACKUP_DIR/latest"
    
    echo "✅ Backup created at $backup_dir" | tee -a "$LOG_DIR/super_automation.log"
}

# Self-healing system
self_healing() {
    echo "⚕️ Running self-healing checks..." | tee -a "$LOG_DIR/super_automation.log"
    
    # Check and restart failed services
    local failed_services=$(docker ps -a --filter "status=exited" --format "{{.Names}}")
    for service in $failed_services; do
        echo "🔧 Restarting failed service: $service" | tee -a "$LOG_DIR/super_automation.log"
        docker start $service || handle_error "Failed to restart service $service"
    done
    
    # Check resource usage
    local memory_usage=$(free -m | awk '/Mem:/ {print $3/$2 * 100}')
    if (( $(echo "$memory_usage > 90" | bc -l) )); then
        echo "⚠️ High memory usage detected: ${memory_usage}%" | tee -a "$LOG_DIR/super_automation.log"
        docker system prune -f
    fi
    
    echo "✅ Self-healing completed" | tee -a "$LOG_DIR/super_automation.log"
}

# Automated updates
system_updates() {
    echo "🔄 Checking for system updates..." | tee -a "$LOG_DIR/super_automation.log"
    
    # Update system packages
    sudo apt-get update && sudo apt-get upgrade -y
    
    # Update Docker images
    docker-compose -f docker-compose.yml pull
    docker-compose -f docker-compose-super.yml pull
    docker-compose -f docker-compose-ai.yml pull
    
    echo "✅ System updates completed" | tee -a "$LOG_DIR/super_automation.log"
}

# Comprehensive monitoring
monitor_system() {
    while true; do
        echo "📊 Monitoring system health..." | tee -a "$LOG_DIR/super_automation.log"
        
        # Check service health
        declare -A SERVICES=(
            ["sutazai-core"]="http://localhost:8000/health"
            ["super-agent"]="http://localhost:8001/health"
            ["ai-service"]="http://localhost:8002/health"
        )
        
        for service in "${!SERVICES[@]}"; do
            if ! curl -sSf "${SERVICES[$service]}" | grep -q "OK"; then
                handle_error "Service $service is unhealthy"
            fi
        done
        
        # Check resource usage
        local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
        local memory_usage=$(free -m | awk '/Mem:/ { printf "%.1f%%", $3/$2*100 }')
        local disk_usage=$(df -h / | awk 'NR==2 {print $5}')
        
        echo "📈 System Stats - CPU: ${cpu_usage}%, Memory: ${memory_usage}, Disk: ${disk_usage}" | tee -a "$LOG_DIR/super_automation.log"
        
        # Run self-healing if needed
        self_healing
        
        sleep $MONITOR_INTERVAL
    done
}

# Main automation function
automate_all() {
    echo "🚀 Starting Super Automation System..." | tee -a "$LOG_DIR/super_automation.log"
    
    # Phase 1: Initialization
    setup_logging
    create_backup
    
    # Phase 2: System Setup
    ./system_audit.sh || handle_error "System audit failed"
    ./hardware_health.sh || handle_error "Hardware health check failed"
    ./resource_limits.sh || handle_error "Resource limits check failed"
    ./performance_tuning.sh || handle_error "Performance tuning failed"
    
    # Phase 3: Service Deployment
    ./deploy_all.sh || handle_error "Deployment failed"
    
    # Phase 4: Verification
    ./service_dependency.sh || handle_error "Service dependency check failed"
    ./log_rotation_check.sh || handle_error "Log rotation check failed"
    ./log_integrity.sh || handle_error "Log integrity check failed"
    
    # Phase 5: Monitoring and Maintenance
    monitor_system &
    
    echo "🎉 Super Automation System is fully operational!" | tee -a "$LOG_DIR/super_automation.log"
}

# Execute automation
automate_all 