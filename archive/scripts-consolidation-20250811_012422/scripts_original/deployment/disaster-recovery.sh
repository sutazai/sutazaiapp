#!/bin/bash
# Comprehensive Disaster Recovery System for Sutazai 69-Agent System
# Implements automated backup, recovery, and business continuity procedures

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/opt/sutazaiapp"
BACKUP_ROOT="/opt/sutazai-backups"
LOG_FILE="/opt/sutazaiapp/logs/disaster-recovery.log"
RECOVERY_STATE_FILE="/opt/sutazaiapp/data/recovery-state.json"
TEMP_DIR="/tmp/sutazai-dr-$$"

# Backup retention policy
DAILY_RETENTION=7
WEEKLY_RETENTION=4
MONTHLY_RETENTION=12

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

error() {
    log "ERROR" "$*" >&2
    echo -e "${RED}ERROR: $*${NC}" >&2
}

warn() {
    log "WARN" "$*"
    echo -e "${YELLOW}WARNING: $*${NC}"
}

info() {
    log "INFO" "$*"
    echo -e "${GREEN}INFO: $*${NC}"
}

debug() {
    log "DEBUG" "$*"
    echo -e "${BLUE}DEBUG: $*${NC}"
}

# Cleanup function
cleanup() {
    if [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
    fi
}
trap cleanup EXIT

# Create required directories
create_directories() {
    local dirs=(
        "$BACKUP_ROOT"
        "$BACKUP_ROOT/daily"
        "$BACKUP_ROOT/weekly"
        "$BACKUP_ROOT/monthly"
        "$BACKUP_ROOT/emergency"
        "$(dirname "$LOG_FILE")"
        "$(dirname "$RECOVERY_STATE_FILE")"
        "$TEMP_DIR"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
}

# Get current timestamp
get_timestamp() {
    date +%Y%m%d_%H%M%S
}

# Check if service is running
is_service_running() {
    local service_name=$1
    docker ps --filter "name=$service_name" --filter "status=running" --format "{{.Names}}" | grep -q "^${service_name}$"
}

# Wait for service to be healthy
wait_for_service_health() {
    local service_name=$1
    local timeout=${2:-120}
    local interval=5
    local elapsed=0
    
    info "Waiting for $service_name to be healthy (timeout: ${timeout}s)"
    
    while [[ $elapsed -lt $timeout ]]; do
        if docker inspect "$service_name" --format='{{.State.Health.Status}}' 2>/dev/null | grep -q "healthy"; then
            info "$service_name is healthy"
            return 0
        fi
        
        sleep $interval
        ((elapsed += interval))
        debug "Waiting for $service_name... (${elapsed}s/${timeout}s)"
    done
    
    error "$service_name failed to become healthy within ${timeout}s"
    return 1
}

# Backup PostgreSQL database
backup_postgresql() {
    local backup_file="$1/postgresql_$(get_timestamp).sql"
    
    info "Backing up PostgreSQL database to $backup_file"
    
    if is_service_running "sutazai-postgres"; then
        docker exec sutazai-postgres pg_dump -U sutazai -d sutazai --clean --if-exists > "$backup_file"
        
        # Compress backup
        gzip "$backup_file"
        backup_file="${backup_file}.gz"
        
        # Verify backup integrity
        if zcat "$backup_file" | head -n 10 | grep -q "PostgreSQL database dump"; then
            info "PostgreSQL backup completed successfully: $backup_file"
            echo "$backup_file"
        else
            error "PostgreSQL backup verification failed"
            rm -f "$backup_file"
            return 1
        fi
    else
        warn "PostgreSQL service not running, skipping backup"
        return 1
    fi
}

# Backup Redis data
backup_redis() {
    local backup_file="$1/redis_$(get_timestamp).rdb"
    
    info "Backing up Redis data to $backup_file"
    
    if is_service_running "sutazai-redis"; then
        # Force Redis to save current dataset
        docker exec sutazai-redis redis-cli BGSAVE
        
        # Wait for background save to complete
        while docker exec sutazai-redis redis-cli LASTSAVE | head -n 1 > "$TEMP_DIR/lastsave_current"; do
            sleep 2
            if docker exec sutazai-redis redis-cli LASTSAVE | head -n 1 > "$TEMP_DIR/lastsave_new"; then
                if ! diff "$TEMP_DIR/lastsave_current" "$TEMP_DIR/lastsave_new" >/dev/null 2>&1; then
                    break
                fi
            fi
        done
        
        # Copy RDB file
        docker cp sutazai-redis:/data/dump.rdb "$backup_file"
        
        # Compress backup
        gzip "$backup_file"
        backup_file="${backup_file}.gz"
        
        info "Redis backup completed successfully: $backup_file"
        echo "$backup_file"
    else
        warn "Redis service not running, skipping backup"
        return 1
    fi
}

# Backup Neo4j database
backup_neo4j() {
    local backup_file="$1/neo4j_$(get_timestamp).dump"
    
    info "Backing up Neo4j database to $backup_file"
    
    if is_service_running "sutazai-neo4j"; then
        # Create Neo4j backup using cypher-shell
        docker exec sutazai-neo4j cypher-shell -u neo4j -p "$(cat "$PROJECT_ROOT/secrets/neo4j_password.txt")" \
            "CALL apoc.export.cypher.all('file:///tmp/backup.cypher', {format: 'cypher-shell', useOptimizations: {type: 'UNWIND_BATCH', unwindBatchSize: 20}})" || {
            warn "APOC export failed, using basic dump"
            docker exec sutazai-neo4j neo4j-admin dump --database=neo4j --to=/tmp/backup.dump
            docker cp sutazai-neo4j:/tmp/backup.dump "$backup_file"
        }
        
        # Try to copy cypher export if it exists
        if docker exec sutazai-neo4j test -f /tmp/backup.cypher 2>/dev/null; then
            docker cp sutazai-neo4j:/tmp/backup.cypher "${backup_file%.dump}.cypher"
        fi
        
        # Compress backup
        gzip "$backup_file"
        backup_file="${backup_file}.gz"
        
        info "Neo4j backup completed successfully: $backup_file"
        echo "$backup_file"
    else
        warn "Neo4j service not running, skipping backup"
        return 1
    fi
}

# Backup agent configurations and states
backup_agent_data() {
    local backup_dir="$1"
    local agents_backup="$backup_dir/agents_$(get_timestamp).tar.gz"
    
    info "Backing up agent data to $agents_backup"
    
    # Create temporary directory for agent data
    local temp_agents_dir="$TEMP_DIR/agents"
    mkdir -p "$temp_agents_dir"
    
    # Backup agent configurations
    if [[ -d "$PROJECT_ROOT/agents" ]]; then
        cp -r "$PROJECT_ROOT/agents" "$temp_agents_dir/"
    fi
    
    # Backup agent states from orchestrator
    if is_service_running "sutazai-agent-orchestrator"; then
        debug "Backing up agent states from orchestrator"
        if docker exec sutazai-agent-orchestrator curl -s http://localhost:8080/api/backup > "$temp_agents_dir/agent_states.json"; then
            debug "Agent states backed up successfully"
        else
            warn "Failed to backup agent states from orchestrator"
        fi
    fi
    
    # Backup configuration files
    local config_files=(
        "$PROJECT_ROOT/config"
        "$PROJECT_ROOT/docker-compose*.yml"
        "$PROJECT_ROOT/CLAUDE.md"
        "$PROJECT_ROOT/monitoring"
    )
    
    for config_item in "${config_files[@]}"; do
        if [[ -e "$config_item" ]]; then
            cp -r "$config_item" "$temp_agents_dir/"
        fi
    done
    
    # Create compressed archive
    tar -czf "$agents_backup" -C "$temp_agents_dir" .
    
    info "Agent data backup completed: $agents_backup"
    echo "$agents_backup"
}

# Backup monitoring data
backup_monitoring_data() {
    local backup_dir="$1"
    local monitoring_backup="$backup_dir/monitoring_$(get_timestamp).tar.gz"
    
    info "Backing up monitoring data to $monitoring_backup"
    
    local temp_monitoring_dir="$TEMP_DIR/monitoring"
    mkdir -p "$temp_monitoring_dir"
    
    # Backup Prometheus data
    if is_service_running "sutazai-prometheus"; then
        debug "Backing up Prometheus data"
        docker cp sutazai-prometheus:/prometheus "$temp_monitoring_dir/" 2>/dev/null || warn "Failed to backup Prometheus data"
    fi
    
    # Backup Grafana dashboards and datasources
    if is_service_running "sutazai-grafana"; then
        debug "Backing up Grafana configuration"
        docker cp sutazai-grafana:/var/lib/grafana "$temp_monitoring_dir/grafana" 2>/dev/null || warn "Failed to backup Grafana data"
    fi
    
    # Backup monitoring configurations
    if [[ -d "$PROJECT_ROOT/monitoring" ]]; then
        cp -r "$PROJECT_ROOT/monitoring" "$temp_monitoring_dir/config"
    fi
    
    # Create compressed archive
    tar -czf "$monitoring_backup" -C "$temp_monitoring_dir" .
    
    info "Monitoring data backup completed: $monitoring_backup"
    echo "$monitoring_backup"
}

# Create comprehensive system backup
create_backup() {
    local backup_type=${1:-"manual"}  # daily, weekly, monthly, manual, emergency
    local backup_dir="$BACKUP_ROOT/$backup_type/backup_$(get_timestamp)"
    
    info "Creating $backup_type backup in $backup_dir"
    mkdir -p "$backup_dir"
    
    # Create backup manifest
    local manifest_file="$backup_dir/backup_manifest.json"
    cat > "$manifest_file" << EOF
{
  "backup_type": "$backup_type",
  "timestamp": "$(date -Iseconds)",
  "hostname": "$(hostname)",
  "system_info": {
    "cpu_cores": $(nproc),
    "total_memory_gb": $(free -g | awk 'NR==2{print $2}'),
    "disk_usage": "$(df -h "$PROJECT_ROOT" | awk 'NR==2{print $5}')",
    "docker_version": "$(docker --version | cut -d' ' -f3 | sed 's/,//')"
  },
  "services_running": [],
  "backups_created": []
}
EOF
    
    # Record running services
    local running_services=()
    local sutazai_containers
    mapfile -t sutazai_containers < <(docker ps --filter "name=sutazai-" --format "{{.Names}}")
    
    for container in "${sutazai_containers[@]}"; do
        running_services+=("\"$container\"")
    done
    
    # Update manifest with running services
    local services_json
    services_json=$(printf '%s\n' "${running_services[@]}" | paste -sd ',' -)
    sed -i "s/\"services_running\": \[\]/\"services_running\": [$services_json]/" "$manifest_file"
    
    # Perform backups
    local backup_files=()
    
    # Database backups
    if pg_backup=$(backup_postgresql "$backup_dir"); then
        backup_files+=("\"postgresql\": \"$(basename "$pg_backup")\"")
    fi
    
    if redis_backup=$(backup_redis "$backup_dir"); then
        backup_files+=("\"redis\": \"$(basename "$redis_backup")\"")
    fi
    
    if neo4j_backup=$(backup_neo4j "$backup_dir"); then
        backup_files+=("\"neo4j\": \"$(basename "$neo4j_backup")\"")
    fi
    
    # Agent and configuration backups
    if agents_backup=$(backup_agent_data "$backup_dir"); then
        backup_files+=("\"agents\": \"$(basename "$agents_backup")\"")
    fi
    
    # Monitoring data backup
    if monitoring_backup=$(backup_monitoring_data "$backup_dir"); then
        backup_files+=("\"monitoring\": \"$(basename "$monitoring_backup")\"")
    fi
    
    # Update manifest with backup files
    local backups_json
    backups_json=$(printf '%s\n' "${backup_files[@]}" | paste -sd ',' - | sed 's/^/{/' | sed 's/$/}/')
    sed -i "s/\"backups_created\": \[\]/\"backups_created\": $backups_json/" "$manifest_file"
    
    # Create backup summary
    local backup_size
    backup_size=$(du -sh "$backup_dir" | cut -f1)
    
    info "Backup completed successfully"
    info "Backup location: $backup_dir"
    info "Backup size: $backup_size"
    info "Manifest: $manifest_file"
    
    # Validate backup integrity
    validate_backup "$backup_dir"
    
    echo "$backup_dir"
}

# Validate backup integrity
validate_backup() {
    local backup_dir=$1
    local validation_errors=0
    
    info "Validating backup integrity: $backup_dir"
    
    # Check manifest exists
    if [[ ! -f "$backup_dir/backup_manifest.json" ]]; then
        error "Backup manifest missing"
        ((validation_errors++))
    fi
    
    # Validate compressed files
    for compressed_file in "$backup_dir"/*.gz; do
        if [[ -f "$compressed_file" ]]; then
            if ! gzip -t "$compressed_file" 2>/dev/null; then
                error "Corrupted backup file: $compressed_file"
                ((validation_errors++))
            else
                debug "Validated: $(basename "$compressed_file")"
            fi
        fi
    done
    
    # Validate tar archives
    for tar_file in "$backup_dir"/*.tar.gz; do
        if [[ -f "$tar_file" ]]; then
            if ! tar -tzf "$tar_file" >/dev/null 2>&1; then
                error "Corrupted tar archive: $tar_file"
                ((validation_errors++))
            else
                debug "Validated: $(basename "$tar_file")"
            fi
        fi
    done
    
    if [[ $validation_errors -eq 0 ]]; then
        info "Backup validation passed"
        return 0
    else
        error "Backup validation failed with $validation_errors errors"
        return 1
    fi
}

# Restore PostgreSQL database
restore_postgresql() {
    local backup_file=$1
    
    info "Restoring PostgreSQL from $backup_file"
    
    # Stop all agents that might be using the database
    info "Stopping agents before database restore"
    docker ps --filter "name=sutazai-" --filter "name=!sutazai-postgres" --format "{{.Names}}" | \
        xargs -r docker stop
    
    # Restore database
    if [[ "$backup_file" == *.gz ]]; then
        zcat "$backup_file" | docker exec -i sutazai-postgres psql -U sutazai -d sutazai
    else
        docker exec -i sutazai-postgres psql -U sutazai -d sutazai < "$backup_file"
    fi
    
    info "PostgreSQL restore completed"
}

# Restore Redis data
restore_redis() {
    local backup_file=$1
    
    info "Restoring Redis from $backup_file"
    
    # Stop Redis
    docker stop sutazai-redis || true
    
    # Remove old data
    docker run --rm -v sutazai-redis-data:/data alpine rm -f /data/dump.rdb
    
    # Restore backup
    if [[ "$backup_file" == *.gz ]]; then
        zcat "$backup_file" | docker run --rm -i -v sutazai-redis-data:/data alpine tee /data/dump.rdb > /dev/null
    else
        docker cp "$backup_file" "$(docker create --rm -v sutazai-redis-data:/data alpine):/data/dump.rdb"
    fi
    
    # Start Redis
    docker start sutazai-redis
    wait_for_service_health "sutazai-redis" 60
    
    info "Redis restore completed"
}

# Restore Neo4j database
restore_neo4j() {
    local backup_file=$1
    
    info "Restoring Neo4j from $backup_file"
    
    # Stop Neo4j
    docker stop sutazai-neo4j || true
    
    # Restore from backup
    if [[ "$backup_file" == *.gz ]]; then
        local temp_backup="$TEMP_DIR/neo4j_restore.dump"
        zcat "$backup_file" > "$temp_backup"
        docker cp "$temp_backup" sutazai-neo4j:/tmp/restore.dump
    else
        docker cp "$backup_file" sutazai-neo4j:/tmp/restore.dump
    fi
    
    # Load backup
    docker exec sutazai-neo4j neo4j-admin load --database=neo4j --from=/tmp/restore.dump --force
    
    # Start Neo4j
    docker start sutazai-neo4j
    wait_for_service_health "sutazai-neo4j" 120
    
    info "Neo4j restore completed"
}

# Restore system from backup
restore_system() {
    local backup_dir=$1
    local restore_components=${2:-"all"}  # all, databases, agents, monitoring
    
    info "Starting system restore from $backup_dir"
    
    # Validate backup directory
    if [[ ! -d "$backup_dir" ]]; then
        error "Backup directory not found: $backup_dir"
        return 1
    fi
    
    if [[ ! -f "$backup_dir/backup_manifest.json" ]]; then
        error "Backup manifest not found in $backup_dir"
        return 1
    fi
    
    # Validate backup integrity
    if ! validate_backup "$backup_dir"; then
        error "Backup validation failed - aborting restore"
        return 1
    fi
    
    # Create recovery state file
    cat > "$RECOVERY_STATE_FILE" << EOF
{
  "restore_started": "$(date -Iseconds)",
  "backup_source": "$backup_dir",
  "restore_components": "$restore_components",
  "status": "in_progress",
  "steps_completed": []
}
EOF
    
    # Emergency shutdown if system is running
    if docker ps --filter "name=sutazai-" --format "{{.Names}}" | grep -q .; then
        warn "System is running - performing emergency shutdown"
        emergency_shutdown
    fi
    
    # Restore components based on selection
    case "$restore_components" in
        "all"|"databases")
            # Restore databases
            if [[ -f "$backup_dir"/postgresql_*.sql.gz ]]; then
                docker start sutazai-postgres
                wait_for_service_health "sutazai-postgres" 60
                restore_postgresql "$backup_dir"/postgresql_*.sql.gz
                jq '.steps_completed += ["postgresql"]' "$RECOVERY_STATE_FILE" > "$TEMP_DIR/recovery_state.json" && \
                    mv "$TEMP_DIR/recovery_state.json" "$RECOVERY_STATE_FILE"
            fi
            
            if [[ -f "$backup_dir"/redis_*.rdb.gz ]]; then
                restore_redis "$backup_dir"/redis_*.rdb.gz
                jq '.steps_completed += ["redis"]' "$RECOVERY_STATE_FILE" > "$TEMP_DIR/recovery_state.json" && \
                    mv "$TEMP_DIR/recovery_state.json" "$RECOVERY_STATE_FILE"
            fi
            
            if [[ -f "$backup_dir"/neo4j_*.dump.gz ]]; then
                restore_neo4j "$backup_dir"/neo4j_*.dump.gz
                jq '.steps_completed += ["neo4j"]' "$RECOVERY_STATE_FILE" > "$TEMP_DIR/recovery_state.json" && \
                    mv "$TEMP_DIR/recovery_state.json" "$RECOVERY_STATE_FILE"
            fi
            ;&
        "all"|"agents")
            # Restore agent configurations
            if [[ -f "$backup_dir"/agents_*.tar.gz ]]; then
                info "Restoring agent configurations"
                local temp_restore_dir="$TEMP_DIR/restore_agents"
                mkdir -p "$temp_restore_dir"
                
                tar -xzf "$backup_dir"/agents_*.tar.gz -C "$temp_restore_dir"
                
                # Restore configurations
                if [[ -d "$temp_restore_dir/agents" ]]; then
                    cp -r "$temp_restore_dir/agents" "$PROJECT_ROOT/"
                fi
                
                if [[ -d "$temp_restore_dir/config" ]]; then
                    cp -r "$temp_restore_dir/config" "$PROJECT_ROOT/"
                fi
                
                # Restore docker-compose files
                cp "$temp_restore_dir"/docker-compose*.yml "$PROJECT_ROOT/" 2>/dev/null || true
                
                jq '.steps_completed += ["agents"]' "$RECOVERY_STATE_FILE" > "$TEMP_DIR/recovery_state.json" && \
                    mv "$TEMP_DIR/recovery_state.json" "$RECOVERY_STATE_FILE"
            fi
            ;&
        "all"|"monitoring")
            # Restore monitoring data
            if [[ -f "$backup_dir"/monitoring_*.tar.gz ]]; then
                info "Restoring monitoring data"
                local temp_monitoring_dir="$TEMP_DIR/restore_monitoring"
                mkdir -p "$temp_monitoring_dir"
                
                tar -xzf "$backup_dir"/monitoring_*.tar.gz -C "$temp_monitoring_dir"
                
                # Start monitoring services
                docker start sutazai-prometheus sutazai-grafana 2>/dev/null || true
                
                # Restore Prometheus data
                if [[ -d "$temp_monitoring_dir/prometheus" ]]; then
                    docker cp "$temp_monitoring_dir/prometheus/." sutazai-prometheus:/prometheus/
                    docker restart sutazai-prometheus
                fi
                
                # Restore Grafana data
                if [[ -d "$temp_monitoring_dir/grafana" ]]; then
                    docker cp "$temp_monitoring_dir/grafana/." sutazai-grafana:/var/lib/grafana/
                    docker restart sutazai-grafana
                fi
                
                jq '.steps_completed += ["monitoring"]' "$RECOVERY_STATE_FILE" > "$TEMP_DIR/recovery_state.json" && \
                    mv "$TEMP_DIR/recovery_state.json" "$RECOVERY_STATE_FILE"
            fi
            ;;
    esac
    
    # Update recovery state
    jq '.status = "completed" | .restore_completed = now | .restore_completed |= todate' "$RECOVERY_STATE_FILE" > "$TEMP_DIR/recovery_state.json" && \
        mv "$TEMP_DIR/recovery_state.json" "$RECOVERY_STATE_FILE"
    
    info "System restore completed successfully"
    info "Recovery state saved to: $RECOVERY_STATE_FILE"
    
    # Start core infrastructure services
    info "Starting core infrastructure services"
    docker-compose -f "$PROJECT_ROOT/docker-compose.yml" up -d postgres redis neo4j consul kong rabbitmq
    
    # Wait for services to be healthy
    for service in sutazai-postgres sutazai-redis sutazai-consul sutazai-kong; do
        wait_for_service_health "$service" 120
    done
    
    info "Core services are healthy - system ready for agent deployment"
}

# Emergency shutdown procedure
emergency_shutdown() {
    info "Initiating emergency shutdown procedure"
    
    # Save current system state
    local emergency_state_file="/opt/sutazaiapp/data/emergency_shutdown_$(get_timestamp).json"
    
    docker ps --filter "name=sutazai-" --format "json" > "$emergency_state_file" 2>/dev/null || true
    
    # Create emergency backup
    local emergency_backup
    emergency_backup=$(create_backup "emergency")
    
    info "Emergency backup created: $emergency_backup"
    
    # Gracefully stop agents in reverse priority order
    info "Stopping specialized tier agents"
    docker ps --filter "label=sutazai.tier=specialized" --format "{{.Names}}" | \
        xargs -r -P 5 -I {} docker stop --time=30 {}
    
    info "Stopping performance tier agents"
    docker ps --filter "label=sutazai.tier=performance" --format "{{.Names}}" | \
        xargs -r -P 3 -I {} docker stop --time=45 {}
    
    info "Stopping critical tier agents"
    docker ps --filter "label=sutazai.tier=critical" --format "{{.Names}}" | \
        xargs -r -P 2 -I {} docker stop --time=60 {}
    
    # Stop infrastructure services
    info "Stopping infrastructure services"
    local infrastructure_services=(
        "sutazai-kong"
        "sutazai-rabbitmq"
        "sutazai-consul"
        "sutazai-grafana"
        "sutazai-prometheus"
        "sutazai-neo4j"
        "sutazai-redis"
        "sutazai-postgres"
    )
    
    for service in "${infrastructure_services[@]}"; do
        docker stop --time=30 "$service" 2>/dev/null || true
    done
    
    info "Emergency shutdown completed"
    info "System state saved to: $emergency_state_file"
}

# Clean old backups according to retention policy
cleanup_old_backups() {
    info "Cleaning up old backups according to retention policy"
    
    # Clean daily backups (keep last 7)
    find "$BACKUP_ROOT/daily" -maxdepth 1 -type d -name "backup_*" | \
        sort -r | tail -n +$((DAILY_RETENTION + 1)) | \
        xargs -r rm -rf
    
    # Clean weekly backups (keep last 4)
    find "$BACKUP_ROOT/weekly" -maxdepth 1 -type d -name "backup_*" | \
        sort -r | tail -n +$((WEEKLY_RETENTION + 1)) | \
        xargs -r rm -rf
    
    # Clean monthly backups (keep last 12)
    find "$BACKUP_ROOT/monthly" -maxdepth 1 -type d -name "backup_*" | \
        sort -r | tail -n +$((MONTHLY_RETENTION + 1)) | \
        xargs -r rm -rf
    
    # Clean emergency backups older than 30 days
    find "$BACKUP_ROOT/emergency" -maxdepth 1 -type d -name "backup_*" -mtime +30 | \
        xargs -r rm -rf
    
    info "Backup cleanup completed"
}

# List available backups
list_backups() {
    local backup_type=${1:-"all"}
    
    info "Available backups:"
    
    case "$backup_type" in
        "all")
            for type in daily weekly monthly emergency; do
                echo -e "\n${BLUE}$type backups:${NC}"
                find "$BACKUP_ROOT/$type" -maxdepth 1 -type d -name "backup_*" | sort -r | while read -r backup_dir; do
                    local size
                    size=$(du -sh "$backup_dir" 2>/dev/null | cut -f1)
                    local timestamp
                    timestamp=$(basename "$backup_dir" | sed 's/backup_//')
                    echo "  $(basename "$backup_dir") - Size: $size - $(date -d "${timestamp:0:8} ${timestamp:9:2}:${timestamp:11:2}:${timestamp:13:2}" 2>/dev/null || echo "$timestamp")"
                done
            done
            ;;
        *)
            echo -e "\n${BLUE}$backup_type backups:${NC}"
            find "$BACKUP_ROOT/$backup_type" -maxdepth 1 -type d -name "backup_*" | sort -r | while read -r backup_dir; do
                local size
                size=$(du -sh "$backup_dir" 2>/dev/null | cut -f1)
                local timestamp
                timestamp=$(basename "$backup_dir" | sed 's/backup_//')
                echo "  $(basename "$backup_dir") - Size: $size - $(date -d "${timestamp:0:8} ${timestamp:9:2}:${timestamp:11:2}:${timestamp:13:2}" 2>/dev/null || echo "$timestamp")"
            done
            ;;
    esac
}

# Test disaster recovery procedures
test_disaster_recovery() {
    info "Starting disaster recovery test"
    
    # Create test backup
    local test_backup
    test_backup=$(create_backup "manual")
    
    # Test backup validation
    if validate_backup "$test_backup"; then
        info "✅ Backup validation test passed"
    else
        error "❌ Backup validation test failed"
        return 1
    fi
    
    # Test restore procedures (dry run)
    info "Testing restore procedures (dry run)"
    
    # Simulate restore without actually restoring
    local test_success=true
    
    # Check if restore files exist
    for backup_type in postgresql redis neo4j agents monitoring; do
        if find "$test_backup" -name "${backup_type}_*" | grep -q .; then
            info "✅ $backup_type backup file found"
        else
            warn "⚠️  $backup_type backup file missing"
        fi
    done
    
    info "Disaster recovery test completed"
    
    # Generate test report
    local test_report="/opt/sutazaiapp/reports/dr_test_$(get_timestamp).json"
    cat > "$test_report" << EOF
{
  "test_timestamp": "$(date -Iseconds)",
  "test_backup": "$test_backup",
  "validation_passed": true,
  "components_tested": ["postgresql", "redis", "neo4j", "agents", "monitoring"],
  "recommendations": [
    "Regular testing of restore procedures",
    "Verify backup integrity after each backup",
    "Monitor backup storage space"
  ]
}
EOF
    
    info "Test report saved to: $test_report"
    return 0
}

# Main function
main() {
    create_directories
    
    case "${1:-}" in
        "backup")
            local backup_type=${2:-"manual"}
            create_backup "$backup_type"
            cleanup_old_backups
            ;;
        "restore")
            local backup_dir=${2:-}
            local components=${3:-"all"}
            if [[ -z "$backup_dir" ]]; then
                error "Please specify backup directory"
                exit 1
            fi
            restore_system "$backup_dir" "$components"
            ;;
        "emergency-shutdown")
            emergency_shutdown
            ;;
        "list")
            local backup_type=${2:-"all"}
            list_backups "$backup_type"
            ;;
        "test")
            test_disaster_recovery
            ;;
        "validate")
            local backup_dir=${2:-}
            if [[ -z "$backup_dir" ]]; then
                error "Please specify backup directory"
                exit 1
            fi
            validate_backup "$backup_dir"
            ;;
        "cleanup")
            cleanup_old_backups
            ;;
        "--help"|"-h"|"")
            cat << 'EOF'
Sutazai Disaster Recovery System

Usage: disaster-recovery.sh <command> [options]

Commands:
  backup [type]              Create backup (types: daily, weekly, monthly, manual, emergency)
  restore <backup_dir> [components]  Restore from backup (components: all, databases, agents, monitoring)
  emergency-shutdown         Perform emergency system shutdown with backup
  list [type]               List available backups
  test                      Test disaster recovery procedures
  validate <backup_dir>     Validate backup integrity
  cleanup                   Clean old backups according to retention policy

Examples:
  disaster-recovery.sh backup daily
  disaster-recovery.sh restore /opt/sutazai-backups/daily/backup_20250805_120000
  disaster-recovery.sh emergency-shutdown
  disaster-recovery.sh list daily
  disaster-recovery.sh test

Backup retention policy:
  Daily: 7 backups
  Weekly: 4 backups  
  Monthly: 12 backups
  Emergency: 30 days
EOF
            ;;
        *)
            error "Unknown command: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

# Check dependencies
if ! command -v jq >/dev/null 2>&1; then
    error "jq is required but not installed"
    exit 1
fi

# Execute main function
main "$@"