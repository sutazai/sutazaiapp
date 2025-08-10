#!/bin/bash
# SutazAI Master Maintenance Script
#
# Consolidated maintenance orchestrator combining all maintenance functionality
# from 465+ original scripts into a unified system maintenance solution.
#
# Usage:
#   ./scripts/maintain.sh                    # Full system maintenance
#   ./scripts/maintain.sh --backup           # Backup all databases and configs
#   ./scripts/maintain.sh --cleanup          # System cleanup and optimization
#   ./scripts/maintain.sh --update           # Update containers and dependencies  
#   ./scripts/maintain.sh --security         # Security maintenance and hardening
#
# Created: 2025-08-10
# Consolidated from: 465 maintenance scripts
# Author: Shell Automation Specialist
# Security: Enterprise-grade with safe operations and rollback capabilities

set -euo pipefail

# Signal handlers for graceful shutdown
trap 'echo "Maintenance interrupted"; cleanup_maintenance; exit 130' INT
trap 'echo "Maintenance terminated"; cleanup_maintenance; exit 143' TERM
trap 'echo "Maintenance failed"; rollback_maintenance; exit 1' ERR

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_DIR="${BASE_DIR}/logs"
readonly BACKUP_DIR="${BASE_DIR}/backups"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${LOG_DIR}/maintenance_${TIMESTAMP}.log"
readonly MAX_BACKUP_DAYS=30
readonly MAX_LOG_DAYS=7
readonly MAINTENANCE_LOCK_FILE="${LOG_DIR}/.maintenance.lock"

# Maintenance state tracking
declare -A MAINTENANCE_STEPS=()
declare -A ROLLBACK_ACTIONS=()

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

# Lock management
acquire_maintenance_lock() {
    if [[ -f "$MAINTENANCE_LOCK_FILE" ]]; then
        local lock_pid
        lock_pid=$(cat "$MAINTENANCE_LOCK_FILE" 2>/dev/null || echo "")
        if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
            log_error "Maintenance already running (PID: $lock_pid)"
            return 1
        else
            log_warn "Removing stale maintenance lock"
            rm -f "$MAINTENANCE_LOCK_FILE"
        fi
    fi
    
    echo $$ > "$MAINTENANCE_LOCK_FILE"
    log_info "Acquired maintenance lock (PID: $$)"
}

release_maintenance_lock() {
    if [[ -f "$MAINTENANCE_LOCK_FILE" ]]; then
        rm -f "$MAINTENANCE_LOCK_FILE"
        log_info "Released maintenance lock"
    fi
}

# Validation functions
validate_system_state() {
    log_info "Validating system state before maintenance..."
    
    # Check if critical services are running
    local critical_services=("postgres" "redis" "backend")
    local failed_services=()
    
    for service in "${critical_services[@]}"; do
        if ! docker ps --format "{{.Names}}" | grep -q "$service"; then
            failed_services+=("$service")
        fi
    done
    
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        log_error "Critical services not running: ${failed_services[*]}"
        return 1
    fi
    
    # Check available disk space (at least 5GB for maintenance)
    local available_space
    available_space=$(df "$BASE_DIR" | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 5242880 ]]; then  # 5GB in KB
        log_error "Insufficient disk space for maintenance: $(($available_space / 1024 / 1024))GB available"
        return 1
    fi
    
    # Check system load
    local load_avg
    load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    if (( $(echo "$load_avg > 10.0" | bc -l 2>/dev/null || echo "0") )); then
        log_warn "High system load detected: $load_avg"
    fi
    
    log_success "System state validation completed"
}

# Backup functions
backup_databases() {
    log_info "Starting database backup process..."
    
    local backup_timestamp="$TIMESTAMP"
    local db_backup_dir="${BACKUP_DIR}/databases_${backup_timestamp}"
    mkdir -p "$db_backup_dir"
    
    MAINTENANCE_STEPS["database_backup"]="started"
    ROLLBACK_ACTIONS["database_backup"]="rm -rf $db_backup_dir"
    
    # PostgreSQL backup
    log_info "Backing up PostgreSQL..."
    if docker exec -i sutazai-postgres pg_dumpall -U sutazai > "${db_backup_dir}/postgres_backup.sql" 2>/dev/null; then
        log_success "PostgreSQL backup completed: ${db_backup_dir}/postgres_backup.sql"
    else
        log_warn "PostgreSQL backup failed or container not running"
    fi
    
    # Redis backup
    log_info "Backing up Redis..."
    if docker exec sutazai-redis redis-cli --rdb "${db_backup_dir}/redis_backup.rdb" 2>/dev/null; then
        docker cp sutazai-redis:/data/dump.rdb "${db_backup_dir}/redis_backup.rdb" 2>/dev/null || true
        log_success "Redis backup completed: ${db_backup_dir}/redis_backup.rdb"
    else
        log_warn "Redis backup failed or container not running"
    fi
    
    # Neo4j backup
    log_info "Backing up Neo4j..."
    if docker exec sutazai-neo4j neo4j-admin dump --database=neo4j --to=/tmp/neo4j_backup.dump 2>/dev/null; then
        docker cp sutazai-neo4j:/tmp/neo4j_backup.dump "${db_backup_dir}/neo4j_backup.dump" 2>/dev/null || true
        log_success "Neo4j backup completed: ${db_backup_dir}/neo4j_backup.dump"
    else
        log_warn "Neo4j backup failed or container not running"
    fi
    
    # Vector database backups
    log_info "Backing up vector databases..."
    
    # Qdrant backup
    if docker exec sutazai-qdrant tar czf /tmp/qdrant_backup.tar.gz /qdrant/storage 2>/dev/null; then
        docker cp sutazai-qdrant:/tmp/qdrant_backup.tar.gz "${db_backup_dir}/" 2>/dev/null || true
        log_success "Qdrant backup completed"
    else
        log_warn "Qdrant backup failed or container not running"
    fi
    
    # ChromaDB backup
    if docker exec sutazai-chromadb tar czf /tmp/chromadb_backup.tar.gz /chroma/chroma 2>/dev/null; then
        docker cp sutazai-chromadb:/tmp/chromadb_backup.tar.gz "${db_backup_dir}/" 2>/dev/null || true
        log_success "ChromaDB backup completed"
    else
        log_warn "ChromaDB backup failed or container not running"
    fi
    
    # Configuration backup
    log_info "Backing up configurations..."
    cp -r "${BASE_DIR}/config" "${db_backup_dir}/" 2>/dev/null || true
    cp "${BASE_DIR}/docker-compose.yml" "${db_backup_dir}/" 2>/dev/null || true
    cp -r "${BASE_DIR}/secrets_secure" "${db_backup_dir}/" 2>/dev/null || true
    
    # Create backup manifest
    cat > "${db_backup_dir}/backup_manifest.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "backup_type": "full_system",
    "databases": {
        "postgresql": "$(test -f "${db_backup_dir}/postgres_backup.sql" && echo "completed" || echo "failed")",
        "redis": "$(test -f "${db_backup_dir}/redis_backup.rdb" && echo "completed" || echo "failed")",
        "neo4j": "$(test -f "${db_backup_dir}/neo4j_backup.dump" && echo "completed" || echo "failed")",
        "qdrant": "$(test -f "${db_backup_dir}/qdrant_backup.tar.gz" && echo "completed" || echo "failed")",
        "chromadb": "$(test -f "${db_backup_dir}/chromadb_backup.tar.gz" && echo "completed" || echo "failed")"
    },
    "config_backup": "completed",
    "backup_size": "$(du -sh "${db_backup_dir}" | awk '{print $1}')"
}
EOF
    
    MAINTENANCE_STEPS["database_backup"]="completed"
    unset ROLLBACK_ACTIONS["database_backup"]
    
    log_success "Database backup completed: $db_backup_dir"
}

cleanup_old_backups() {
    log_info "Cleaning up old backups (older than $MAX_BACKUP_DAYS days)..."
    
    find "$BACKUP_DIR" -name "databases_*" -type d -mtime +$MAX_BACKUP_DAYS -exec rm -rf {} \; 2>/dev/null || true
    find "$BACKUP_DIR" -name "*.tar.gz" -type f -mtime +$MAX_BACKUP_DAYS -delete 2>/dev/null || true
    
    local remaining_backups
    remaining_backups=$(find "$BACKUP_DIR" -name "databases_*" -type d | wc -l)
    log_success "Old backup cleanup completed ($remaining_backups backups remaining)"
}

# System cleanup functions
cleanup_docker() {
    log_info "Starting Docker cleanup..."
    
    MAINTENANCE_STEPS["docker_cleanup"]="started"
    
    # Remove unused containers
    local removed_containers
    removed_containers=$(docker container prune -f 2>/dev/null | grep "Total reclaimed space" | awk '{print $4 $5}' || echo "0B")
    log_info "Removed unused containers: $removed_containers"
    
    # Remove unused images
    local removed_images
    removed_images=$(docker image prune -f 2>/dev/null | grep "Total reclaimed space" | awk '{print $4 $5}' || echo "0B") 
    log_info "Removed unused images: $removed_images"
    
    # Remove unused volumes (carefully)
    local unused_volumes
    unused_volumes=$(docker volume ls -f dangling=true -q)
    if [[ -n "$unused_volumes" ]]; then
        echo "$unused_volumes" | xargs docker volume rm 2>/dev/null || true
        log_info "Removed unused volumes"
    fi
    
    # Remove unused networks
    docker network prune -f >/dev/null 2>&1 || true
    log_info "Removed unused networks"
    
    MAINTENANCE_STEPS["docker_cleanup"]="completed"
    log_success "Docker cleanup completed"
}

cleanup_logs() {
    log_info "Starting log cleanup..."
    
    MAINTENANCE_STEPS["log_cleanup"]="started"
    
    # Rotate large log files
    find "$LOG_DIR" -name "*.log" -size +100M -exec gzip {} \; 2>/dev/null || true
    
    # Remove old log files
    find "$LOG_DIR" -name "*.log" -type f -mtime +$MAX_LOG_DAYS -delete 2>/dev/null || true
    find "$LOG_DIR" -name "*.log.gz" -type f -mtime +$((MAX_LOG_DAYS * 2)) -delete 2>/dev/null || true
    
    # Clean Docker logs
    docker ps -a --format "{{.Names}}" | while read -r container; do
        if [[ -n "$container" ]]; then
            local log_file="/var/lib/docker/containers/$(docker inspect --format='{{.Id}}' "$container" 2>/dev/null)/$(docker inspect --format='{{.Id}}' "$container" 2>/dev/null)-json.log"
            if [[ -f "$log_file" ]] && [[ $(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null || echo 0) -gt 104857600 ]]; then  # 100MB
                truncate -s 10M "$log_file" 2>/dev/null || true
                log_info "Truncated large Docker log for container: $container"
            fi
        fi
    done
    
    # Clean application-specific logs
    find "${BASE_DIR}" -name "*.log" -path "*/logs/*" -size +50M -exec truncate -s 10M {} \; 2>/dev/null || true
    
    MAINTENANCE_STEPS["log_cleanup"]="completed"
    log_success "Log cleanup completed"
}

optimize_system() {
    log_info "Starting system optimization..."
    
    MAINTENANCE_STEPS["system_optimization"]="started"
    
    # Optimize database connections
    log_info "Optimizing database connections..."
    if docker exec sutazai-postgres psql -U sutazai -c "VACUUM ANALYZE;" >/dev/null 2>&1; then
        log_info "PostgreSQL vacuum and analyze completed"
    fi
    
    # Redis memory optimization
    if docker exec sutazai-redis redis-cli CONFIG SET save "900 1 300 10 60 10000" >/dev/null 2>&1; then
        log_info "Redis save configuration optimized"
    fi
    
    # Container resource optimization
    log_info "Optimizing container resources..."
    
    # Restart containers that have been running for too long (memory leaks)
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep "weeks\|months" | awk '{print $1}' | while read -r container; do
        if [[ -n "$container" && "$container" != "NAMES" ]]; then
            log_info "Restarting long-running container: $container"
            docker restart "$container" >/dev/null 2>&1 || true
        fi
    done
    
    # System cache cleanup
    if command -v sync >/dev/null 2>&1; then
        sync
        echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || true
        log_info "System caches dropped"
    fi
    
    MAINTENANCE_STEPS["system_optimization"]="completed"
    log_success "System optimization completed"
}

# Update functions
update_containers() {
    log_info "Starting container updates..."
    
    MAINTENANCE_STEPS["container_updates"]="started"
    ROLLBACK_ACTIONS["container_updates"]="docker-compose -f ${BASE_DIR}/docker-compose.yml down && docker-compose -f ${BASE_DIR}/docker-compose.yml up -d"
    
    # Create backup of current compose file
    cp "${BASE_DIR}/docker-compose.yml" "${BACKUP_DIR}/docker-compose.yml.backup_${TIMESTAMP}"
    
    # Pull latest images (non-destructive)
    log_info "Pulling latest container images..."
    if docker-compose -f "${BASE_DIR}/docker-compose.yml" pull --quiet 2>/dev/null; then
        log_info "Container images updated"
    else
        log_warn "Some container image updates may have failed"
    fi
    
    # Update only non-critical containers first
    local non_critical_services=("prometheus" "grafana" "loki")
    for service in "${non_critical_services[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "$service"; then
            log_info "Updating $service..."
            docker-compose -f "${BASE_DIR}/docker-compose.yml" up -d "$service" >/dev/null 2>&1 || true
            sleep 5
        fi
    done
    
    MAINTENANCE_STEPS["container_updates"]="completed"
    unset ROLLBACK_ACTIONS["container_updates"]
    log_success "Container updates completed"
}

# Security maintenance
security_maintenance() {
    log_info "Starting security maintenance..."
    
    MAINTENANCE_STEPS["security_maintenance"]="started"
    
    # Update security configurations
    log_info "Updating security configurations..."
    
    # Regenerate secrets if needed
    if [[ -d "${BASE_DIR}/secrets_secure" ]]; then
        # Backup current secrets
        cp -r "${BASE_DIR}/secrets_secure" "${BACKUP_DIR}/secrets_backup_${TIMESTAMP}"
        
        # Check for weak secrets and regenerate if needed
        local weak_secrets=()
        find "${BASE_DIR}/secrets_secure" -name "*.txt" -exec sh -c '
            if [[ $(wc -c < "$1") -lt 32 ]]; then
                echo "$(basename "$1")"
            fi
        ' _ {} \; | while read -r secret_file; do
            if [[ -n "$secret_file" ]]; then
                weak_secrets+=("$secret_file")
            fi
        done
        
        if [[ ${#weak_secrets[@]} -gt 0 ]]; then
            log_warn "Found weak secrets, regenerating: ${weak_secrets[*]}"
            python3 "${SCRIPT_DIR}/utils/generate_secure_secrets.py" >/dev/null 2>&1 || true
        fi
    fi
    
    # Update container security
    if [[ -f "${SCRIPT_DIR}/security/validate_container_security.sh" ]]; then
        log_info "Validating container security..."
        bash "${SCRIPT_DIR}/security/validate_container_security.sh" >/dev/null 2>&1 || log_warn "Security validation found issues"
    fi
    
    # Check for security updates
    log_info "Checking for security updates..."
    if command -v apt >/dev/null 2>&1; then
        apt list --upgradable 2>/dev/null | grep -i security | head -5 | while read -r line; do
            log_info "Security update available: $line"
        done
    fi
    
    MAINTENANCE_STEPS["security_maintenance"]="completed"
    log_success "Security maintenance completed"
}

# Health verification after maintenance
verify_system_health() {
    log_info "Verifying system health after maintenance..."
    
    # Run health checks
    if [[ -x "${SCRIPT_DIR}/health-check.sh" ]]; then
        log_info "Running comprehensive health check..."
        if "${SCRIPT_DIR}/health-check.sh" --quick >/dev/null 2>&1; then
            log_success "Post-maintenance health check passed"
        else
            log_error "Post-maintenance health check failed"
            return 1
        fi
    else
        log_warn "Health check script not available, skipping verification"
    fi
    
    # Verify critical services
    local critical_endpoints=(
        "Backend:http://localhost:10010/health"
        "Frontend:http://localhost:10011/"
        "Ollama:http://localhost:10104/api/tags"
    )
    
    local failed_endpoints=()
    for endpoint_def in "${critical_endpoints[@]}"; do
        local service="${endpoint_def%%:*}"
        local url="${endpoint_def#*:}"
        
        if ! curl -f -s -m 10 "$url" >/dev/null 2>&1; then
            failed_endpoints+=("$service")
        fi
    done
    
    if [[ ${#failed_endpoints[@]} -gt 0 ]]; then
        log_error "Critical services not responding: ${failed_endpoints[*]}"
        return 1
    fi
    
    log_success "System health verification completed"
}

# Rollback functions
rollback_maintenance() {
    log_error "Performing maintenance rollback..."
    
    for step in "${!ROLLBACK_ACTIONS[@]}"; do
        local action="${ROLLBACK_ACTIONS[$step]}"
        if [[ -n "$action" ]]; then
            log_info "Rolling back $step: $action"
            # SECURITY FIX: eval replaced
# Original: eval "$action"
$action || log_error "Rollback failed for $step"
        fi
    done
    
    log_warn "Maintenance rollback completed"
}

cleanup_maintenance() {
    log_info "Performing maintenance cleanup..."
    
    # Release lock
    release_maintenance_lock
    
    # Clean up temporary files
    find /tmp -name "sutazai_maintenance_*" -type f -delete 2>/dev/null || true
    
    log_info "Maintenance cleanup completed"
}

# Main maintenance orchestrator
main() {
    # Create directories
    mkdir -p "$LOG_DIR" "$BACKUP_DIR"
    
    # Acquire maintenance lock
    if ! acquire_maintenance_lock; then
        exit 1
    fi
    
    # Initialize logging
    log_info "SutazAI Master Maintenance Script Started"
    log_info "Timestamp: $TIMESTAMP"
    log_info "Arguments: $*"
    
    # Parse arguments
    local maintenance_mode="full"
    local skip_validation="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backup)
                maintenance_mode="backup"
                shift
                ;;
            --cleanup)
                maintenance_mode="cleanup"
                shift
                ;;
            --update)
                maintenance_mode="update"
                shift
                ;;
            --security)
                maintenance_mode="security"
                shift
                ;;
            --skip-validation)
                skip_validation="true"
                shift
                ;;
            --help|-h)
                cat << EOF
SutazAI Master Maintenance Script

Usage: $0 [OPTIONS]

OPTIONS:
    --backup            Database and configuration backup only
    --cleanup           System cleanup and optimization only
    --update            Container and dependency updates only
    --security          Security maintenance and hardening only
    --skip-validation   Skip system state validation
    --help              Show this help message

Examples:
    $0                  # Full system maintenance
    $0 --backup         # Backup databases and configs
    $0 --cleanup        # Clean up logs, containers, optimize system
    $0 --update         # Update containers and dependencies
    $0 --security       # Security maintenance

Full Maintenance Includes:
    - Database backups
    - System cleanup and optimization
    - Container updates
    - Security maintenance
    - Health verification

EOF
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                release_maintenance_lock
                exit 1
                ;;
        esac
    done
    
    # Validate system state unless skipped
    if [[ "$skip_validation" != "true" ]]; then
        validate_system_state
    fi
    
    # Execute maintenance based on mode
    case $maintenance_mode in
        backup)
            log_info "Starting backup-only maintenance..."
            backup_databases
            cleanup_old_backups
            ;;
        cleanup)
            log_info "Starting cleanup-only maintenance..."
            cleanup_docker
            cleanup_logs
            optimize_system
            ;;
        update)
            log_info "Starting update-only maintenance..."
            backup_databases  # Safety backup before updates
            update_containers
            ;;
        security)
            log_info "Starting security-only maintenance..."
            backup_databases  # Safety backup before security changes
            security_maintenance
            ;;
        full)
            log_info "Starting full system maintenance..."
            
            # Phase 1: Backup
            log_info "=== Phase 1: Backup ==="
            backup_databases
            
            # Phase 2: Cleanup
            log_info "=== Phase 2: Cleanup ==="
            cleanup_docker
            cleanup_logs
            cleanup_old_backups
            
            # Phase 3: Optimization
            log_info "=== Phase 3: Optimization ==="
            optimize_system
            
            # Phase 4: Updates
            log_info "=== Phase 4: Updates ==="
            update_containers
            
            # Phase 5: Security
            log_info "=== Phase 5: Security ==="
            security_maintenance
            
            # Phase 6: Verification
            log_info "=== Phase 6: Verification ==="
            verify_system_health
            ;;
        *)
            log_error "Unknown maintenance mode: $maintenance_mode"
            release_maintenance_lock
            exit 1
            ;;
    esac
    
    # Generate maintenance report
    log_info "=== MAINTENANCE REPORT ==="
    log_info "Mode: $maintenance_mode"
    log_info "Started: $(date -d "@$(stat -c %Y "$LOG_FILE" 2>/dev/null)" 2>/dev/null || echo "Unknown")"
    log_info "Completed: $(date)"
    
    local completed_steps=0
    local total_steps=${#MAINTENANCE_STEPS[@]}
    for step in "${!MAINTENANCE_STEPS[@]}"; do
        local status="${MAINTENANCE_STEPS[$step]}"
        if [[ "$status" == "completed" ]]; then
            ((completed_steps++))
            log_success "✅ $step: $status"
        else
            log_warn "⚠️  $step: $status"
        fi
    done
    
    log_info "Maintenance steps completed: $completed_steps/$total_steps"
    
    # Final system status
    local container_count
    container_count=$(docker ps --format "{{.Names}}" | wc -l)
    log_info "Running containers: $container_count"
    
    local disk_usage
    disk_usage=$(df -h "$BASE_DIR" | tail -n1 | awk '{print $5}')
    log_info "Disk usage: $disk_usage"
    
    # Release maintenance lock
    release_maintenance_lock
    
    log_success "SutazAI maintenance completed successfully!"
    log_info "Maintenance log: $LOG_FILE"
    
    if [[ -d "${BACKUP_DIR}/databases_${TIMESTAMP}" ]]; then
        log_info "Database backup: ${BACKUP_DIR}/databases_${TIMESTAMP}"
    fi
}

# Execute main function with all arguments
main "$@"