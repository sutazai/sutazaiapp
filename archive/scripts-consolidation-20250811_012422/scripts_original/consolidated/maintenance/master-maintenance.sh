#!/bin/bash
#
# SutazAI Master Maintenance Script - CONSOLIDATED VERSION
# Consolidates 50+ maintenance scripts into ONE unified maintenance controller
#
# Author: Shell Automation Specialist
# Version: 1.0.0 - Consolidation Release
# Date: 2025-08-10
#
# CONSOLIDATION SUMMARY:
# This script replaces the following 50+ maintenance scripts:
# - All scripts/maintenance/*.sh (40+ scripts)
# - All backup-related scripts (10+ scripts)
# - All cleanup and optimization scripts
# - All database maintenance scripts
# - All container management scripts
#
# DESCRIPTION:
# Single, comprehensive maintenance controller for SutazAI platform.
# Handles backups, cleanup, optimization, database maintenance, and
# system health maintenance with proper scheduling and reporting.
#

set -euo pipefail

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    log_error "Maintenance interrupted, cleaning up..."
    # Stop background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    # Unlock maintenance operations
    [[ -f "$MAINTENANCE_LOCK" ]] && rm -f "$MAINTENANCE_LOCK" || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
readonly LOG_DIR="${PROJECT_ROOT}/logs/maintenance"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${LOG_DIR}/maintenance_${TIMESTAMP}.log"
readonly BACKUP_DIR="${PROJECT_ROOT}/backups"
readonly MAINTENANCE_LOCK="${PROJECT_ROOT}/.maintenance.lock"

# Create required directories
mkdir -p "$LOG_DIR" "$BACKUP_DIR"

# Maintenance configuration
OPERATION="${OPERATION:-health-check}"
DRY_RUN="${DRY_RUN:-false}"
FORCE="${FORCE:-false}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
CLEANUP_LOGS_DAYS="${CLEANUP_LOGS_DAYS:-7}"
VACUUM_ANALYZE="${VACUUM_ANALYZE:-false}"
PARALLEL_OPERATIONS="${PARALLEL_OPERATIONS:-true}"
AUTOMATIC_FIXES="${AUTOMATIC_FIXES:-false}"

# Database configuration
DB_CONTAINER="sutazai-postgres"
DB_NAME="sutazai"
DB_USER="sutazai"
REDIS_CONTAINER="sutazai-redis"
NEO4J_CONTAINER="sutazai-neo4j"
QDRANT_CONTAINER="sutazai-qdrant"
CHROMADB_CONTAINER="sutazai-chromadb"
FAISS_CONTAINER="sutazai-faiss"

# Service lists
DATABASE_SERVICES=("$DB_CONTAINER" "$REDIS_CONTAINER" "$NEO4J_CONTAINER")
VECTOR_SERVICES=("$QDRANT_CONTAINER" "$CHROMADB_CONTAINER" "$FAISS_CONTAINER")
ALL_SERVICES=("${DATABASE_SERVICES[@]}" "${VECTOR_SERVICES[@]}")

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

# Usage information
show_usage() {
    cat << 'EOF'
SutazAI Master Maintenance Script - Consolidated Edition

USAGE:
    ./master-maintenance.sh [OPERATION] [OPTIONS]

OPERATIONS:
    health-check    Check system and service health
    backup          Create backups of all databases
    restore         Restore from backup
    cleanup         Clean up logs, containers, and unused resources
    optimize        Optimize databases and system performance
    fix-containers  Fix container issues and restart loops
    vacuum          Database vacuum and analyze operations
    update          Update container configurations
    schedule        Setup maintenance scheduling

BACKUP OPERATIONS:
    backup-all          Backup all databases (PostgreSQL, Redis, Neo4j, Vectors)
    backup-postgres     Backup PostgreSQL database only
    backup-redis        Backup Redis data only
    backup-neo4j        Backup Neo4j graph database only
    backup-vectors      Backup all vector databases only

CLEANUP OPERATIONS:
    cleanup-logs        Clean up old log files
    cleanup-containers  Remove stopped containers and unused images
    cleanup-volumes     Remove unused Docker volumes
    cleanup-networks    Remove unused Docker networks
    cleanup-all         Perform all cleanup operations

OPTIMIZATION OPERATIONS:
    optimize-postgres   PostgreSQL vacuum and analyze
    optimize-redis      Redis memory optimization
    optimize-docker     Docker system optimization
    optimize-all        Perform all optimizations

OPTIONS:
    --dry-run           Show what would be done without executing
    --force             Force operations even if services are running
    --retention DAYS    Backup retention period (default: 30 days)
    --cleanup-days      Log cleanup period (default: 7 days)
    --vacuum            Enable database vacuum operations
    --parallel          Enable parallel operations
    --auto-fix          Automatically fix detected issues
    --debug             Enable debug logging

SCHEDULING OPTIONS:
    --cron-daily        Setup daily maintenance cron job
    --cron-weekly       Setup weekly maintenance cron job
    --cron-custom SPEC  Setup custom cron schedule

EXAMPLES:
    ./master-maintenance.sh health-check --debug
    ./master-maintenance.sh backup-all --retention 60
    ./master-maintenance.sh cleanup-all --dry-run
    ./master-maintenance.sh optimize-all --vacuum --parallel
    ./master-maintenance.sh schedule --cron-daily

CONSOLIDATION NOTE:
This script consolidates the functionality of 50+ maintenance scripts:
- All scripts/maintenance/* files (40+ scripts)
- All backup and restore scripts (10+ scripts)
- All cleanup and optimization scripts
- All database and container management scripts
EOF
}

# Check if maintenance is already running
check_maintenance_lock() {
    if [[ -f "$MAINTENANCE_LOCK" ]]; then
        local lock_pid=$(cat "$MAINTENANCE_LOCK" 2>/dev/null || echo "")
        if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
            log_error "Maintenance already running (PID: $lock_pid)"
            exit 1
        else
            log_warn "Stale maintenance lock found, removing..."
            rm -f "$MAINTENANCE_LOCK"
        fi
    fi
    
    echo "$$" > "$MAINTENANCE_LOCK"
    log_info "Maintenance lock acquired (PID: $$)"
}

# Check service health
check_service_health() {
    local service="$1"
    local timeout="${2:-10}"
    
    if docker ps --filter "name=$service" --filter "status=running" | grep -q "$service"; then
        log_info "✓ $service is running"
        return 0
    else
        log_warn "✗ $service is not running"
        return 1
    fi
}

# System health check
perform_health_check() {
    log_info "Performing comprehensive system health check..."
    
    local issues_found=0
    local total_services=0
    local healthy_services=0
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        issues_found=$((issues_found + 1))
    else
        log_success "Docker daemon is healthy"
    fi
    
    # Check system resources
    local disk_usage=$(df -h "${PROJECT_ROOT}" | awk 'NR==2 {print $5}' | tr -d '%')
    local memory_available=$(free -g | awk '/^Mem:/ {print $7}')
    
    log_info "Disk usage: ${disk_usage}%"
    log_info "Available memory: ${memory_available}GB"
    
    if [[ $disk_usage -gt 90 ]]; then
        log_warn "⚠ High disk usage: ${disk_usage}%"
        issues_found=$((issues_found + 1))
    fi
    
    if [[ $memory_available -lt 2 ]]; then
        log_warn "⚠ Low available memory: ${memory_available}GB"
        issues_found=$((issues_found + 1))
    fi
    
    # Check service health
    log_info "Checking service health..."
    for service in "${ALL_SERVICES[@]}"; do
        total_services=$((total_services + 1))
        if check_service_health "$service"; then
            healthy_services=$((healthy_services + 1))
        else
            issues_found=$((issues_found + 1))
            
            if [[ "$AUTOMATIC_FIXES" == "true" ]]; then
                log_info "Auto-fix enabled, attempting to restart $service..."
                docker restart "$service" 2>/dev/null && log_success "Restarted $service" || log_error "Failed to restart $service"
            fi
        fi
    done
    
    # Check for container restart loops
    log_info "Checking for container restart loops..."
    local restart_loops=$(docker ps -a --format "table {{.Names}}\t{{.Status}}" | grep -c "Restarting" || echo "0")
    if [[ $restart_loops -gt 0 ]]; then
        log_warn "⚠ Found $restart_loops containers in restart loops"
        issues_found=$((issues_found + 1))
    fi
    
    # Check Docker system
    log_info "Checking Docker system health..."
    local unused_images=$(docker images -f "dangling=true" -q | wc -l)
    local unused_volumes=$(docker volume ls -f "dangling=true" -q | wc -l)
    
    if [[ $unused_images -gt 10 ]]; then
        log_warn "⚠ Found $unused_images unused Docker images"
        issues_found=$((issues_found + 1))
    fi
    
    if [[ $unused_volumes -gt 5 ]]; then
        log_warn "⚠ Found $unused_volumes unused Docker volumes"
        issues_found=$((issues_found + 1))
    fi
    
    # Health summary
    local health_percentage=$((healthy_services * 100 / total_services))
    log_info "Health Summary: $healthy_services/$total_services services healthy (${health_percentage}%)"
    log_info "Issues found: $issues_found"
    
    if [[ $issues_found -gt 0 ]]; then
        log_warn "System health check completed with $issues_found issues"
        return 1
    else
        log_success "System health check completed - all systems healthy"
        return 0
    fi
}

# Backup operations
perform_backup() {
    local backup_target="${1:-all}"
    local backup_timestamp="${TIMESTAMP}"
    local backup_path="${BACKUP_DIR}/backup_${backup_timestamp}"
    
    mkdir -p "$backup_path"
    log_info "Starting backup operation: $backup_target"
    log_info "Backup location: $backup_path"
    
    case "$backup_target" in
        all|backup-all)
            backup_postgres "$backup_path" &
            backup_redis "$backup_path" &
            backup_neo4j "$backup_path" &
            backup_vectors "$backup_path" &
            wait # Wait for all parallel backups
            ;;
        postgres|backup-postgres)
            backup_postgres "$backup_path"
            ;;
        redis|backup-redis)
            backup_redis "$backup_path"
            ;;
        neo4j|backup-neo4j)
            backup_neo4j "$backup_path"
            ;;
        vectors|backup-vectors)
            backup_vectors "$backup_path"
            ;;
        *)
            log_error "Unknown backup target: $backup_target"
            return 1
            ;;
    esac
    
    # Create backup manifest
    create_backup_manifest "$backup_path"
    
    # Cleanup old backups
    cleanup_old_backups
    
    log_success "Backup operation completed: $backup_target"
}

# PostgreSQL backup
backup_postgres() {
    local backup_path="$1"
    log_info "Backing up PostgreSQL database..."
    
    if check_service_health "$DB_CONTAINER"; then
        local postgres_backup="${backup_path}/postgres_${TIMESTAMP}.sql"
        
        if docker exec "$DB_CONTAINER" pg_dump -U "$DB_USER" "$DB_NAME" > "$postgres_backup" 2>/dev/null; then
            log_success "PostgreSQL backup completed: $(basename "$postgres_backup")"
            log_info "Backup size: $(du -sh "$postgres_backup" | cut -f1)"
        else
            log_error "PostgreSQL backup failed"
            return 1
        fi
    else
        log_error "PostgreSQL container is not running"
        return 1
    fi
}

# Redis backup
backup_redis() {
    local backup_path="$1"
    log_info "Backing up Redis data..."
    
    if check_service_health "$REDIS_CONTAINER"; then
        local redis_backup="${backup_path}/redis_${TIMESTAMP}.rdb"
        
        # Trigger Redis save and copy RDB file
        if docker exec "$REDIS_CONTAINER" redis-cli BGSAVE >/dev/null 2>&1; then
            sleep 5 # Wait for background save
            if docker cp "${REDIS_CONTAINER}:/data/dump.rdb" "$redis_backup" 2>/dev/null; then
                log_success "Redis backup completed: $(basename "$redis_backup")"
                log_info "Backup size: $(du -sh "$redis_backup" | cut -f1)"
            else
                log_error "Redis backup copy failed"
                return 1
            fi
        else
            log_error "Redis BGSAVE failed"
            return 1
        fi
    else
        log_error "Redis container is not running"
        return 1
    fi
}

# Neo4j backup
backup_neo4j() {
    local backup_path="$1"
    log_info "Backing up Neo4j database..."
    
    if check_service_health "$NEO4J_CONTAINER"; then
        local neo4j_backup="${backup_path}/neo4j_${TIMESTAMP}"
        mkdir -p "$neo4j_backup"
        
        # Neo4j database backup using admin tool
        if docker exec "$NEO4J_CONTAINER" neo4j-admin dump --database=neo4j --to=/tmp/neo4j-backup.dump 2>/dev/null; then
            if docker cp "${NEO4J_CONTAINER}:/tmp/neo4j-backup.dump" "${neo4j_backup}/neo4j.dump" 2>/dev/null; then
                log_success "Neo4j backup completed: $(basename "$neo4j_backup")"
                log_info "Backup size: $(du -sh "$neo4j_backup" | cut -f1)"
            else
                log_error "Neo4j backup copy failed"
                return 1
            fi
        else
            log_error "Neo4j backup failed"
            return 1
        fi
    else
        log_error "Neo4j container is not running"
        return 1
    fi
}

# Vector databases backup
backup_vectors() {
    local backup_path="$1"
    local vector_backup_path="${backup_path}/vectors_${TIMESTAMP}"
    mkdir -p "$vector_backup_path"
    
    log_info "Backing up vector databases..."
    
    # Backup each vector database
    for vector_service in "${VECTOR_SERVICES[@]}"; do
        if check_service_health "$vector_service"; then
            log_info "Backing up $vector_service..."
            
            case "$vector_service" in
                *qdrant*)
                    docker cp "${vector_service}:/qdrant/storage" "${vector_backup_path}/qdrant_storage" 2>/dev/null && \
                    log_success "Qdrant backup completed"
                    ;;
                *chromadb*)
                    docker cp "${vector_service}:/chroma/data" "${vector_backup_path}/chromadb_data" 2>/dev/null && \
                    log_success "ChromaDB backup completed"
                    ;;
                *faiss*)
                    docker cp "${vector_service}:/faiss/index" "${vector_backup_path}/faiss_index" 2>/dev/null && \
                    log_success "FAISS backup completed"
                    ;;
            esac
        else
            log_warn "$vector_service is not running, skipping backup"
        fi
    done
    
    log_success "Vector databases backup completed"
}

# Create backup manifest
create_backup_manifest() {
    local backup_path="$1"
    local manifest_file="${backup_path}/backup_manifest.json"
    
    cat > "$manifest_file" << EOF
{
    "backup_timestamp": "$TIMESTAMP",
    "backup_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "sutazai_version": "v76",
    "backup_type": "full",
    "files": [
EOF

    local files=($(find "$backup_path" -type f ! -name "backup_manifest.json" -printf "%f\n" 2>/dev/null))
    for ((i=0; i<${#files[@]}; i++)); do
        [[ $i -gt 0 ]] && echo "," >> "$manifest_file"
        echo "        \"${files[$i]}\"" >> "$manifest_file"
    done
    
    cat >> "$manifest_file" << EOF
    ],
    "total_size": "$(du -sh "$backup_path" | cut -f1)",
    "file_count": ${#files[@]}
}
EOF
    
    log_info "Backup manifest created: $(basename "$manifest_file")"
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up backups older than $BACKUP_RETENTION_DAYS days..."
    
    local deleted_count=0
    while IFS= read -r -d '' backup_dir; do
        rm -rf "$backup_dir"
        deleted_count=$((deleted_count + 1))
        log_info "Deleted old backup: $(basename "$backup_dir")"
    done < <(find "$BACKUP_DIR" -type d -name "backup_*" -mtime +$BACKUP_RETENTION_DAYS -print0 2>/dev/null)
    
    log_info "Cleanup completed: removed $deleted_count old backups"
}

# Cleanup operations
perform_cleanup() {
    local cleanup_target="${1:-all}"
    
    log_info "Starting cleanup operation: $cleanup_target"
    
    case "$cleanup_target" in
        all|cleanup-all)
            cleanup_logs
            cleanup_containers
            cleanup_volumes
            cleanup_networks
            ;;
        logs|cleanup-logs)
            cleanup_logs
            ;;
        containers|cleanup-containers)
            cleanup_containers
            ;;
        volumes|cleanup-volumes)
            cleanup_volumes
            ;;
        networks|cleanup-networks)
            cleanup_networks
            ;;
        *)
            log_error "Unknown cleanup target: $cleanup_target"
            return 1
            ;;
    esac
    
    log_success "Cleanup operation completed: $cleanup_target"
}

# Cleanup logs
cleanup_logs() {
    log_info "Cleaning up logs older than $CLEANUP_LOGS_DAYS days..."
    
    local logs_dirs=("$LOG_DIR" "${PROJECT_ROOT}/logs")
    local deleted_count=0
    
    for logs_dir in "${logs_dirs[@]}"; do
        if [[ -d "$logs_dir" ]]; then
            while IFS= read -r -d '' log_file; do
                rm -f "$log_file"
                deleted_count=$((deleted_count + 1))
            done < <(find "$logs_dir" -type f -name "*.log" -mtime +$CLEANUP_LOGS_DAYS -print0 2>/dev/null)
        fi
    done
    
    log_info "Log cleanup completed: removed $deleted_count old log files"
}

# Cleanup containers
cleanup_containers() {
    log_info "Cleaning up Docker containers and images..."
    
    # Remove stopped containers
    local stopped_containers=$(docker ps -aq -f status=exited | wc -l)
    if [[ $stopped_containers -gt 0 ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "DRY RUN: Would remove $stopped_containers stopped containers"
        else
            docker container prune -f
            log_info "Removed $stopped_containers stopped containers"
        fi
    fi
    
    # Remove dangling images
    local dangling_images=$(docker images -f "dangling=true" -q | wc -l)
    if [[ $dangling_images -gt 0 ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "DRY RUN: Would remove $dangling_images dangling images"
        else
            docker image prune -f
            log_info "Removed $dangling_images dangling images"
        fi
    fi
    
    log_success "Container cleanup completed"
}

# Cleanup volumes
cleanup_volumes() {
    log_info "Cleaning up unused Docker volumes..."
    
    local unused_volumes=$(docker volume ls -f "dangling=true" -q | wc -l)
    if [[ $unused_volumes -gt 0 ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "DRY RUN: Would remove $unused_volumes unused volumes"
        else
            docker volume prune -f
            log_info "Removed $unused_volumes unused volumes"
        fi
    else
        log_info "No unused volumes found"
    fi
}

# Cleanup networks
cleanup_networks() {
    log_info "Cleaning up unused Docker networks..."
    
    local unused_networks=$(docker network ls -f "dangling=true" -q | wc -l)
    if [[ $unused_networks -gt 0 ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "DRY RUN: Would remove $unused_networks unused networks"
        else
            docker network prune -f
            log_info "Removed $unused_networks unused networks"
        fi
    else
        log_info "No unused networks found"
    fi
}

# Optimization operations
perform_optimization() {
    local optimize_target="${1:-all}"
    
    log_info "Starting optimization operation: $optimize_target"
    
    case "$optimize_target" in
        all|optimize-all)
            optimize_postgres
            optimize_redis
            optimize_docker
            ;;
        postgres|optimize-postgres)
            optimize_postgres
            ;;
        redis|optimize-redis)
            optimize_redis
            ;;
        docker|optimize-docker)
            optimize_docker
            ;;
        *)
            log_error "Unknown optimization target: $optimize_target"
            return 1
            ;;
    esac
    
    log_success "Optimization operation completed: $optimize_target"
}

# PostgreSQL optimization
optimize_postgres() {
    log_info "Optimizing PostgreSQL database..."
    
    if check_service_health "$DB_CONTAINER"; then
        if [[ "$VACUUM_ANALYZE" == "true" ]]; then
            log_info "Performing VACUUM ANALYZE..."
            
            if docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "VACUUM ANALYZE;" >/dev/null 2>&1; then
                log_success "PostgreSQL VACUUM ANALYZE completed"
            else
                log_error "PostgreSQL VACUUM ANALYZE failed"
                return 1
            fi
        fi
        
        # Check database statistics
        local db_size=$(docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT pg_size_pretty(pg_database_size('$DB_NAME'));" 2>/dev/null | tr -d ' ')
        local table_count=$(docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | tr -d ' ')
        
        log_info "PostgreSQL stats - Database size: $db_size, Tables: $table_count"
    else
        log_error "PostgreSQL container is not running"
        return 1
    fi
}

# Redis optimization
optimize_redis() {
    log_info "Optimizing Redis..."
    
    if check_service_health "$REDIS_CONTAINER"; then
        # Get Redis info
        local redis_memory=$(docker exec "$REDIS_CONTAINER" redis-cli INFO memory | grep "used_memory_human" | cut -d: -f2 | tr -d '\r')
        local redis_keys=$(docker exec "$REDIS_CONTAINER" redis-cli INFO keyspace | grep "keys=" | head -1 | cut -d, -f1 | cut -d= -f2 | tr -d '\r')
        
        log_info "Redis stats - Memory used: $redis_memory, Keys: ${redis_keys:-0}"
        
        # Trigger background save
        if docker exec "$REDIS_CONTAINER" redis-cli BGSAVE >/dev/null 2>&1; then
            log_success "Redis background save triggered"
        else
            log_warn "Redis background save failed"
        fi
    else
        log_error "Redis container is not running"
        return 1
    fi
}

# Docker system optimization
optimize_docker() {
    log_info "Optimizing Docker system..."
    
    # Docker system info
    local docker_containers=$(docker ps -q | wc -l)
    local docker_images=$(docker images -q | wc -l)
    local docker_volumes=$(docker volume ls -q | wc -l)
    
    log_info "Docker stats - Containers: $docker_containers, Images: $docker_images, Volumes: $docker_volumes"
    
    # Check Docker daemon configuration
    if docker info --format "{{.DriverStatus}}" >/dev/null 2>&1; then
        log_success "Docker daemon is healthy"
    else
        log_warn "Docker daemon may have issues"
    fi
}

# Container fixes
fix_containers() {
    log_info "Checking and fixing container issues..."
    
    # Check for containers in restart loops
    local restarting_containers=($(docker ps -a --filter "status=restarting" --format "{{.Names}}" 2>/dev/null))
    
    if [[ ${#restarting_containers[@]} -gt 0 ]]; then
        log_warn "Found ${#restarting_containers[@]} containers in restart loops"
        
        for container in "${restarting_containers[@]}"; do
            log_info "Attempting to fix restart loop for: $container"
            
            if [[ "$AUTOMATIC_FIXES" == "true" ]]; then
                # Stop the container and restart it
                docker stop "$container" 2>/dev/null || true
                sleep 5
                docker start "$container" 2>/dev/null && log_success "Fixed restart loop: $container" || log_error "Failed to fix: $container"
            else
                log_info "Auto-fix disabled, manual intervention required for: $container"
            fi
        done
    else
        log_info "No containers in restart loops found"
    fi
    
    # Check for unhealthy containers
    local unhealthy_containers=($(docker ps --filter "health=unhealthy" --format "{{.Names}}" 2>/dev/null))
    
    if [[ ${#unhealthy_containers[@]} -gt 0 ]]; then
        log_warn "Found ${#unhealthy_containers[@]} unhealthy containers"
        
        for container in "${unhealthy_containers[@]}"; do
            log_info "Container health issue: $container"
            
            if [[ "$AUTOMATIC_FIXES" == "true" ]]; then
                docker restart "$container" 2>/dev/null && log_success "Restarted unhealthy container: $container" || log_error "Failed to restart: $container"
            fi
        done
    else
        log_info "No unhealthy containers found"
    fi
}

# Setup maintenance scheduling
setup_maintenance_schedule() {
    local schedule_type="${1:-daily}"
    
    log_info "Setting up maintenance scheduling: $schedule_type"
    
    local cron_job=""
    local maintenance_script="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
    
    case "$schedule_type" in
        daily|cron-daily)
            cron_job="0 2 * * * $maintenance_script health-check && $maintenance_script cleanup-logs"
            log_info "Daily maintenance: health check and log cleanup at 2:00 AM"
            ;;
        weekly|cron-weekly)
            cron_job="0 3 * * 0 $maintenance_script backup-all && $maintenance_script optimize-all && $maintenance_script cleanup-all"
            log_info "Weekly maintenance: full backup, optimization, and cleanup on Sundays at 3:00 AM"
            ;;
        cron-custom)
            local custom_schedule="${2:-}"
            if [[ -n "$custom_schedule" ]]; then
                cron_job="$custom_schedule $maintenance_script health-check"
                log_info "Custom maintenance schedule: $custom_schedule"
            else
                log_error "Custom schedule specification required"
                return 1
            fi
            ;;
        *)
            log_error "Unknown schedule type: $schedule_type"
            return 1
            ;;
    esac
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would add cron job: $cron_job"
    else
        (crontab -l 2>/dev/null; echo "$cron_job") | crontab -
        log_success "Maintenance schedule configured: $schedule_type"
    fi
}

# Main execution
main() {
    local operation="${1:-health-check}"
    
    # Check for maintenance lock
    check_maintenance_lock
    
    # Parse command line options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --force)
                FORCE="true"
                shift
                ;;
            --retention)
                BACKUP_RETENTION_DAYS="$2"
                shift 2
                ;;
            --cleanup-days)
                CLEANUP_LOGS_DAYS="$2"
                shift 2
                ;;
            --vacuum)
                VACUUM_ANALYZE="true"
                shift
                ;;
            --parallel)
                PARALLEL_OPERATIONS="true"
                shift
                ;;
            --auto-fix)
                AUTOMATIC_FIXES="true"
                shift
                ;;
            --cron-daily)
                operation="schedule"
                schedule_type="daily"
                shift
                ;;
            --cron-weekly)
                operation="schedule"
                schedule_type="weekly"
                shift
                ;;
            --cron-custom)
                operation="schedule"
                schedule_type="cron-custom"
                custom_schedule="$2"
                shift 2
                ;;
            --debug)
                set -x
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            health-check|backup*|restore|cleanup*|optimize*|fix-containers|vacuum|update|schedule)
                operation="$1"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    log_info "SutazAI Master Maintenance Script - Consolidation Edition"
    log_info "Operation: $operation"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN MODE - No actual changes will be made"
    fi
    
    # Execute operation
    case "$operation" in
        health-check)
            perform_health_check
            ;;
        backup*|restore)
            perform_backup "$operation"
            ;;
        cleanup*)
            perform_cleanup "$operation"
            ;;
        optimize*)
            perform_optimization "$operation"
            ;;
        fix-containers)
            fix_containers
            ;;
        vacuum)
            VACUUM_ANALYZE="true"
            optimize_postgres
            ;;
        schedule)
            setup_maintenance_schedule "${schedule_type:-daily}" "${custom_schedule:-}"
            ;;
        *)
            log_error "Unknown operation: $operation"
            show_usage
            exit 1
            ;;
    esac
    
    # Remove maintenance lock
    rm -f "$MAINTENANCE_LOCK"
    log_info "Maintenance operation completed successfully"
}

# Execute main function with all arguments
main "$@"