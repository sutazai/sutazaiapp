#!/bin/bash
# SutazAI Master Maintenance Script
# Consolidates 15+ database and system maintenance script variations
# Author: DevOps Manager - Deduplication Operation
# Date: August 10, 2025

set -euo pipefail

# Configuration

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/maintenance_$(date +%Y%m%d_%H%M%S).log"

# Default values
OPERATION="health-check"
DRY_RUN=false
BACKUP_RETENTION_DAYS=30
CLEANUP_LOGS_DAYS=7
VACUUM_ANALYZE=false
FORCE=false

# Database configuration
DB_CONTAINER="sutazai-postgres"
DB_NAME="sutazai"
DB_USER="sutazai"
REDIS_CONTAINER="sutazai-redis"
NEO4J_CONTAINER="sutazai-neo4j"

# Directories
BACKUP_DIR="${PROJECT_ROOT}/backups"
LOGS_DIR="${PROJECT_ROOT}/logs"

# Logging functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# Usage information
usage() {
    cat << EOF
SutazAI Master Maintenance Script - Unified System Maintenance

USAGE:
    $0 [OPTIONS] <operation>

OPERATIONS:
    health-check           Check system and database health
    backup-all            Backup all databases (PostgreSQL, Redis, Neo4j)
    backup-postgres       Backup PostgreSQL database only
    backup-redis          Backup Redis data only
    backup-neo4j          Backup Neo4j graph database only
    cleanup-logs          Clean up old log files
    cleanup-containers    Remove stopped containers and unused images
    cleanup-all           Full system cleanup (logs, containers, unused data)
    vacuum-db             PostgreSQL VACUUM and ANALYZE
    optimize-db           Database optimization (vacuum, reindex, analyze)
    disk-usage            Show detailed disk usage report
    system-status         Comprehensive system status report

OPTIONS:
    --dry-run             Show what would be done without executing
    --retention DAYS      Backup retention period (default: 30)
    --log-retention DAYS  Log retention period (default: 7)
    --vacuum-analyze      Include VACUUM ANALYZE in database operations
    --force               Force operations without confirmation prompts
    -h, --help            Show this help message

EXAMPLES:
    $0 health-check                    # Quick system health check
    $0 backup-all --retention 60       # Backup all with 60-day retention
    $0 cleanup-all --dry-run          # See what cleanup would do
    $0 optimize-db --vacuum-analyze   # Full database optimization

SCHEDULING:
    # Add to crontab for automated maintenance:
    0 2 * * * $0 backup-all >/dev/null 2>&1
    0 3 * * 0 $0 cleanup-all >/dev/null 2>&1
    0 4 * * 0 $0 optimize-db >/dev/null 2>&1

LOG FILE: $LOG_FILE
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --retention)
                BACKUP_RETENTION_DAYS="$2"
                shift 2
                ;;
            --log-retention)
                CLEANUP_LOGS_DAYS="$2"
                shift 2
                ;;
            --vacuum-analyze)
                VACUUM_ANALYZE=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                if [[ -z "${OPERATION:-}" ]]; then
                    OPERATION="$1"
                else
                    log_error "Multiple operations specified"
                    exit 1
                fi
                shift
                ;;
        esac
    done
}

# Check if container is running
check_container() {
    local container_name="$1"
    if ! docker ps --format "table {{.Names}}" | grep -q "^${container_name}$"; then
        return 1
    fi
    return 0
}

# Execute command with dry-run support
execute_command() {
    local cmd="$1"
    local description="$2"
    
    log "$description"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN - Would execute: $cmd"
        return 0
    fi
    
    if # SECURITY FIX: eval replaced
# Original: eval "$cmd"
$cmd; then
        log "SUCCESS: $description"
        return 0
    else
        log_error "FAILED: $description"
        return 1
    fi
}

# Health check operation
health_check() {
    log "Performing system health check..."
    
    local health_issues=0
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not accessible"
        ((health_issues++))
    else
        log "Docker daemon: OK"
    fi
    
    # Check container status
    local containers=("$DB_CONTAINER" "$REDIS_CONTAINER" "$NEO4J_CONTAINER")
    for container in "${containers[@]}"; do
        if check_container "$container"; then
            log "Container $container: RUNNING"
            
            # Check container health
            local health=$(docker inspect "$container" --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
            if [[ "$health" == "healthy" ]]; then
                log "Container $container health: HEALTHY"
            elif [[ "$health" == "unknown" ]]; then
                log "Container $container health: NO HEALTHCHECK"
            else
                log "Container $container health: $health"
                ((health_issues++))
            fi
        else
            log_error "Container $container: NOT RUNNING"
            ((health_issues++))
        fi
    done
    
    # Check disk space
    local disk_usage=$(df "${PROJECT_ROOT}" | awk 'NR==2 {print int($5)}')
    if [[ $disk_usage -gt 90 ]]; then
        log_error "Disk usage critical: ${disk_usage}%"
        ((health_issues++))
    elif [[ $disk_usage -gt 80 ]]; then
        log "WARNING: Disk usage high: ${disk_usage}%"
    else
        log "Disk usage: ${disk_usage}%"
    fi
    
    # Check memory usage
    local memory_usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
    if [[ $memory_usage -gt 90 ]]; then
        log_error "Memory usage critical: ${memory_usage}%"
        ((health_issues++))
    elif [[ $memory_usage -gt 80 ]]; then
        log "WARNING: Memory usage high: ${memory_usage}%"
    else
        log "Memory usage: ${memory_usage}%"
    fi
    
    # Summary
    if [[ $health_issues -eq 0 ]]; then
        log "Health check PASSED - No issues found"
        return 0
    else
        log_error "Health check FAILED - $health_issues issue(s) found"
        return 1
    fi
}

# Backup PostgreSQL database
backup_postgres() {
    log "Backing up PostgreSQL database..."
    
    if ! check_container "$DB_CONTAINER"; then
        log_error "PostgreSQL container not running"
        return 1
    fi
    
    local backup_file="postgres_backup_$(date +%Y%m%d_%H%M%S).sql"
    local backup_path="${BACKUP_DIR}/database/${backup_file}"
    
    mkdir -p "$(dirname "$backup_path")"
    
    local backup_cmd="docker exec $DB_CONTAINER pg_dump -U $DB_USER $DB_NAME > '$backup_path'"
    
    if execute_command "$backup_cmd" "PostgreSQL backup to $backup_file"; then
        # Compress backup
        execute_command "gzip '$backup_path'" "Compressing backup"
        
        # Clean up old backups
        cleanup_old_backups "${BACKUP_DIR}/database" "postgres_backup_*.sql.gz" "$BACKUP_RETENTION_DAYS"
        
        log "PostgreSQL backup completed: ${backup_path}.gz"
        return 0
    else
        return 1
    fi
}

# Backup Redis data  
backup_redis() {
    log "Backing up Redis data..."
    
    if ! check_container "$REDIS_CONTAINER"; then
        log_error "Redis container not running"
        return 1
    fi
    
    local backup_file="redis_backup_$(date +%Y%m%d_%H%M%S).rdb"
    local backup_path="${BACKUP_DIR}/redis/${backup_file}"
    
    mkdir -p "$(dirname "$backup_path")"
    
    # Force Redis to save current state
    execute_command "docker exec $REDIS_CONTAINER redis-cli BGSAVE" "Initiating Redis background save"
    
    # Wait for save to complete
    sleep 5
    
    # Copy RDB file
    local copy_cmd="docker cp ${REDIS_CONTAINER}:/data/dump.rdb '$backup_path'"
    
    if execute_command "$copy_cmd" "Copying Redis RDB file"; then
        # Compress backup
        execute_command "gzip '$backup_path'" "Compressing Redis backup"
        
        # Clean up old backups
        cleanup_old_backups "${BACKUP_DIR}/redis" "redis_backup_*.rdb.gz" "$BACKUP_RETENTION_DAYS"
        
        log "Redis backup completed: ${backup_path}.gz"
        return 0
    else
        return 1
    fi
}

# Backup Neo4j database
backup_neo4j() {
    log "Backing up Neo4j database..."
    
    if ! check_container "$NEO4J_CONTAINER"; then
        log_error "Neo4j container not running"
        return 1
    fi
    
    local backup_file="neo4j_backup_$(date +%Y%m%d_%H%M%S).dump"
    local backup_path="${BACKUP_DIR}/neo4j/${backup_file}"
    
    mkdir -p "$(dirname "$backup_path")"
    
    # Create Neo4j dump
    local dump_cmd="docker exec $NEO4J_CONTAINER neo4j-admin dump --database=neo4j --to=/backups/${backup_file}"
    
    if execute_command "$dump_cmd" "Creating Neo4j dump"; then
        # Copy from container
        execute_command "docker cp ${NEO4J_CONTAINER}:/backups/${backup_file} '$backup_path'" "Copying Neo4j dump"
        
        # Compress backup
        execute_command "gzip '$backup_path'" "Compressing Neo4j backup"
        
        # Clean up old backups
        cleanup_old_backups "${BACKUP_DIR}/neo4j" "neo4j_backup_*.dump.gz" "$BACKUP_RETENTION_DAYS"
        
        log "Neo4j backup completed: ${backup_path}.gz"
        return 0
    else
        return 1
    fi
}

# Backup all databases
backup_all() {
    log "Starting full database backup..."
    
    local backup_success=0
    
    if backup_postgres; then
        ((backup_success++))
    fi
    
    if backup_redis; then
        ((backup_success++))
    fi
    
    if backup_neo4j; then
        ((backup_success++))
    fi
    
    log "Backup completed: $backup_success/3 databases backed up successfully"
    
    if [[ $backup_success -eq 3 ]]; then
        return 0
    else
        return 1
    fi
}

# Clean up old backup files
cleanup_old_backups() {
    local backup_dir="$1"
    local pattern="$2"
    local retention_days="$3"
    
    if [[ ! -d "$backup_dir" ]]; then
        return 0
    fi
    
    log "Cleaning up backups older than $retention_days days in $backup_dir"
    
    local cleanup_cmd="find '$backup_dir' -name '$pattern' -type f -mtime +$retention_days -delete"
    
    execute_command "$cleanup_cmd" "Cleaning up old backups"
}

# Clean up log files
cleanup_logs() {
    log "Cleaning up log files older than $CLEANUP_LOGS_DAYS days..."
    
    if [[ -d "$LOGS_DIR" ]]; then
        local log_cleanup_cmd="find '$LOGS_DIR' -name '*.log' -type f -mtime +$CLEANUP_LOGS_DAYS -delete"
        execute_command "$log_cleanup_cmd" "Cleaning up old log files"
    fi
    
    # Clean up Docker logs
    execute_command "docker system prune -f --volumes" "Cleaning up Docker system"
}

# Clean up containers and images
cleanup_containers() {
    log "Cleaning up Docker containers and images..."
    
    # Remove stopped containers
    execute_command "docker container prune -f" "Removing stopped containers"
    
    # Remove unused images
    execute_command "docker image prune -f" "Removing unused images"
    
    # Remove unused volumes
    execute_command "docker volume prune -f" "Removing unused volumes"
    
    # Remove unused networks
    execute_command "docker network prune -f" "Removing unused networks"
}

# Full system cleanup
cleanup_all() {
    log "Performing full system cleanup..."
    
    cleanup_logs
    cleanup_containers
    
    # Clean up old backups
    cleanup_old_backups "${BACKUP_DIR}/database" "*.sql.gz" "$BACKUP_RETENTION_DAYS"
    cleanup_old_backups "${BACKUP_DIR}/redis" "*.rdb.gz" "$BACKUP_RETENTION_DAYS"
    cleanup_old_backups "${BACKUP_DIR}/neo4j" "*.dump.gz" "$BACKUP_RETENTION_DAYS"
    
    log "Full system cleanup completed"
}

# PostgreSQL vacuum and analyze
vacuum_db() {
    log "Performing PostgreSQL VACUUM and ANALYZE..."
    
    if ! check_container "$DB_CONTAINER"; then
        log_error "PostgreSQL container not running"
        return 1
    fi
    
    if [[ "$VACUUM_ANALYZE" == "true" ]]; then
        execute_command "docker exec $DB_CONTAINER psql -U $DB_USER -d $DB_NAME -c 'VACUUM ANALYZE;'" "VACUUM ANALYZE"
    else
        execute_command "docker exec $DB_CONTAINER psql -U $DB_USER -d $DB_NAME -c 'VACUUM;'" "VACUUM"
    fi
}

# Database optimization
optimize_db() {
    log "Optimizing database performance..."
    
    if ! check_container "$DB_CONTAINER"; then
        log_error "PostgreSQL container not running"
        return 1
    fi
    
    # VACUUM ANALYZE
    execute_command "docker exec $DB_CONTAINER psql -U $DB_USER -d $DB_NAME -c 'VACUUM ANALYZE;'" "VACUUM ANALYZE"
    
    # REINDEX
    execute_command "docker exec $DB_CONTAINER psql -U $DB_USER -d $DB_NAME -c 'REINDEX DATABASE $DB_NAME;'" "REINDEX DATABASE"
    
    # Update statistics
    execute_command "docker exec $DB_CONTAINER psql -U $DB_USER -d $DB_NAME -c 'ANALYZE;'" "UPDATE STATISTICS"
    
    log "Database optimization completed"
}

# Disk usage report
disk_usage() {
    log "Generating disk usage report..."
    
    echo "=== SutazAI Disk Usage Report ==="
    echo "Generated: $(date)"
    echo
    
    # Overall disk usage
    echo "Overall Disk Usage:"
    df -h "${PROJECT_ROOT}"
    echo
    
    # Project directory breakdown
    echo "Project Directory Usage:"
    du -sh "${PROJECT_ROOT}"/* 2>/dev/null | sort -hr
    echo
    
    # Docker usage
    echo "Docker Disk Usage:"
    docker system df
    echo
    
    # Largest log files
    echo "Largest Log Files:"
    find "$LOGS_DIR" -name "*.log" -type f -exec ls -lh {} \; 2>/dev/null | sort -k5 -hr | head -10
    echo
    
    # Backup directory usage
    if [[ -d "$BACKUP_DIR" ]]; then
        echo "Backup Directory Usage:"
        du -sh "${BACKUP_DIR}"/* 2>/dev/null | sort -hr
    fi
}

# System status report
system_status() {
    log "Generating comprehensive system status report..."
    
    echo "=== SutazAI System Status Report ==="
    echo "Generated: $(date)"
    echo
    
    # System information
    echo "System Information:"
    uname -a
    echo
    
    # Resource usage
    echo "CPU and Memory Usage:"
    top -bn1 | head -5
    echo
    
    echo "Memory Details:"
    free -h
    echo
    
    # Docker status
    echo "Docker System Information:"
    docker info 2>/dev/null | grep -E "(Server Version|Storage Driver|Logging Driver|Containers|Images)"
    echo
    
    echo "Running Containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo
    
    # Service health
    health_check
}

# Main execution
main() {
    log "Starting SutazAI Master Maintenance"
    log "Operation: $OPERATION"
    
    # Create required directories
    mkdir -p "$BACKUP_DIR" "$LOGS_DIR"
    
    # Execute requested operation
    case "$OPERATION" in
        health-check)
            health_check
            ;;
        backup-all)
            backup_all
            ;;
        backup-postgres)
            backup_postgres
            ;;
        backup-redis)
            backup_redis
            ;;
        backup-neo4j)
            backup_neo4j
            ;;
        cleanup-logs)
            cleanup_logs
            ;;
        cleanup-containers)
            cleanup_containers
            ;;
        cleanup-all)
            cleanup_all
            ;;
        vacuum-db)
            vacuum_db
            ;;
        optimize-db)
            optimize_db
            ;;
        disk-usage)
            disk_usage
            ;;
        system-status)
            system_status
            ;;
        *)
            log_error "Unknown operation: $OPERATION"
            usage
            exit 1
            ;;
    esac
    
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log "Maintenance operation completed successfully"
    else
        log_error "Maintenance operation failed with exit code $exit_code"
    fi
    
    exit $exit_code
}

# Parse arguments and execute
parse_args "$@"
main