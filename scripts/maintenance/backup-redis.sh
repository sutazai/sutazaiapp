#!/bin/bash

# Redis Backup Script for SutazAI System
# Implements both RDB and AOF backup strategies
# Author: DevOps Manager
# Date: 2025-08-09

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

REDIS_CONTAINER="sutazai-redis"
BACKUP_DIR="/opt/sutazaiapp/backups/redis"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Logging
LOG_FILE="/opt/sutazaiapp/logs/backup_redis_${TIMESTAMP}.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

# Pre-flight checks
preflight_checks() {
    log "Starting Redis backup preflight checks..."
    
    # Check if Redis container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${REDIS_CONTAINER}$"; then
        error_exit "Redis container '${REDIS_CONTAINER}' is not running"
    fi
    
    # Check Redis connectivity
    if ! docker exec "$REDIS_CONTAINER" redis-cli ping > /dev/null 2>&1; then
        error_exit "Cannot connect to Redis in container '${REDIS_CONTAINER}'"
    fi
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    # Check disk space (require at least 1GB free)
    AVAILABLE_SPACE=$(df "$BACKUP_DIR" | awk 'NR==2 {print $4}')
    if [ "$AVAILABLE_SPACE" -lt 1048576 ]; then
        error_exit "Insufficient disk space. Available: ${AVAILABLE_SPACE}KB, Required: 1048576KB"
    fi
    
    log "Preflight checks completed successfully"
}

# Get Redis configuration info
get_redis_info() {
    log "Gathering Redis configuration information..."
    
    local redis_info
    redis_info=$(docker exec "$REDIS_CONTAINER" redis-cli INFO)
    
    local redis_version
    redis_version=$(echo "$redis_info" | grep "redis_version:" | cut -d: -f2 | tr -d '\r')
    log "Redis Version: $redis_version"
    
    local used_memory_human
    used_memory_human=$(echo "$redis_info" | grep "used_memory_human:" | cut -d: -f2 | tr -d '\r')
    log "Used Memory: $used_memory_human"
    
    local connected_clients
    connected_clients=$(echo "$redis_info" | grep "connected_clients:" | cut -d: -f2 | tr -d '\r')
    log "Connected Clients: $connected_clients"
    
    local total_commands_processed
    total_commands_processed=$(echo "$redis_info" | grep "total_commands_processed:" | cut -d: -f2 | tr -d '\r')
    log "Total Commands Processed: $total_commands_processed"
    
    # Save info to backup metadata
    echo "$redis_info" > "${BACKUP_DIR}/redis_info_${TIMESTAMP}.txt"
}

# Perform RDB backup
backup_rdb() {
    log "Starting RDB backup..."
    
    # Force Redis to save current state to RDB file
    docker exec "$REDIS_CONTAINER" redis-cli BGSAVE
    
    # Wait for background save to complete
    local save_in_progress=1
    local max_wait=300  # 5 minutes max wait
    local wait_time=0
    
    while [ $save_in_progress -eq 1 ] && [ $wait_time -lt $max_wait ]; do
        if docker exec "$REDIS_CONTAINER" redis-cli LASTSAVE > /dev/null 2>&1; then
            sleep 2
            wait_time=$((wait_time + 2))
            
            # Check if BGSAVE is still running
            local bgsave_status
            bgsave_status=$(docker exec "$REDIS_CONTAINER" redis-cli INFO persistence | grep "rdb_bgsave_in_progress" | cut -d: -f2 | tr -d '\r')
            
            if [ "$bgsave_status" = "0" ]; then
                save_in_progress=0
            fi
        else
            error_exit "Failed to check Redis LASTSAVE status"
        fi
    done
    
    if [ $save_in_progress -eq 1 ]; then
        error_exit "RDB backup timed out after ${max_wait} seconds"
    fi
    
    # Copy RDB file from container
    local rdb_backup_file="${BACKUP_DIR}/dump_${TIMESTAMP}.rdb"
    if docker exec "$REDIS_CONTAINER" test -f /data/dump.rdb; then
        docker cp "${REDIS_CONTAINER}:/data/dump.rdb" "$rdb_backup_file"
        
        # Compress the RDB file
        gzip "$rdb_backup_file"
        log "RDB backup completed: ${rdb_backup_file}.gz"
        
        # Verify backup file exists and has content
        if [ ! -s "${rdb_backup_file}.gz" ]; then
            error_exit "RDB backup file is empty or missing"
        fi
    else
        error_exit "RDB file not found in Redis container"
    fi
}

# Perform AOF backup (if enabled)
backup_aof() {
    log "Checking for AOF backup..."
    
    # Check if AOF is enabled
    local aof_enabled
    aof_enabled=$(docker exec "$REDIS_CONTAINER" redis-cli CONFIG GET appendonly | tail -1 | tr -d '\r')
    
    if [ "$aof_enabled" = "yes" ]; then
        log "AOF is enabled, performing AOF backup..."
        
        # Rewrite AOF to compact it
        docker exec "$REDIS_CONTAINER" redis-cli BGREWRITEAOF
        
        # Wait for AOF rewrite to complete
        local rewrite_in_progress=1
        local max_wait=300  # 5 minutes max wait
        local wait_time=0
        
        while [ $rewrite_in_progress -eq 1 ] && [ $wait_time -lt $max_wait ]; do
            sleep 2
            wait_time=$((wait_time + 2))
            
            # Check if AOF rewrite is still running
            local aof_rewrite_status
            aof_rewrite_status=$(docker exec "$REDIS_CONTAINER" redis-cli INFO persistence | grep "aof_rewrite_in_progress" | cut -d: -f2 | tr -d '\r')
            
            if [ "$aof_rewrite_status" = "0" ]; then
                rewrite_in_progress=0
            fi
        done
        
        if [ $rewrite_in_progress -eq 1 ]; then
            log "WARNING: AOF rewrite timed out, using existing AOF file"
        fi
        
        # Copy AOF file from container
        local aof_backup_file="${BACKUP_DIR}/appendonly_${TIMESTAMP}.aof"
        if docker exec "$REDIS_CONTAINER" test -f /data/appendonly.aof; then
            docker cp "${REDIS_CONTAINER}:/data/appendonly.aof" "$aof_backup_file"
            
            # Compress the AOF file
            gzip "$aof_backup_file"
            log "AOF backup completed: ${aof_backup_file}.gz"
            
            # Verify backup file exists and has content
            if [ ! -s "${aof_backup_file}.gz" ]; then
                log "WARNING: AOF backup file is empty"
            fi
        else
            log "WARNING: AOF file not found in Redis container"
        fi
    else
        log "AOF is disabled, skipping AOF backup"
    fi
}

# Verify backup integrity
verify_backup() {
    log "Verifying backup integrity..."
    
    # Find the most recent RDB backup
    local latest_rdb
    latest_rdb=$(find "$BACKUP_DIR" -name "dump_*.rdb.gz" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -n "$latest_rdb" ] && [ -f "$latest_rdb" ]; then
        # Test if the gzip file is valid
        if gzip -t "$latest_rdb" 2>/dev/null; then
            log "RDB backup integrity verified: $latest_rdb"
            
            # Get file size
            local file_size
            file_size=$(stat -c%s "$latest_rdb")
            log "RDB backup size: ${file_size} bytes"
        else
            error_exit "RDB backup integrity check failed: $latest_rdb"
        fi
    else
        error_exit "No RDB backup found for verification"
    fi
    
    # Check AOF backup if it exists
    local latest_aof
    latest_aof=$(find "$BACKUP_DIR" -name "appendonly_*.aof.gz" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- 2>/dev/null || true)
    
    if [ -n "$latest_aof" ] && [ -f "$latest_aof" ]; then
        if gzip -t "$latest_aof" 2>/dev/null; then
            log "AOF backup integrity verified: $latest_aof"
            
            local file_size
            file_size=$(stat -c%s "$latest_aof")
            log "AOF backup size: ${file_size} bytes"
        else
            log "WARNING: AOF backup integrity check failed: $latest_aof"
        fi
    fi
}

# Clean old backups
cleanup_old_backups() {
    log "Cleaning up backups older than ${RETENTION_DAYS} days..."
    
    # Clean RDB backups
    local deleted_count=0
    while IFS= read -r -d '' file; do
        rm "$file"
        deleted_count=$((deleted_count + 1))
        log "Deleted old backup: $(basename "$file")"
    done < <(find "$BACKUP_DIR" -name "dump_*.rdb.gz" -mtime +$RETENTION_DAYS -print0 2>/dev/null || true)
    
    # Clean AOF backups
    while IFS= read -r -d '' file; do
        rm "$file"
        deleted_count=$((deleted_count + 1))
        log "Deleted old backup: $(basename "$file")"
    done < <(find "$BACKUP_DIR" -name "appendonly_*.aof.gz" -mtime +$RETENTION_DAYS -print0 2>/dev/null || true)
    
    # Clean info files
    while IFS= read -r -d '' file; do
        rm "$file"
        deleted_count=$((deleted_count + 1))
        log "Deleted old info: $(basename "$file")"
    done < <(find "$BACKUP_DIR" -name "redis_info_*.txt" -mtime +$RETENTION_DAYS -print0 2>/dev/null || true)
    
    log "Cleanup completed. Deleted $deleted_count old files"
}

# Generate backup report
generate_report() {
    log "Generating backup report..."
    
    local report_file="${BACKUP_DIR}/backup_report_${TIMESTAMP}.json"
    
    cat > "$report_file" << EOF
{
  "backup_timestamp": "${TIMESTAMP}",
  "backup_date": "$(date -Iseconds)",
  "redis_container": "${REDIS_CONTAINER}",
  "backup_directory": "${BACKUP_DIR}",
  "retention_days": ${RETENTION_DAYS},
  "backups_created": {
    "rdb": "$(find "$BACKUP_DIR" -name "dump_${TIMESTAMP}.rdb.gz" -printf '%f' 2>/dev/null || echo 'none')",
    "aof": "$(find "$BACKUP_DIR" -name "appendonly_${TIMESTAMP}.aof.gz" -printf '%f' 2>/dev/null || echo 'none')"
  },
  "backup_sizes": {
    "rdb_bytes": $(stat -c%s "${BACKUP_DIR}/dump_${TIMESTAMP}.rdb.gz" 2>/dev/null || echo 0),
    "aof_bytes": $(stat -c%s "${BACKUP_DIR}/appendonly_${TIMESTAMP}.aof.gz" 2>/dev/null || echo 0)
  },
  "total_backups": $(find "$BACKUP_DIR" -name "*.gz" | wc -l),
  "log_file": "${LOG_FILE}",
  "status": "SUCCESS"
}
EOF
    
    log "Backup report generated: $report_file"
}

# Main execution
main() {
    log "========================================="
    log "Starting Redis backup process"
    log "Timestamp: $TIMESTAMP"
    log "========================================="
    
    preflight_checks
    get_redis_info
    backup_rdb
    backup_aof
    verify_backup
    cleanup_old_backups
    generate_report
    
    log "========================================="
    log "Redis backup completed successfully"
    log "========================================="
}

# Execute main function
main "$@"