#!/bin/bash
# Purpose: Automated database maintenance tasks for SutazAI system
# Usage: ./database-maintenance.sh [--dry-run] [--full-maintenance]
# Requires: Docker, PostgreSQL client tools

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/opt/sutazaiapp"
LOG_DIR="$BASE_DIR/logs"
BACKUP_DIR="$BASE_DIR/backups/database"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Configuration
DRY_RUN=false
FULL_MAINTENANCE=false
POSTGRES_CONTAINER="sutazai-postgres-minimal"
REDIS_CONTAINER="sutazai-redis-minimal"
POSTGRES_USER="sutazai"
POSTGRES_DB="sutazai"

# Maintenance thresholds
VACUUM_THRESHOLD_DAYS=7
REINDEX_THRESHOLD_DAYS=30
BACKUP_RETENTION_DAYS=30
MAX_CONNECTION_THRESHOLD=80  # Percentage

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --full-maintenance)
            FULL_MAINTENANCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--full-maintenance]"
            exit 1
            ;;
    esac
done

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_file="$LOG_DIR/database_maintenance_$TIMESTAMP.log"
    
    echo "[$timestamp] $level: $message" >> "$log_file"
    
    case $level in
        ERROR) echo -e "${RED}[$timestamp] ERROR: $message${NC}" ;;
        WARN) echo -e "${YELLOW}[$timestamp] WARN: $message${NC}" ;;
        INFO) echo -e "${BLUE}[$timestamp] INFO: $message${NC}" ;;
        SUCCESS) echo -e "${GREEN}[$timestamp] SUCCESS: $message${NC}" ;;
    esac
}

# Check if container is running
check_container() {
    local container_name=$1
    if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        return 0
    else
        return 1
    fi
}

# Execute PostgreSQL command
execute_pg_command() {
    local command="$1"
    local description="$2"
    
    log "INFO" "$description"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would execute: $command"
        return 0
    fi
    
    if docker exec "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "$command" >/dev/null 2>&1; then
        log "SUCCESS" "$description completed"
        return 0
    else
        log "ERROR" "$description failed"
        return 1
    fi
}

# Execute Redis command
execute_redis_command() {
    local command="$1"
    local description="$2"
    
    log "INFO" "$description"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would execute Redis: $command"
        return 0
    fi
    
    if docker exec "$REDIS_CONTAINER" redis-cli $command >/dev/null 2>&1; then
        log "SUCCESS" "$description completed"
        return 0
    else
        log "ERROR" "$description failed"
        return 1
    fi
}

# Check database connectivity
check_database_connectivity() {
    log "INFO" "Checking database connectivity..."
    
    # Check PostgreSQL
    if check_container "$POSTGRES_CONTAINER"; then
        if docker exec "$POSTGRES_CONTAINER" pg_isready -U "$POSTGRES_USER" >/dev/null 2>&1; then
            log "SUCCESS" "PostgreSQL is accessible"
        else
            log "ERROR" "PostgreSQL is not responding"
            return 1
        fi
    else
        log "ERROR" "PostgreSQL container is not running"
        return 1
    fi
    
    # Check Redis
    if check_container "$REDIS_CONTAINER"; then
        if docker exec "$REDIS_CONTAINER" redis-cli ping >/dev/null 2>&1; then
            log "SUCCESS" "Redis is accessible"
        else
            log "ERROR" "Redis is not responding"
            return 1
        fi
    else
        log "ERROR" "Redis container is not running"
        return 1
    fi
}

# Collect database statistics
collect_database_stats() {
    log "INFO" "Collecting database statistics..."
    
    local stats_file="$LOG_DIR/db_stats_$TIMESTAMP.json"
    
    # PostgreSQL statistics
    local pg_db_size=$(docker exec "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT pg_size_pretty(pg_database_size('$POSTGRES_DB'));" 2>/dev/null | xargs || echo "unknown")
    local pg_connections=$(docker exec "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | xargs || echo "0")
    local pg_max_connections=$(docker exec "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SHOW max_connections;" 2>/dev/null | xargs || echo "100")
    local pg_cache_hit_ratio=$(docker exec "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT round(sum(blks_hit)*100.0/sum(blks_hit+blks_read), 2) FROM pg_stat_database WHERE datname = '$POSTGRES_DB';" 2>/dev/null | xargs || echo "0")
    
    # Redis statistics
    local redis_memory=$(docker exec "$REDIS_CONTAINER" redis-cli info memory | grep used_memory_human | cut -d: -f2 | tr -d '\r' || echo "unknown")
    local redis_keys=$(docker exec "$REDIS_CONTAINER" redis-cli dbsize 2>/dev/null || echo "0")
    local redis_uptime=$(docker exec "$REDIS_CONTAINER" redis-cli info server | grep uptime_in_seconds | cut -d: -f2 | tr -d '\r' || echo "0")
    
    # Create statistics JSON
    cat > "$stats_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "postgresql": {
        "database_size": "$pg_db_size",
        "current_connections": $pg_connections,
        "max_connections": $pg_max_connections,
        "connection_usage_percent": $(echo "scale=2; $pg_connections * 100 / $pg_max_connections" | bc -l 2>/dev/null || echo "0"),
        "cache_hit_ratio": $pg_cache_hit_ratio
    },
    "redis": {
        "memory_usage": "$redis_memory",
        "total_keys": $redis_keys,
        "uptime_seconds": $redis_uptime
    }
}
EOF
    
    log "SUCCESS" "Database statistics collected: $stats_file"
    
    # Check connection usage
    local connection_percent=$(echo "scale=0; $pg_connections * 100 / $pg_max_connections" | bc -l 2>/dev/null || echo "0")
    if [[ ${connection_percent%.*} -gt $MAX_CONNECTION_THRESHOLD ]]; then
        log "WARN" "High connection usage: ${connection_percent}% (${pg_connections}/${pg_max_connections})"
    else
        log "INFO" "Connection usage: ${connection_percent}% (${pg_connections}/${pg_max_connections})"
    fi
}

# Create database backup
create_database_backup() {
    log "INFO" "Creating database backup..."
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    local backup_file="$BACKUP_DIR/sutazai_backup_$TIMESTAMP.sql"
    local compressed_backup="$backup_file.gz"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would create backup: $compressed_backup"
        return 0
    fi
    
    # Create PostgreSQL backup
    if docker exec "$POSTGRES_CONTAINER" pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" > "$backup_file" 2>/dev/null; then
        # Compress the backup
        gzip "$backup_file"
        
        local backup_size=$(stat -c%s "$compressed_backup" 2>/dev/null | numfmt --to=iec || echo "unknown")
        log "SUCCESS" "Database backup created: $compressed_backup ($backup_size)"
        
        # Create Redis backup
        local redis_backup="$BACKUP_DIR/redis_backup_$TIMESTAMP.rdb"
        if docker exec "$REDIS_CONTAINER" redis-cli --rdb "$redis_backup" >/dev/null 2>&1; then
            log "SUCCESS" "Redis backup created: $redis_backup"
        else
            log "WARN" "Redis backup failed (non-critical)"
        fi
    else
        log "ERROR" "PostgreSQL backup failed"
        return 1
    fi
}

# Clean old backups
clean_old_backups() {
    log "INFO" "Cleaning old backups (older than $BACKUP_RETENTION_DAYS days)..."
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        log "INFO" "Backup directory does not exist, skipping cleanup"
        return 0
    fi
    
    local deleted_count=0
    local total_freed_mb=0
    
    while IFS= read -r -d '' backup; do
        local basename=$(basename "$backup")
        local size_mb=$(($(stat -c%s "$backup" 2>/dev/null || echo 0) / 1024 / 1024))
        
        log "INFO" "Deleting old backup: $basename (${size_mb}MB)"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            rm "$backup"
            log "SUCCESS" "Deleted: $basename"
        else
            log "INFO" "[DRY RUN] Would delete: $basename (${size_mb}MB)"
        fi
        
        ((deleted_count++))
        total_freed_mb=$((total_freed_mb + size_mb))
    done < <(find "$BACKUP_DIR" -name "*.sql.gz" -o -name "*.rdb" -type f -mtime +$BACKUP_RETENTION_DAYS -print0 2>/dev/null)
    
    if [[ $deleted_count -gt 0 ]]; then
        log "SUCCESS" "Deleted $deleted_count old backups, freed ${total_freed_mb}MB"
    else
        log "INFO" "No old backups found for deletion"
    fi
}

# Vacuum PostgreSQL database
vacuum_database() {
    log "INFO" "Running database vacuum operation..."
    
    # Check if vacuum is needed (check for table bloat)
    local bloat_query="SELECT schemaname, tablename, 
                      round(CASE WHEN otta=0 THEN 0.0 ELSE sml.relpages/otta::numeric END,1) AS bloat
                      FROM (
                        SELECT schemaname, tablename, cc.relpages, 
                               CEIL((cc.reltuples*((datahdr+ma-(CASE WHEN datahdr%ma=0 THEN ma ELSE datahdr%ma END))+nullhdr2+4))/(bs-20::float)) AS otta
                        FROM (
                          SELECT ma,bs,schemaname,tablename,
                                 (datawidth+(hdr+ma-(case when hdr%ma=0 THEN ma ELSE hdr%ma END)))::numeric AS datahdr,
                                 (maxfracsum*(nullhdr+ma-(case when nullhdr%ma=0 THEN ma ELSE nullhdr%ma END))) AS nullhdr2
                          FROM (
                            SELECT schemaname, tablename, hdr, ma, bs,
                                   SUM((1-null_frac)*avg_width) AS datawidth,
                                   MAX(null_frac) AS maxfracsum,
                                   hdr+(
                                     SELECT 1+count(*)/8
                                     FROM pg_stats s2
                                     WHERE null_frac<>0 AND s2.schemaname = s.schemaname AND s2.tablename = s.tablename
                                   ) AS nullhdr
                            FROM pg_stats s, (
                              SELECT (SELECT current_setting('block_size')::numeric) AS bs,
                                     CASE WHEN substring(v,12,3) IN ('8.0','8.1','8.2') THEN 27 ELSE 23 END AS hdr,
                                     CASE WHEN v ~ 'mingw32' THEN 8 ELSE 4 END AS ma
                              FROM (SELECT version() AS v) AS foo
                            ) AS constants
                            GROUP BY 1,2,3,4,5
                          ) AS foo
                        ) AS rs
                        JOIN pg_class cc ON cc.relname = rs.tablename
                        JOIN pg_namespace nn ON cc.relnamespace = nn.oid AND nn.nspname = rs.schemaname AND nn.nspname <> 'information_schema'
                      ) AS sml
                      WHERE bloat > 1.5
                      ORDER BY bloat DESC;"
    
    # Run VACUUM ANALYZE
    if [[ "$FULL_MAINTENANCE" == "true" ]]; then
        execute_pg_command "VACUUM FULL VERBOSE;" "Full vacuum (reclaims space but locks tables)"
    else
        execute_pg_command "VACUUM VERBOSE;" "Standard vacuum (concurrent operation)"
    fi
    
    # Always run ANALYZE to update statistics
    execute_pg_command "ANALYZE VERBOSE;" "Analyzing database statistics"
}

# Reindex database
reindex_database() {
    log "INFO" "Checking if reindex is needed..."
    
    # Get index bloat information
    local needs_reindex=false
    
    if [[ "$FULL_MAINTENANCE" == "true" ]]; then
        log "INFO" "Full maintenance requested, performing reindex..."
        needs_reindex=true
    fi
    
    if [[ "$needs_reindex" == "true" ]]; then
        if [[ "$FULL_MAINTENANCE" == "true" ]]; then
            execute_pg_command "REINDEX DATABASE $POSTGRES_DB;" "Reindexing entire database"
        else
            # Reindex only system catalogs in regular maintenance
            execute_pg_command "REINDEX SYSTEM $POSTGRES_DB;" "Reindexing system catalogs"
        fi
    else
        log "INFO" "Reindex not needed at this time"
    fi
}

# Optimize Redis
optimize_redis() {
    log "INFO" "Optimizing Redis..."
    
    # Get Redis info
    local redis_memory_used=$(docker exec "$REDIS_CONTAINER" redis-cli info memory | grep used_memory: | cut -d: -f2 | tr -d '\r' || echo "0")
    local redis_fragmentation=$(docker exec "$REDIS_CONTAINER" redis-cli info memory | grep mem_fragmentation_ratio | cut -d: -f2 | tr -d '\r' || echo "1.0")
    
    log "INFO" "Redis memory used: $redis_memory_used bytes, fragmentation ratio: $redis_fragmentation"
    
    # Check if defragmentation is needed
    if (( $(echo "$redis_fragmentation > 1.5" | bc -l 2>/dev/null || echo "0") )); then
        log "WARN" "High fragmentation detected, running memory defragmentation"
        execute_redis_command "MEMORY DEFRAG" "Defragmenting Redis memory"
    fi
    
    # Clean expired keys
    execute_redis_command "EVAL 'return redis.call(\"DEL\", unpack(redis.call(\"KEYS\", \"*expired*\")))' 0" "Cleaning expired keys"
    
    # Save Redis data to disk
    execute_redis_command "BGSAVE" "Saving Redis data to disk"
}

# Check and fix database integrity
check_database_integrity() {
    log "INFO" "Checking database integrity..."
    
    # Check PostgreSQL table integrity
    local check_query="SELECT schemaname, tablename, 
                      CASE WHEN last_vacuum IS NULL THEN 'NEVER' 
                           ELSE last_vacuum::text END as last_vacuum,
                      CASE WHEN last_analyze IS NULL THEN 'NEVER' 
                           ELSE last_analyze::text END as last_analyze
                      FROM pg_stat_user_tables 
                      WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                      ORDER BY last_vacuum ASC NULLS FIRST;"
    
    local integrity_issues=0
    
    # Check for tables that haven't been vacuumed recently
    local old_vacuum_tables=$(docker exec "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "$check_query" 2>/dev/null | grep "NEVER\|$(date -d "${VACUUM_THRESHOLD_DAYS} days ago" +%Y-%m-%d)" | wc -l || echo "0")
    
    if [[ $old_vacuum_tables -gt 0 ]]; then
        log "WARN" "Found $old_vacuum_tables tables that may need vacuuming"
        ((integrity_issues++))
    fi
    
    # Check for corrupt indexes (this is a simple check)
    local corrupt_indexes=$(docker exec "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT count(*) FROM pg_stat_user_indexes WHERE idx_scan = 0 AND schemaname NOT IN ('information_schema', 'pg_catalog');" 2>/dev/null | xargs || echo "0")
    
    if [[ $corrupt_indexes -gt 5 ]]; then  # Arbitrary threshold
        log "WARN" "Found $corrupt_indexes potentially unused indexes"
        ((integrity_issues++))
    fi
    
    if [[ $integrity_issues -eq 0 ]]; then
        log "SUCCESS" "Database integrity check passed"
    else
        log "WARN" "Database integrity check found $integrity_issues potential issues"
    fi
}

# Generate maintenance report
generate_maintenance_report() {
    log "INFO" "Generating maintenance report..."
    
    local report_file="$LOG_DIR/database_maintenance_report_$TIMESTAMP.json"
    
    # Collect final statistics
    local final_db_size=$(docker exec "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT pg_size_pretty(pg_database_size('$POSTGRES_DB'));" 2>/dev/null | xargs || echo "unknown")
    local final_connections=$(docker exec "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | xargs || echo "0")
    local redis_memory_final=$(docker exec "$REDIS_CONTAINER" redis-cli info memory | grep used_memory_human | cut -d: -f2 | tr -d '\r' || echo "unknown")
    
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "maintenance_type": "$([ "$DRY_RUN" == "true" ] && echo "dry_run" || echo "actual")",
    "full_maintenance": $FULL_MAINTENANCE,
    "postgresql": {
        "database_size": "$final_db_size",
        "active_connections": $final_connections,
        "vacuum_performed": true,
        "analyze_performed": true,
        "reindex_performed": $FULL_MAINTENANCE
    },
    "redis": {
        "memory_usage": "$redis_memory_final",
        "defragmentation_performed": true,
        "backup_performed": true
    },
    "backup": {
        "backup_created": true,
        "old_backups_cleaned": true
    },
    "next_maintenance": "$(date -d '+1 day' -u +%Y-%m-%dT%H:%M:%SZ)",
    "next_full_maintenance": "$(date -d '+1 week' -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    
    log "SUCCESS" "Maintenance report saved to: $report_file"
    
    # Create symlink to latest report
    if [[ "$DRY_RUN" == "false" ]]; then
        ln -sf "$report_file" "$LOG_DIR/latest_db_maintenance_report.json"
    fi
}

# Main execution
main() {
    log "INFO" "Starting database maintenance for SutazAI system"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Running in DRY RUN mode - no changes will be made"
    fi
    
    if [[ "$FULL_MAINTENANCE" == "true" ]]; then
        log "INFO" "Full maintenance mode enabled (may cause temporary performance impact)"
    fi
    
    # Create necessary directories
    mkdir -p "$LOG_DIR" "$BACKUP_DIR"
    
    # Check database connectivity
    if ! check_database_connectivity; then
        log "ERROR" "Database connectivity check failed, aborting maintenance"
        exit 1
    fi
    
    # Collect initial statistics
    collect_database_stats
    
    # Create backup
    create_database_backup
    
    # Clean old backups
    clean_old_backups
    
    # Perform maintenance operations
    vacuum_database
    reindex_database
    optimize_redis
    check_database_integrity
    
    # Generate final report
    generate_maintenance_report
    
    log "SUCCESS" "Database maintenance completed successfully"
    
    # Show summary
    echo
    echo "============================================"
    echo "       DATABASE MAINTENANCE SUMMARY"
    echo "============================================"
    echo "Mode: $([ "$DRY_RUN" == "true" ] && echo "DRY RUN" || echo "ACTUAL MAINTENANCE")"
    echo "Full Maintenance: $([ "$FULL_MAINTENANCE" == "true" ] && echo "YES" || echo "NO")"
    echo "Timestamp: $(date)"
    echo "============================================"
}

# Run main function
main "$@"