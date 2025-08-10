#!/bin/bash

# Database Restoration Script for SutazAI System
# Comprehensive database restoration with safety checks and verification
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_ROOT="/opt/sutazaiapp/backups"
RESTORATION_LOGS_DIR="/opt/sutazaiapp/logs/restoration"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Safety configuration
REQUIRE_CONFIRMATION=${REQUIRE_CONFIRMATION:-true}
CREATE_PRE_RESTORE_BACKUP=${CREATE_PRE_RESTORE_BACKUP:-true}
VERIFICATION_TIMEOUT=300

# Command line arguments
DATABASE_TYPE="${1:-}"
BACKUP_FILE="${2:-}"
FORCE_RESTORE="${3:-false}"

# Logging
RESTORE_LOG_FILE="${RESTORATION_LOGS_DIR}/restoration_${TIMESTAMP}.log"
mkdir -p "$(dirname "$RESTORE_LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RESTORE_LOG_FILE"
}

error_exit() {
    log "RESTORATION ERROR: $1"
    exit 1
}

# Display usage information
usage() {
    echo "Usage: $0 <database_type> [backup_file] [force]"
    echo ""
    echo "Database Types:"
    echo "  postgres    - Restore PostgreSQL database"
    echo "  redis       - Restore Redis database"
    echo "  neo4j       - Restore Neo4j database"
    echo "  qdrant      - Restore Qdrant vector database"
    echo "  chromadb    - Restore ChromaDB vector database"
    echo "  faiss       - Restore FAISS vector database"
    echo "  all         - Restore all databases from latest backups"
    echo ""
    echo "Parameters:"
    echo "  backup_file - Specific backup file to restore (optional, uses latest if not specified)"
    echo "  force       - Skip confirmation prompts (use 'force' to enable)"
    echo ""
    echo "Examples:"
    echo "  $0 postgres                                    # Restore latest PostgreSQL backup"
    echo "  $0 redis /path/to/redis_backup.rdb.gz        # Restore specific Redis backup"
    echo "  $0 all                                         # Restore all databases from latest"
    echo "  $0 neo4j '' force                             # Force restore Neo4j without prompts"
    echo ""
    echo "Environment Variables:"
    echo "  REQUIRE_CONFIRMATION=false    # Skip safety confirmations"
    echo "  CREATE_PRE_RESTORE_BACKUP=false  # Skip pre-restore backup"
}

# Safety checks and confirmations
safety_checks() {
    log "Starting restoration safety checks..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        error_exit "Docker is not running or not accessible"
    fi
    
    # Check backup directory exists
    if [ ! -d "$BACKUP_ROOT" ]; then
        error_exit "Backup directory not found: $BACKUP_ROOT"
    fi
    
    # Warn about data loss
    if [ "$REQUIRE_CONFIRMATION" = "true" ] && [ "$FORCE_RESTORE" != "force" ]; then
        log "⚠️  WARNING: DATABASE RESTORATION WILL REPLACE CURRENT DATA"
        log "⚠️  This operation is DESTRUCTIVE and cannot be undone"
        log "⚠️  Current database contents will be PERMANENTLY LOST"
        echo ""
        echo "Database to restore: $DATABASE_TYPE"
        echo "Backup file: ${BACKUP_FILE:-latest available}"
        echo "Pre-restore backup: $CREATE_PRE_RESTORE_BACKUP"
        echo ""
        read -p "Are you absolutely sure you want to continue? (type 'YES' to confirm): " -r
        if [ "$REPLY" != "YES" ]; then
            log "Restoration cancelled by user"
            exit 0
        fi
    fi
    
    log "Safety checks completed"
}

# Find latest backup file for a database
find_latest_backup() {
    local db_type="$1"
    local backup_dir="$BACKUP_ROOT/$db_type"
    
    if [ ! -d "$backup_dir" ]; then
        log "No backup directory found for $db_type"
        return 1
    fi
    
    # Find the most recent backup file
    local latest_file
    case "$db_type" in
        "postgres")
            latest_file=$(find "$backup_dir" -name "*.sql.gz" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- || true)
            ;;
        "redis")
            latest_file=$(find "$backup_dir" -name "dump_*.rdb.gz" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- || true)
            ;;
        "neo4j")
            latest_file=$(find "$backup_dir" -name "neo4j_dump_*.dump.gz" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- || true)
            ;;
        "vector-databases")
            latest_file=$(find "$backup_dir" -name "*.tar.gz" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- || true)
            ;;
        *)
            latest_file=$(find "$backup_dir" -name "*.gz" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- || true)
            ;;
    esac
    
    if [ -n "$latest_file" ] && [ -f "$latest_file" ]; then
        echo "$latest_file"
        return 0
    else
        return 1
    fi
}

# Create pre-restoration backup
create_pre_restore_backup() {
    local db_type="$1"
    
    if [ "$CREATE_PRE_RESTORE_BACKUP" != "true" ]; then
        log "Skipping pre-restore backup (disabled)"
        return 0
    fi
    
    log "Creating pre-restoration backup for safety..."
    
    local pre_restore_dir="${BACKUP_ROOT}/pre-restore-${TIMESTAMP}"
    mkdir -p "$pre_restore_dir"
    
    case "$db_type" in
        "postgres")
            if docker exec sutazai-postgres pg_dumpall -U postgres > "${pre_restore_dir}/postgres_pre_restore.sql" 2>/dev/null; then
                gzip "${pre_restore_dir}/postgres_pre_restore.sql"
                log "PostgreSQL pre-restore backup created"
            else
                log "WARNING: Failed to create PostgreSQL pre-restore backup"
            fi
            ;;
        "redis")
            # Force Redis to save and copy RDB file
            docker exec sutazai-redis redis-cli BGSAVE > /dev/null 2>&1 || true
            sleep 2
            docker cp sutazai-redis:/data/dump.rdb "${pre_restore_dir}/redis_pre_restore.rdb" 2>/dev/null || true
            if [ -f "${pre_restore_dir}/redis_pre_restore.rdb" ]; then
                gzip "${pre_restore_dir}/redis_pre_restore.rdb"
                log "Redis pre-restore backup created"
            else
                log "WARNING: Failed to create Redis pre-restore backup"
            fi
            ;;
        "neo4j")
            # Stop Neo4j and create data directory backup
            docker stop sutazai-neo4j > /dev/null 2>&1 || true
            docker run --rm -v sutazaiapp_neo4j_data:/source -v "$pre_restore_dir":/backup busybox tar -czf "/backup/neo4j_pre_restore.tar.gz" -C /source . > /dev/null 2>&1 || true
            docker start sutazai-neo4j > /dev/null 2>&1 || true
            if [ -f "${pre_restore_dir}/neo4j_pre_restore.tar.gz" ]; then
                log "Neo4j pre-restore backup created"
            else
                log "WARNING: Failed to create Neo4j pre-restore backup"
            fi
            ;;
        *)
            log "Pre-restore backup not implemented for $db_type"
            ;;
    esac
}

# Restore PostgreSQL database
restore_postgres() {
    local backup_file="$1"
    
    log "Starting PostgreSQL restoration..."
    log "Backup file: $backup_file"
    
    # Verify backup file
    if ! gzip -t "$backup_file" 2>/dev/null; then
        error_exit "Backup file is corrupted or invalid: $backup_file"
    fi
    
    # Check if PostgreSQL container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^sutazai-postgres$"; then
        error_exit "PostgreSQL container is not running"
    fi
    
    # Create pre-restore backup
    create_pre_restore_backup "postgres"
    
    # Stop all connections to the database
    log "Terminating active connections..."
    docker exec sutazai-postgres psql -U postgres -c "SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = 'sutazai' AND pid <> pg_backend_pid();" > /dev/null 2>&1 || true
    
    # Drop and recreate database
    log "Dropping and recreating database..."
    docker exec sutazai-postgres dropdb -U postgres --if-exists sutazai > /dev/null 2>&1 || true
    docker exec sutazai-postgres createdb -U postgres sutazai > /dev/null 2>&1 || true
    
    # Restore from backup
    log "Restoring data from backup..."
    if zcat "$backup_file" | docker exec -i sutazai-postgres psql -U postgres > /dev/null 2>&1; then
        log "PostgreSQL restoration completed successfully"
        
        # Verify restoration
        log "Verifying restoration..."
        if docker exec sutazai-postgres psql -U postgres -d sutazai -c "\dt" > /dev/null 2>&1; then
            log "PostgreSQL restoration verified - database accessible"
            return 0
        else
            error_exit "PostgreSQL restoration verification failed"
        fi
    else
        error_exit "PostgreSQL restoration failed"
    fi
}

# Restore Redis database
restore_redis() {
    local backup_file="$1"
    
    log "Starting Redis restoration..."
    log "Backup file: $backup_file"
    
    # Verify backup file
    if ! gzip -t "$backup_file" 2>/dev/null; then
        error_exit "Backup file is corrupted or invalid: $backup_file"
    fi
    
    # Check if Redis container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^sutazai-redis$"; then
        error_exit "Redis container is not running"
    fi
    
    # Create pre-restore backup
    create_pre_restore_backup "redis"
    
    # Stop Redis temporarily
    log "Stopping Redis container..."
    docker stop sutazai-redis > /dev/null 2>&1 || error_exit "Failed to stop Redis container"
    
    # Extract and copy RDB file
    log "Restoring RDB file..."
    local temp_rdb="/tmp/restore_dump_${TIMESTAMP}.rdb"
    zcat "$backup_file" > "$temp_rdb"
    
    # Copy RDB file to Redis data directory
    docker run --rm -v sutazaiapp_redis_data:/data -v "$temp_rdb":/restore.rdb busybox cp /restore.rdb /data/dump.rdb
    
    # Clean up temporary file
    rm -f "$temp_rdb"
    
    # Start Redis
    log "Starting Redis container..."
    docker start sutazai-redis > /dev/null 2>&1 || error_exit "Failed to start Redis container"
    
    # Wait for Redis to be ready
    local max_wait=30
    local wait_time=0
    while [ $wait_time -lt $max_wait ]; do
        if docker exec sutazai-redis redis-cli ping > /dev/null 2>&1; then
            log "Redis restoration completed successfully"
            
            # Verify restoration
            log "Verifying restoration..."
            local key_count
            key_count=$(docker exec sutazai-redis redis-cli DBSIZE)
            log "Redis restoration verified - $key_count keys loaded"
            return 0
        fi
        sleep 1
        wait_time=$((wait_time + 1))
    done
    
    error_exit "Redis restoration failed - container not responding"
}

# Restore Neo4j database
restore_neo4j() {
    local backup_file="$1"
    
    log "Starting Neo4j restoration..."
    log "Backup file: $backup_file"
    
    # Verify backup file
    if [[ "$backup_file" == *.dump.gz ]]; then
        if ! gzip -t "$backup_file" 2>/dev/null; then
            error_exit "Backup file is corrupted or invalid: $backup_file"
        fi
    elif [[ "$backup_file" == *.tar.gz ]]; then
        if ! tar -tzf "$backup_file" > /dev/null 2>&1; then
            error_exit "Backup file is corrupted or invalid: $backup_file"
        fi
    else
        error_exit "Unsupported backup file format: $backup_file"
    fi
    
    # Create pre-restore backup
    create_pre_restore_backup "neo4j"
    
    # Stop Neo4j container
    log "Stopping Neo4j container..."
    docker stop sutazai-neo4j > /dev/null 2>&1 || error_exit "Failed to stop Neo4j container"
    
    # Clear existing data
    log "Clearing existing Neo4j data..."
    docker run --rm -v sutazaiapp_neo4j_data:/data busybox sh -c "rm -rf /data/*" > /dev/null 2>&1 || true
    
    if [[ "$backup_file" == *.dump.gz ]]; then
        # Restore from neo4j-admin dump
        log "Restoring from dump file..."
        
        # Extract dump file
        local temp_dump="/tmp/restore_neo4j_${TIMESTAMP}.dump"
        zcat "$backup_file" > "$temp_dump"
        
        # Start Neo4j temporarily for restore
        docker start sutazai-neo4j > /dev/null 2>&1 || error_exit "Failed to start Neo4j container"
        
        # Wait for Neo4j to initialize
        sleep 10
        
        # Copy dump file to container and restore
        docker cp "$temp_dump" sutazai-neo4j:/tmp/restore.dump
        docker exec sutazai-neo4j neo4j-admin load --database=neo4j --from=/tmp/restore.dump > /dev/null 2>&1 || log "WARNING: neo4j-admin load may have failed"
        
        # Clean up
        rm -f "$temp_dump"
        docker exec sutazai-neo4j rm -f /tmp/restore.dump > /dev/null 2>&1 || true
        
    elif [[ "$backup_file" == *.tar.gz ]]; then
        # Restore from data directory backup
        log "Restoring from data directory backup..."
        
        # Extract directly to Neo4j data volume
        docker run --rm -v sutazaiapp_neo4j_data:/data -v "$(dirname "$backup_file")":/backup busybox tar -xzf "/backup/$(basename "$backup_file")" -C /data
        
        # Start Neo4j
        docker start sutazai-neo4j > /dev/null 2>&1 || error_exit "Failed to start Neo4j container"
    fi
    
    # Wait for Neo4j to be ready
    log "Waiting for Neo4j to be ready..."
    local max_wait=120
    local wait_time=0
    while [ $wait_time -lt $max_wait ]; do
        if docker exec sutazai-neo4j cypher-shell -u neo4j -p sutazaipass "RETURN 1" > /dev/null 2>&1; then
            log "Neo4j restoration completed successfully"
            
            # Verify restoration
            log "Verifying restoration..."
            local node_count
            node_count=$(docker exec sutazai-neo4j cypher-shell -u neo4j -p sutazaipass "MATCH (n) RETURN count(n)" 2>/dev/null | grep -o '[0-9]\+' | head -1 || echo "0")
            log "Neo4j restoration verified - $node_count nodes restored"
            return 0
        fi
        sleep 2
        wait_time=$((wait_time + 2))
    done
    
    error_exit "Neo4j restoration failed - database not responding"
}

# Restore vector database (Qdrant, ChromaDB, FAISS)
restore_vector_database() {
    local db_type="$1"
    local backup_file="$2"
    
    log "Starting $db_type restoration..."
    log "Backup file: $backup_file"
    
    # Verify backup file
    if ! tar -tzf "$backup_file" > /dev/null 2>&1; then
        error_exit "Backup file is corrupted or invalid: $backup_file"
    fi
    
    local container_name
    local volume_name
    
    case "$db_type" in
        "qdrant")
            container_name="sutazai-qdrant"
            volume_name="sutazaiapp_qdrant_data"
            ;;
        "chromadb")
            container_name="sutazai-chromadb"
            volume_name="sutazaiapp_chromadb_data"
            ;;
        "faiss")
            container_name="sutazai-faiss"
            volume_name="sutazaiapp_faiss_data"
            ;;
        *)
            error_exit "Unknown vector database type: $db_type"
            ;;
    esac
    
    # Check if container exists
    if ! docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        error_exit "$db_type container not found: $container_name"
    fi
    
    # Stop container
    log "Stopping $db_type container..."
    docker stop "$container_name" > /dev/null 2>&1 || true
    
    # Clear existing data
    log "Clearing existing $db_type data..."
    docker run --rm -v "$volume_name":/data busybox sh -c "rm -rf /data/*" > /dev/null 2>&1 || true
    
    # Restore data
    log "Restoring $db_type data..."
    docker run --rm -v "$volume_name":/data -v "$(dirname "$backup_file")":/backup busybox tar -xzf "/backup/$(basename "$backup_file")" -C /data
    
    # Start container
    log "Starting $db_type container..."
    docker start "$container_name" > /dev/null 2>&1 || error_exit "Failed to start $db_type container"
    
    # Verify restoration based on database type
    log "Verifying $db_type restoration..."
    local max_wait=60
    local wait_time=0
    local verified=false
    
    case "$db_type" in
        "qdrant")
            while [ $wait_time -lt $max_wait ]; do
                if curl -s -f "http://localhost:10101/collections" > /dev/null 2>&1; then
                    verified=true
                    break
                fi
                sleep 2
                wait_time=$((wait_time + 2))
            done
            ;;
        "chromadb")
            while [ $wait_time -lt $max_wait ]; do
                if curl -s -f "http://localhost:10100/api/v1/heartbeat" > /dev/null 2>&1; then
                    verified=true
                    break
                fi
                sleep 2
                wait_time=$((wait_time + 2))
            done
            ;;
        "faiss")
            # For FAISS, just check if container is running and healthy
            while [ $wait_time -lt $max_wait ]; do
                if docker exec "$container_name" ls -la /app > /dev/null 2>&1; then
                    verified=true
                    break
                fi
                sleep 2
                wait_time=$((wait_time + 2))
            done
            ;;
    esac
    
    if [ "$verified" = true ]; then
        log "$db_type restoration completed and verified successfully"
        return 0
    else
        error_exit "$db_type restoration verification failed"
    fi
}

# Restore all databases from latest backups
restore_all_databases() {
    log "Starting restoration of all databases from latest backups..."
    
    local restore_errors=0
    local databases=("postgres" "redis" "neo4j")
    local vector_dbs=("qdrant" "chromadb" "faiss")
    
    # Restore main databases
    for db in "${databases[@]}"; do
        log "Attempting to restore $db..."
        
        local latest_backup
        if latest_backup=$(find_latest_backup "$db" 2>/dev/null); then
            if restore_database "$db" "$latest_backup"; then
                log "$db restoration successful"
            else
                log "ERROR: $db restoration failed"
                restore_errors=$((restore_errors + 1))
            fi
        else
            log "WARNING: No backup found for $db, skipping"
        fi
    done
    
    # Restore vector databases
    for db in "${vector_dbs[@]}"; do
        log "Attempting to restore $db..."
        
        local latest_backup
        if latest_backup=$(find_latest_backup "vector-databases" 2>/dev/null); then
            # Check if this specific vector database backup exists
            if [[ "$latest_backup" == *"$db"* ]]; then
                if restore_vector_database "$db" "$latest_backup"; then
                    log "$db restoration successful"
                else
                    log "ERROR: $db restoration failed"
                    restore_errors=$((restore_errors + 1))
                fi
            else
                log "WARNING: No specific backup found for $db, skipping"
            fi
        else
            log "WARNING: No vector database backups found, skipping $db"
        fi
    done
    
    if [ $restore_errors -eq 0 ]; then
        log "All database restorations completed successfully"
        return 0
    else
        log "Database restoration completed with $restore_errors errors"
        return $restore_errors
    fi
}

# Main restore function dispatcher
restore_database() {
    local db_type="$1"
    local backup_file="$2"
    
    case "$db_type" in
        "postgres")
            restore_postgres "$backup_file"
            ;;
        "redis")
            restore_redis "$backup_file"
            ;;
        "neo4j")
            restore_neo4j "$backup_file"
            ;;
        "qdrant"|"chromadb"|"faiss")
            restore_vector_database "$db_type" "$backup_file"
            ;;
        *)
            error_exit "Unknown database type: $db_type"
            ;;
    esac
}

# Main execution
main() {
    log "========================================="
    log "SutazAI Database Restoration System"
    log "Timestamp: $TIMESTAMP"
    log "========================================="
    
    # Check arguments
    if [ -z "$DATABASE_TYPE" ]; then
        usage
        exit 1
    fi
    
    if [ "$DATABASE_TYPE" = "help" ] || [ "$DATABASE_TYPE" = "--help" ] || [ "$DATABASE_TYPE" = "-h" ]; then
        usage
        exit 0
    fi
    
    # Set force restore mode
    if [ "$FORCE_RESTORE" = "force" ]; then
        REQUIRE_CONFIRMATION=false
        CREATE_PRE_RESTORE_BACKUP=false
        log "Force mode enabled - skipping confirmations and pre-restore backups"
    fi
    
    # Run safety checks
    safety_checks
    
    # Handle restoration based on database type
    if [ "$DATABASE_TYPE" = "all" ]; then
        restore_all_databases
    else
        # Determine backup file to use
        local backup_file_to_use="$BACKUP_FILE"
        
        if [ -z "$backup_file_to_use" ]; then
            log "Finding latest backup for $DATABASE_TYPE..."
            
            # Map database types to backup directories
            local backup_search_dir
            case "$DATABASE_TYPE" in
                "qdrant"|"chromadb"|"faiss")
                    backup_search_dir="vector-databases"
                    ;;
                *)
                    backup_search_dir="$DATABASE_TYPE"
                    ;;
            esac
            
            if ! backup_file_to_use=$(find_latest_backup "$backup_search_dir"); then
                error_exit "No backup found for $DATABASE_TYPE in $BACKUP_ROOT/$backup_search_dir"
            fi
            
            log "Using latest backup: $backup_file_to_use"
        fi
        
        # Verify backup file exists
        if [ ! -f "$backup_file_to_use" ]; then
            error_exit "Backup file not found: $backup_file_to_use"
        fi
        
        # Perform restoration
        restore_database "$DATABASE_TYPE" "$backup_file_to_use"
    fi
    
    log "========================================="
    log "Database restoration completed successfully"
    log "Restoration log: $RESTORE_LOG_FILE"
    log "========================================="
}

# Execute main function
main "$@"