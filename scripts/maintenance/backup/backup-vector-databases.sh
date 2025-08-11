#!/bin/bash

# Vector Databases Backup Script for SutazAI System
# Backs up Qdrant, ChromaDB, and FAISS vector databases
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

QDRANT_CONTAINER="sutazai-qdrant"
CHROMADB_CONTAINER="sutazai-chromadb"
FAISS_CONTAINER="sutazai-faiss"
BACKUP_DIR="/opt/sutazaiapp/backups/vector-databases"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Logging
LOG_FILE="/opt/sutazaiapp/logs/backup_vector_${TIMESTAMP}.log"
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
    log "Starting vector databases backup preflight checks..."
    
    # Check running containers
    local qdrant_running=false
    local chromadb_running=false
    local faiss_running=false
    
    if docker ps --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER}$"; then
        qdrant_running=true
        log "Qdrant container is running"
    else
        log "WARNING: Qdrant container '${QDRANT_CONTAINER}' is not running"
    fi
    
    if docker ps --format '{{.Names}}' | grep -q "^${CHROMADB_CONTAINER}$"; then
        chromadb_running=true
        log "ChromaDB container is running"
    else
        log "WARNING: ChromaDB container '${CHROMADB_CONTAINER}' is not running"
    fi
    
    if docker ps --format '{{.Names}}' | grep -q "^${FAISS_CONTAINER}$"; then
        faiss_running=true
        log "FAISS container is running"
    else
        log "WARNING: FAISS container '${FAISS_CONTAINER}' is not running"
    fi
    
    if ! $qdrant_running && ! $chromadb_running && ! $faiss_running; then
        error_exit "No vector database containers are running"
    fi
    
    # Create backup directories
    mkdir -p "$BACKUP_DIR/qdrant"
    mkdir -p "$BACKUP_DIR/chromadb"
    mkdir -p "$BACKUP_DIR/faiss"
    
    # Check disk space (require at least 2GB free for vector databases)
    AVAILABLE_SPACE=$(df "$BACKUP_DIR" | awk 'NR==2 {print $4}')
    if [ "$AVAILABLE_SPACE" -lt 2097152 ]; then
        error_exit "Insufficient disk space. Available: ${AVAILABLE_SPACE}KB, Required: 2097152KB"
    fi
    
    log "Preflight checks completed successfully"
}

# Test vector database connectivity
test_connectivity() {
    local service=$1
    local container=$2
    local endpoint=$3
    
    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        if curl -s -f "$endpoint" > /dev/null 2>&1; then
            log "$service connectivity: OK"
            return 0
        else
            log "WARNING: $service is running but not responding on $endpoint"
            return 1
        fi
    else
        log "WARNING: $service container not running"
        return 1
    fi
}

# Get database information and statistics
get_database_info() {
    log "Gathering vector database information..."
    
    local info_file="${BACKUP_DIR}/vector_databases_info_${TIMESTAMP}.txt"
    
    {
        echo "=== Vector Databases Information ==="
        echo "Backup Timestamp: $(date -Iseconds)"
        echo "Containers: $QDRANT_CONTAINER, $CHROMADB_CONTAINER, $FAISS_CONTAINER"
        echo ""
        
        # Qdrant information
        echo "=== Qdrant Information ==="
        if test_connectivity "Qdrant" "$QDRANT_CONTAINER" "http://localhost:10101/collections"; then
            echo "Qdrant Status: Active"
            
            # Get collections list
            if curl -s "http://localhost:10101/collections" 2>/dev/null; then
                echo ""
            else
                echo "Unable to retrieve Qdrant collections"
            fi
            
            # Get cluster info
            echo "Qdrant Cluster Info:"
            if curl -s "http://localhost:10101/cluster" 2>/dev/null; then
                echo ""
            else
                echo "Unable to retrieve Qdrant cluster info"
            fi
        else
            echo "Qdrant Status: Unavailable"
        fi
        echo ""
        
        # ChromaDB information
        echo "=== ChromaDB Information ==="
        if test_connectivity "ChromaDB" "$CHROMADB_CONTAINER" "http://localhost:10100/api/v1/heartbeat"; then
            echo "ChromaDB Status: Active"
            
            # Get version
            echo "ChromaDB Version:"
            if curl -s "http://localhost:10100/api/v1/version" 2>/dev/null; then
                echo ""
            else
                echo "Unable to retrieve ChromaDB version"
            fi
            
            # Get collections
            echo "ChromaDB Collections:"
            if curl -s "http://localhost:10100/api/v1/collections" 2>/dev/null; then
                echo ""
            else
                echo "Unable to retrieve ChromaDB collections"
            fi
        else
            echo "ChromaDB Status: Unavailable"
        fi
        echo ""
        
        # FAISS information
        echo "=== FAISS Information ==="
        if docker ps --format '{{.Names}}' | grep -q "^${FAISS_CONTAINER}$"; then
            echo "FAISS Status: Running"
            
            # Try to get some basic info from FAISS container
            if docker exec "$FAISS_CONTAINER" ls -la /app 2>/dev/null; then
                echo ""
            else
                echo "Unable to access FAISS container filesystem"
            fi
        else
            echo "FAISS Status: Not running"
        fi
        
    } > "$info_file"
    
    log "Database information saved to: $info_file"
}

# Backup Qdrant database
backup_qdrant() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER}$"; then
        log "Skipping Qdrant backup - container not running"
        return 0
    fi
    
    log "Starting Qdrant backup..."
    
    local backup_dir="${BACKUP_DIR}/qdrant"
    local collections_backup="${backup_dir}/collections_${TIMESTAMP}.json"
    local data_backup="${backup_dir}/qdrant_data_${TIMESTAMP}.tar.gz"
    
    # Get list of collections
    if curl -s "http://localhost:10101/collections" > "$collections_backup" 2>/dev/null; then
        log "Qdrant collections metadata saved"
    else
        log "WARNING: Failed to get Qdrant collections metadata"
        echo '{"result": {"collections": []}}' > "$collections_backup"
    fi
    
    # Backup each collection's data
    local collections_dir="${backup_dir}/collections_data_${TIMESTAMP}"
    mkdir -p "$collections_dir"
    
    # Parse collection names and backup each
    if command -v jq > /dev/null 2>&1; then
        # Use jq if available
        local collections
        collections=$(cat "$collections_backup" | jq -r '.result.collections[].name' 2>/dev/null || echo "")
        
        if [ -n "$collections" ]; then
            while IFS= read -r collection_name; do
                if [ -n "$collection_name" ] && [ "$collection_name" != "null" ]; then
                    log "Backing up Qdrant collection: $collection_name"
                    
                    # Get collection info
                    curl -s "http://localhost:10101/collections/${collection_name}" > "${collections_dir}/${collection_name}_info.json" 2>/dev/null || log "WARNING: Failed to get info for collection $collection_name"
                    
                    # Get collection points (with limit to avoid memory issues)
                    curl -s "http://localhost:10101/collections/${collection_name}/points/scroll" -H "Content-Type: application/json" -d '{"limit": 1000, "with_payload": true, "with_vector": true}' > "${collections_dir}/${collection_name}_points.json" 2>/dev/null || log "WARNING: Failed to get points for collection $collection_name"
                fi
            done <<< "$collections"
        else
            log "No collections found or unable to parse collection names"
        fi
    else
        log "jq not available, skipping detailed collection backup"
    fi
    
    # Create data directory backup
    if docker run --rm -v sutazaiapp_qdrant_data:/source -v "$backup_dir":/backup busybox tar -czf "/backup/qdrant_data_${TIMESTAMP}.tar.gz" -C /source . 2>/dev/null; then
        log "Qdrant data directory backup completed"
    else
        log "WARNING: Qdrant data directory backup failed"
    fi
    
    # Compress collections data
    if [ -d "$collections_dir" ]; then
        tar -czf "${collections_dir}.tar.gz" -C "$backup_dir" "$(basename "$collections_dir")" 2>/dev/null && rm -rf "$collections_dir"
        log "Qdrant collections data compressed"
    fi
    
    # Compress collections metadata
    gzip "$collections_backup"
    
    log "Qdrant backup completed"
}

# Backup ChromaDB database
backup_chromadb() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CHROMADB_CONTAINER}$"; then
        log "Skipping ChromaDB backup - container not running"
        return 0
    fi
    
    log "Starting ChromaDB backup..."
    
    local backup_dir="${BACKUP_DIR}/chromadb"
    local collections_backup="${backup_dir}/collections_${TIMESTAMP}.json"
    local data_backup="${backup_dir}/chromadb_data_${TIMESTAMP}.tar.gz"
    
    # Get list of collections
    if curl -s "http://localhost:10100/api/v1/collections" > "$collections_backup" 2>/dev/null; then
        log "ChromaDB collections metadata saved"
    else
        log "WARNING: Failed to get ChromaDB collections metadata"
        echo '[]' > "$collections_backup"
    fi
    
    # Backup each collection's data
    local collections_dir="${backup_dir}/collections_data_${TIMESTAMP}"
    mkdir -p "$collections_dir"
    
    # Parse collection names and backup each
    if command -v jq > /dev/null 2>&1; then
        local collections
        collections=$(cat "$collections_backup" | jq -r '.[].name' 2>/dev/null || echo "")
        
        if [ -n "$collections" ]; then
            while IFS= read -r collection_name; do
                if [ -n "$collection_name" ] && [ "$collection_name" != "null" ]; then
                    log "Backing up ChromaDB collection: $collection_name"
                    
                    # Get collection details
                    curl -s "http://localhost:10100/api/v1/collections/${collection_name}" > "${collections_dir}/${collection_name}_details.json" 2>/dev/null || log "WARNING: Failed to get details for collection $collection_name"
                    
                    # Get collection items (with limit)
                    curl -s "http://localhost:10100/api/v1/collections/${collection_name}/get" -H "Content-Type: application/json" -d '{"limit": 1000, "include": ["metadatas", "documents", "embeddings"]}' > "${collections_dir}/${collection_name}_items.json" 2>/dev/null || log "WARNING: Failed to get items for collection $collection_name"
                fi
            done <<< "$collections"
        else
            log "No collections found or unable to parse collection names"
        fi
    else
        log "jq not available, skipping detailed collection backup"
    fi
    
    # Create data directory backup
    if docker run --rm -v sutazaiapp_chromadb_data:/source -v "$backup_dir":/backup busybox tar -czf "/backup/chromadb_data_${TIMESTAMP}.tar.gz" -C /source . 2>/dev/null; then
        log "ChromaDB data directory backup completed"
    else
        log "WARNING: ChromaDB data directory backup failed"
    fi
    
    # Compress collections data
    if [ -d "$collections_dir" ]; then
        tar -czf "${collections_dir}.tar.gz" -C "$backup_dir" "$(basename "$collections_dir")" 2>/dev/null && rm -rf "$collections_dir"
        log "ChromaDB collections data compressed"
    fi
    
    # Compress collections metadata
    gzip "$collections_backup"
    
    log "ChromaDB backup completed"
}

# Backup FAISS database
backup_faiss() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${FAISS_CONTAINER}$"; then
        log "Skipping FAISS backup - container not running"
        return 0
    fi
    
    log "Starting FAISS backup..."
    
    local backup_dir="${BACKUP_DIR}/faiss"
    local data_backup="${backup_dir}/faiss_data_${TIMESTAMP}.tar.gz"
    
    # Get FAISS service info
    local faiss_info="${backup_dir}/faiss_info_${TIMESTAMP}.txt"
    {
        echo "=== FAISS Service Information ==="
        echo "Backup Timestamp: $(date -Iseconds)"
        echo "Container: $FAISS_CONTAINER"
        echo ""
        
        # Try to get basic container info
        echo "Container filesystem:"
        if docker exec "$FAISS_CONTAINER" ls -la /app 2>/dev/null; then
            echo ""
        else
            echo "Unable to access /app directory"
        fi
        
        # Try to get any FAISS-specific info
        echo "FAISS indices:"
        if docker exec "$FAISS_CONTAINER" find /app -name "*.index" -o -name "*.faiss" 2>/dev/null; then
            echo ""
        else
            echo "No FAISS index files found"
        fi
        
    } > "$faiss_info"
    
    # Create data directory backup
    if docker run --rm -v sutazaiapp_faiss_data:/source -v "$backup_dir":/backup busybox tar -czf "/backup/faiss_data_${TIMESTAMP}.tar.gz" -C /source . 2>/dev/null; then
        log "FAISS data directory backup completed"
    else
        log "WARNING: FAISS data directory backup failed"
    fi
    
    # Try to backup application directory from container
    if docker exec "$FAISS_CONTAINER" tar -czf "/tmp/faiss_app_${TIMESTAMP}.tar.gz" -C /app . 2>/dev/null; then
        docker cp "${FAISS_CONTAINER}:/tmp/faiss_app_${TIMESTAMP}.tar.gz" "${backup_dir}/"
        docker exec "$FAISS_CONTAINER" rm -f "/tmp/faiss_app_${TIMESTAMP}.tar.gz"
        log "FAISS application directory backup completed"
    else
        log "WARNING: FAISS application directory backup failed"
    fi
    
    log "FAISS backup completed"
}

# Verify all backup integrity
verify_backups() {
    log "Verifying vector database backup integrity..."
    
    local verification_failed=false
    
    # Verify Qdrant backups
    for gz_file in "$BACKUP_DIR/qdrant"/*.gz; do
        if [ -f "$gz_file" ]; then
            if gzip -t "$gz_file" 2>/dev/null; then
                log "Qdrant backup verified: $(basename "$gz_file")"
            else
                log "WARNING: Qdrant backup verification failed: $(basename "$gz_file")"
                verification_failed=true
            fi
        fi
    done
    
    # Verify ChromaDB backups
    for gz_file in "$BACKUP_DIR/chromadb"/*.gz; do
        if [ -f "$gz_file" ]; then
            if gzip -t "$gz_file" 2>/dev/null; then
                log "ChromaDB backup verified: $(basename "$gz_file")"
            else
                log "WARNING: ChromaDB backup verification failed: $(basename "$gz_file")"
                verification_failed=true
            fi
        fi
    done
    
    # Verify FAISS backups
    for gz_file in "$BACKUP_DIR/faiss"/*.gz; do
        if [ -f "$gz_file" ]; then
            if gzip -t "$gz_file" 2>/dev/null; then
                log "FAISS backup verified: $(basename "$gz_file")"
            else
                log "WARNING: FAISS backup verification failed: $(basename "$gz_file")"
                verification_failed=true
            fi
        fi
    done
    
    if $verification_failed; then
        log "WARNING: Some backup verifications failed"
    else
        log "All vector database backups verified successfully"
    fi
}

# Clean old backups
cleanup_old_backups() {
    log "Cleaning up backups older than ${RETENTION_DAYS} days..."
    
    local deleted_count=0
    
    # Clean all backup directories
    for db_dir in "$BACKUP_DIR"/{qdrant,chromadb,faiss}; do
        if [ -d "$db_dir" ]; then
            while IFS= read -r -d '' file; do
                rm "$file"
                deleted_count=$((deleted_count + 1))
                log "Deleted old backup: $(basename "$file")"
            done < <(find "$db_dir" -type f -mtime +$RETENTION_DAYS -print0 2>/dev/null || true)
        fi
    done
    
    # Clean info files
    while IFS= read -r -d '' file; do
        rm "$file"
        deleted_count=$((deleted_count + 1))
        log "Deleted old info: $(basename "$file")"
    done < <(find "$BACKUP_DIR" -name "*_info_*.txt" -mtime +$RETENTION_DAYS -print0 2>/dev/null || true)
    
    log "Cleanup completed. Deleted $deleted_count old files"
}

# Generate backup report
generate_report() {
    log "Generating backup report..."
    
    local report_file="${BACKUP_DIR}/backup_report_${TIMESTAMP}.json"
    
    local qdrant_backups
    qdrant_backups=$(find "$BACKUP_DIR/qdrant" -name "*_${TIMESTAMP}*" -type f 2>/dev/null | wc -l || echo 0)
    
    local chromadb_backups
    chromadb_backups=$(find "$BACKUP_DIR/chromadb" -name "*_${TIMESTAMP}*" -type f 2>/dev/null | wc -l || echo 0)
    
    local faiss_backups
    faiss_backups=$(find "$BACKUP_DIR/faiss" -name "*_${TIMESTAMP}*" -type f 2>/dev/null | wc -l || echo 0)
    
    cat > "$report_file" << EOF
{
  "backup_timestamp": "${TIMESTAMP}",
  "backup_date": "$(date -Iseconds)",
  "containers": {
    "qdrant": "${QDRANT_CONTAINER}",
    "chromadb": "${CHROMADB_CONTAINER}",
    "faiss": "${FAISS_CONTAINER}"
  },
  "backup_directory": "${BACKUP_DIR}",
  "retention_days": ${RETENTION_DAYS},
  "backups_created": {
    "qdrant_files": ${qdrant_backups},
    "chromadb_files": ${chromadb_backups},
    "faiss_files": ${faiss_backups}
  },
  "total_backup_files": $((qdrant_backups + chromadb_backups + faiss_backups)),
  "log_file": "${LOG_FILE}",
  "status": "SUCCESS"
}
EOF
    
    log "Backup report generated: $report_file"
}

# Main execution
main() {
    log "========================================="
    log "Starting vector databases backup process"
    log "Timestamp: $TIMESTAMP"
    log "========================================="
    
    preflight_checks
    get_database_info
    backup_qdrant
    backup_chromadb
    backup_faiss
    verify_backups
    cleanup_old_backups
    generate_report
    
    log "========================================="
    log "Vector databases backup completed successfully"
    log "========================================="
}

# Execute main function
main "$@"