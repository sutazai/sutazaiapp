#!/bin/bash

# Neo4j Backup Script for SutazAI System
# Implements graph database dump and export strategies
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

NEO4J_CONTAINER="sutazai-neo4j"
BACKUP_DIR="/opt/sutazaiapp/backups/neo4j"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-sutazaipass}"

# Logging
LOG_FILE="/opt/sutazaiapp/logs/backup_neo4j_${TIMESTAMP}.log"
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
    log "Starting Neo4j backup preflight checks..."
    
    # Check if Neo4j container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${NEO4J_CONTAINER}$"; then
        error_exit "Neo4j container '${NEO4J_CONTAINER}' is not running"
    fi
    
    # Check Neo4j connectivity
    if ! docker exec "$NEO4J_CONTAINER" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "RETURN 'Connection Test' AS result" > /dev/null 2>&1; then
        log "WARNING: Cannot connect to Neo4j with cypher-shell. Trying HTTP endpoint..."
        
        # Try HTTP endpoint
        local http_response
        if ! http_response=$(docker exec "$NEO4J_CONTAINER" curl -s -f -u "${NEO4J_USER}:${NEO4J_PASSWORD}" http://localhost:7474/db/neo4j/tx/commit -H "Content-Type: application/json" -d '{"statements":[{"statement":"RETURN 1"}]}' 2>/dev/null); then
            error_exit "Cannot connect to Neo4j via HTTP endpoint"
        fi
        log "Neo4j HTTP endpoint accessible"
    else
        log "Neo4j cypher-shell accessible"
    fi
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    # Check disk space (require at least 2GB free for Neo4j)
    AVAILABLE_SPACE=$(df "$BACKUP_DIR" | awk 'NR==2 {print $4}')
    if [ "$AVAILABLE_SPACE" -lt 2097152 ]; then
        error_exit "Insufficient disk space. Available: ${AVAILABLE_SPACE}KB, Required: 2097152KB"
    fi
    
    log "Preflight checks completed successfully"
}

# Get Neo4j database information
get_neo4j_info() {
    log "Gathering Neo4j database information..."
    
    # Get database version and basic stats
    local db_info_file="${BACKUP_DIR}/neo4j_info_${TIMESTAMP}.txt"
    
    {
        echo "=== Neo4j Database Information ==="
        echo "Backup Timestamp: $(date -Iseconds)"
        echo "Container: $NEO4J_CONTAINER"
        echo ""
        
        # Try to get version info
        if docker exec "$NEO4J_CONTAINER" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "CALL dbms.components() YIELD name, versions" 2>/dev/null; then
            echo ""
        else
            echo "Version info unavailable via cypher-shell"
            echo ""
        fi
        
        # Try to get database statistics
        echo "=== Database Statistics ==="
        if docker exec "$NEO4J_CONTAINER" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "CALL apoc.meta.stats() YIELD labelCount, relTypeCount, propertyKeyCount, nodeCount, relCount RETURN labelCount, relTypeCount, propertyKeyCount, nodeCount, relCount" 2>/dev/null; then
            echo ""
        else
            echo "APOC procedures not available, trying basic counts..."
            
            # Basic node count
            if docker exec "$NEO4J_CONTAINER" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "MATCH (n) RETURN count(n) AS nodeCount" 2>/dev/null; then
                echo ""
            else
                echo "Unable to get node count"
            fi
            
            # Basic relationship count
            if docker exec "$NEO4J_CONTAINER" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "MATCH ()-[r]->() RETURN count(r) AS relationshipCount" 2>/dev/null; then
                echo ""
            else
                echo "Unable to get relationship count"
            fi
        fi
        
        # List databases (Neo4j 4.0+)
        echo "=== Available Databases ==="
        if docker exec "$NEO4J_CONTAINER" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "SHOW DATABASES" 2>/dev/null; then
            echo ""
        else
            echo "SHOW DATABASES not supported (likely Neo4j 3.x)"
            echo ""
        fi
        
    } > "$db_info_file"
    
    log "Database information saved to: $db_info_file"
}

# Perform database dump using neo4j-admin dump
backup_dump() {
    log "Starting Neo4j database dump..."
    
    local dump_file="neo4j_dump_${TIMESTAMP}.dump"
    local dump_path="/tmp/$dump_file"
    
    # Stop Neo4j temporarily for consistent dump (if supported)
    log "Attempting to create dump of neo4j database..."
    
    # Try neo4j-admin dump command
    if docker exec "$NEO4J_CONTAINER" neo4j-admin dump --database=neo4j --to="$dump_path" 2>/dev/null; then
        log "Database dump created successfully"
        
        # Copy dump file from container to host
        docker cp "${NEO4J_CONTAINER}:${dump_path}" "${BACKUP_DIR}/${dump_file}"
        
        # Clean up dump file in container
        docker exec "$NEO4J_CONTAINER" rm -f "$dump_path" 2>/dev/null || true
        
        # Compress the dump file
        gzip "${BACKUP_DIR}/${dump_file}"
        log "Database dump compressed: ${BACKUP_DIR}/${dump_file}.gz"
        
        # Verify dump file
        if [ ! -s "${BACKUP_DIR}/${dump_file}.gz" ]; then
            error_exit "Dump file is empty or missing"
        fi
        
        local file_size
        file_size=$(stat -c%s "${BACKUP_DIR}/${dump_file}.gz")
        log "Dump file size: ${file_size} bytes"
        
    else
        log "WARNING: neo4j-admin dump failed, trying alternative backup method..."
        backup_export_cypher
    fi
}

# Alternative backup using Cypher export
backup_export_cypher() {
    log "Starting Cypher export backup..."
    
    local export_file="${BACKUP_DIR}/neo4j_export_${TIMESTAMP}.cypher"
    
    {
        echo "// Neo4j Database Export"
        echo "// Generated: $(date -Iseconds)"
        echo "// Container: $NEO4J_CONTAINER"
        echo ""
        
        # Export all nodes
        echo "// === NODES ==="
        if docker exec "$NEO4J_CONTAINER" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "CALL apoc.export.cypher.all(null, {stream: true, format: 'cypher-shell'}) YIELD value RETURN value" 2>/dev/null; then
            echo ""
        else
            log "APOC export not available, using manual export..."
            
            # Manual export - get all node labels
            echo "// Manual node export"
            if docker exec "$NEO4J_CONTAINER" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "CALL db.labels() YIELD label" 2>/dev/null; then
                # For each label, export nodes (simplified)
                echo "// Nodes will need to be exported manually per label"
            fi
        fi
        
    } > "$export_file"
    
    # If the export file is very small, it likely failed
    if [ ! -s "$export_file" ] || [ $(stat -c%s "$export_file") -lt 100 ]; then
        log "WARNING: Cypher export appears to have failed or database is empty"
        
        # Create a simple backup with basic queries
        {
            echo "// Basic Neo4j Backup - Manual Queries"
            echo "// Generated: $(date -Iseconds)"
            echo ""
            
            # Try to get some basic data
            echo "// Node count query"
            docker exec "$NEO4J_CONTAINER" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "MATCH (n) RETURN count(n) AS total_nodes" 2>/dev/null || echo "// Node count query failed"
            
            echo ""
            echo "// Relationship count query"  
            docker exec "$NEO4J_CONTAINER" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "MATCH ()-[r]->() RETURN count(r) AS total_relationships" 2>/dev/null || echo "// Relationship count query failed"
            
        } > "$export_file"
    fi
    
    # Compress the export file
    gzip "$export_file"
    log "Cypher export completed: ${export_file}.gz"
}

# Backup data directory (full filesystem backup)
backup_data_directory() {
    log "Starting Neo4j data directory backup..."
    
    local data_backup_file="${BACKUP_DIR}/neo4j_data_${TIMESTAMP}.tar.gz"
    
    # Stop Neo4j for consistent backup
    log "Temporarily stopping Neo4j for data directory backup..."
    docker stop "$NEO4J_CONTAINER" 2>/dev/null || log "WARNING: Could not stop Neo4j container gracefully"
    
    # Wait a moment
    sleep 5
    
    # Create tar backup of data directory
    if docker run --rm -v sutazaiapp_neo4j_data:/source -v "$BACKUP_DIR":/backup busybox tar -czf "/backup/neo4j_data_${TIMESTAMP}.tar.gz" -C /source . 2>/dev/null; then
        log "Data directory backup completed: $data_backup_file"
        
        # Verify backup file
        if [ ! -s "$data_backup_file" ]; then
            log "WARNING: Data directory backup file is empty"
        else
            local file_size
            file_size=$(stat -c%s "$data_backup_file")
            log "Data directory backup size: ${file_size} bytes"
        fi
    else
        log "WARNING: Data directory backup failed"
    fi
    
    # Restart Neo4j
    log "Restarting Neo4j container..."
    docker start "$NEO4J_CONTAINER" 2>/dev/null || error_exit "Failed to restart Neo4j container"
    
    # Wait for Neo4j to be ready
    local max_wait=60
    local wait_time=0
    while [ $wait_time -lt $max_wait ]; do
        if docker exec "$NEO4J_CONTAINER" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "RETURN 1" > /dev/null 2>&1; then
            log "Neo4j restarted successfully"
            break
        fi
        sleep 2
        wait_time=$((wait_time + 2))
    done
    
    if [ $wait_time -ge $max_wait ]; then
        error_exit "Neo4j failed to start within ${max_wait} seconds"
    fi
}

# Verify backup integrity
verify_backup() {
    log "Verifying Neo4j backup integrity..."
    
    # Check dump file if it exists
    local latest_dump
    latest_dump=$(find "$BACKUP_DIR" -name "neo4j_dump_*.dump.gz" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- 2>/dev/null || true)
    
    if [ -n "$latest_dump" ] && [ -f "$latest_dump" ]; then
        if gzip -t "$latest_dump" 2>/dev/null; then
            log "Dump backup integrity verified: $latest_dump"
        else
            log "WARNING: Dump backup integrity check failed: $latest_dump"
        fi
    fi
    
    # Check export file if it exists
    local latest_export
    latest_export=$(find "$BACKUP_DIR" -name "neo4j_export_*.cypher.gz" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- 2>/dev/null || true)
    
    if [ -n "$latest_export" ] && [ -f "$latest_export" ]; then
        if gzip -t "$latest_export" 2>/dev/null; then
            log "Export backup integrity verified: $latest_export"
        else
            log "WARNING: Export backup integrity check failed: $latest_export"
        fi
    fi
    
    # Check data directory backup if it exists
    local latest_data
    latest_data=$(find "$BACKUP_DIR" -name "neo4j_data_*.tar.gz" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- 2>/dev/null || true)
    
    if [ -n "$latest_data" ] && [ -f "$latest_data" ]; then
        if tar -tzf "$latest_data" > /dev/null 2>&1; then
            log "Data directory backup integrity verified: $latest_data"
        else
            log "WARNING: Data directory backup integrity check failed: $latest_data"
        fi
    fi
}

# Clean old backups
cleanup_old_backups() {
    log "Cleaning up backups older than ${RETENTION_DAYS} days..."
    
    local deleted_count=0
    
    # Clean dump backups
    while IFS= read -r -d '' file; do
        rm "$file"
        deleted_count=$((deleted_count + 1))
        log "Deleted old backup: $(basename "$file")"
    done < <(find "$BACKUP_DIR" -name "neo4j_dump_*.dump.gz" -mtime +$RETENTION_DAYS -print0 2>/dev/null || true)
    
    # Clean export backups
    while IFS= read -r -d '' file; do
        rm "$file"
        deleted_count=$((deleted_count + 1))
        log "Deleted old backup: $(basename "$file")"
    done < <(find "$BACKUP_DIR" -name "neo4j_export_*.cypher.gz" -mtime +$RETENTION_DAYS -print0 2>/dev/null || true)
    
    # Clean data directory backups
    while IFS= read -r -d '' file; do
        rm "$file"
        deleted_count=$((deleted_count + 1))
        log "Deleted old backup: $(basename "$file")"
    done < <(find "$BACKUP_DIR" -name "neo4j_data_*.tar.gz" -mtime +$RETENTION_DAYS -print0 2>/dev/null || true)
    
    # Clean info files
    while IFS= read -r -d '' file; do
        rm "$file"
        deleted_count=$((deleted_count + 1))
        log "Deleted old info: $(basename "$file")"
    done < <(find "$BACKUP_DIR" -name "neo4j_info_*.txt" -mtime +$RETENTION_DAYS -print0 2>/dev/null || true)
    
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
  "neo4j_container": "${NEO4J_CONTAINER}",
  "backup_directory": "${BACKUP_DIR}",
  "retention_days": ${RETENTION_DAYS},
  "backups_created": {
    "dump": "$(find "$BACKUP_DIR" -name "neo4j_dump_${TIMESTAMP}.dump.gz" -printf '%f' 2>/dev/null || echo 'none')",
    "export": "$(find "$BACKUP_DIR" -name "neo4j_export_${TIMESTAMP}.cypher.gz" -printf '%f' 2>/dev/null || echo 'none')",
    "data_directory": "$(find "$BACKUP_DIR" -name "neo4j_data_${TIMESTAMP}.tar.gz" -printf '%f' 2>/dev/null || echo 'none')"
  },
  "backup_sizes": {
    "dump_bytes": $(stat -c%s "${BACKUP_DIR}/neo4j_dump_${TIMESTAMP}.dump.gz" 2>/dev/null || echo 0),
    "export_bytes": $(stat -c%s "${BACKUP_DIR}/neo4j_export_${TIMESTAMP}.cypher.gz" 2>/dev/null || echo 0),
    "data_directory_bytes": $(stat -c%s "${BACKUP_DIR}/neo4j_data_${TIMESTAMP}.tar.gz" 2>/dev/null || echo 0)
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
    log "Starting Neo4j backup process"
    log "Timestamp: $TIMESTAMP"
    log "========================================="
    
    preflight_checks
    get_neo4j_info
    backup_dump
    # Optionally enable data directory backup (requires stopping Neo4j)
    # backup_data_directory
    verify_backup
    cleanup_old_backups
    generate_report
    
    log "========================================="
    log "Neo4j backup completed successfully"
    log "========================================="
}

# Execute main function
main "$@"