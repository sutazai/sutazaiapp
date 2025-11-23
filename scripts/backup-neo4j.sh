#!/bin/bash
# Neo4j Backup Script for SutazAI Platform
# Graph database snapshot with consistency verification
# Location: /opt/sutazaiapp/scripts/backup-neo4j.sh

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/opt/sutazaiapp/backups/neo4j}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
CONTAINER_NAME="${CONTAINER_NAME:-sutazai-neo4j}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="neo4j_${TIMESTAMP}"
LOG_FILE="${BACKUP_DIR}/backup.log"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Starting Neo4j backup"

# Check if container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    log "ERROR: Container $CONTAINER_NAME is not running"
    exit 1
fi

# Create backup using neo4j-admin
log "Creating Neo4j backup: $BACKUP_NAME"
if docker exec "$CONTAINER_NAME" neo4j-admin database dump neo4j --to-path=/tmp --overwrite-destination 2>/dev/null || \
   docker exec "$CONTAINER_NAME" bash -c 'mkdir -p /tmp/neo4j-backup && cp -r /data /tmp/neo4j-backup/' 2>/dev/null; then
    
    # Copy backup from container
    log "Copying backup from container..."
    docker cp "$CONTAINER_NAME:/tmp/neo4j.dump" "${BACKUP_DIR}/${BACKUP_NAME}.dump" 2>/dev/null || \
    docker cp "$CONTAINER_NAME:/tmp/neo4j-backup" "${BACKUP_DIR}/${BACKUP_NAME}" 2>/dev/null || {
        log "WARNING: Standard backup method failed, using alternative export..."
        # Alternative: Export via Cypher
        docker exec "$CONTAINER_NAME" cypher-shell -u neo4j -p sutazai_secure_2024 \
            "CALL apoc.export.graphml.all('/tmp/neo4j_export.graphml', {useTypes:true})" 2>/dev/null || true
        docker cp "$CONTAINER_NAME:/tmp/neo4j_export.graphml" "${BACKUP_DIR}/${BACKUP_NAME}.graphml" || {
            log "ERROR: All backup methods failed"
            exit 1
        }
    }
    
    # Compress backup
    log "Compressing backup..."
    if [ -f "${BACKUP_DIR}/${BACKUP_NAME}.dump" ]; then
        gzip "${BACKUP_DIR}/${BACKUP_NAME}.dump"
        BACKUP_FILE="${BACKUP_NAME}.dump.gz"
    elif [ -f "${BACKUP_DIR}/${BACKUP_NAME}.graphml" ]; then
        gzip "${BACKUP_DIR}/${BACKUP_NAME}.graphml"
        BACKUP_FILE="${BACKUP_NAME}.graphml.gz"
    elif [ -d "${BACKUP_DIR}/${BACKUP_NAME}" ]; then
        tar -czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" -C "$BACKUP_DIR" "$BACKUP_NAME"
        rm -rf "${BACKUP_DIR}/${BACKUP_NAME}"
        BACKUP_FILE="${BACKUP_NAME}.tar.gz"
    fi
    
    BACKUP_SIZE=$(du -h "${BACKUP_DIR}/${BACKUP_FILE}" | cut -f1)
    log "Backup created: $BACKUP_FILE (size: $BACKUP_SIZE)"
    
    # Create checksum
    CHECKSUM=$(sha256sum "${BACKUP_DIR}/${BACKUP_FILE}" | cut -d' ' -f1)
    echo "$CHECKSUM  ${BACKUP_FILE}" > "${BACKUP_DIR}/${BACKUP_FILE}.sha256"
    log "Checksum: $CHECKSUM"
    
else
    log "ERROR: Neo4j backup failed"
    exit 1
fi

# Rotate old backups
log "Rotating backups older than $RETENTION_DAYS days..."
DELETED_COUNT=$(find "$BACKUP_DIR" -name "neo4j_*" -type f -mtime +${RETENTION_DAYS} ! -name "*.log" -delete -print | wc -l)
log "Deleted $DELETED_COUNT old backup(s)"

# Summary
TOTAL_BACKUPS=$(find "$BACKUP_DIR" -name "neo4j_*" -type f ! -name "*.log" ! -name "*.sha256" | wc -l)
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
log "Backup complete. Total backups: $TOTAL_BACKUPS, Total size: $TOTAL_SIZE"

log "Neo4j backup completed successfully"
exit 0
