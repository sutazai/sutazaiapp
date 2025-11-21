#!/bin/bash
# Vector Databases Backup Script for SutazAI Platform
# Backs up ChromaDB, Qdrant, and FAISS vector stores
# Location: /opt/sutazaiapp/scripts/backup-vectors.sh

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/opt/sutazaiapp/backups/vectors}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${BACKUP_DIR}/backup.log"

# Container names
CHROMADB_CONTAINER="sutazai-chromadb"
QDRANT_CONTAINER="sutazai-qdrant"
FAISS_CONTAINER="sutazai-faiss"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Starting vector databases backup"

# Function to backup a container's data volume
backup_container_data() {
    local container=$1
    local backup_name=$2
    local data_path=$3
    
    if ! docker ps | grep -q "$container"; then
        log "WARNING: Container $container is not running, skipping"
        return 1
    fi
    
    log "Backing up $container..."
    
    # Check if data path exists in container
    if ! docker exec "$container" test -e "$data_path" 2>/dev/null; then
        log "WARNING: Data path $data_path not found in $container"
        return 1
    fi
    
    # Create backup using docker cp
    local backup_path="${BACKUP_DIR}/${backup_name}_${TIMESTAMP}"
    mkdir -p "$backup_path"
    
    if docker cp "${container}:${data_path}" "${backup_path}/" 2>/dev/null; then
        # Compress backup
        tar -czf "${backup_path}.tar.gz" -C "$BACKUP_DIR" "$(basename $backup_path)"
        rm -rf "$backup_path"
        
        local size=$(du -h "${backup_path}.tar.gz" | cut -f1)
        log "$container backup created: $(basename ${backup_path}.tar.gz) (size: $size)"
        
        # Create checksum
        sha256sum "${backup_path}.tar.gz" | cut -d' ' -f1 > "${backup_path}.tar.gz.sha256"
        return 0
    else
        log "ERROR: Failed to backup $container"
        return 1
    fi
}

# Backup ChromaDB
if backup_container_data "$CHROMADB_CONTAINER" "chromadb" "/chroma/chroma"; then
    CHROMADB_SUCCESS=1
else
    CHROMADB_SUCCESS=0
    # Alternative paths for ChromaDB
    backup_container_data "$CHROMADB_CONTAINER" "chromadb" "/data" || true
fi

# Backup Qdrant
if backup_container_data "$QDRANT_CONTAINER" "qdrant" "/qdrant/storage"; then
    QDRANT_SUCCESS=1
else
    QDRANT_SUCCESS=0
fi

# Backup FAISS
# FAISS is typically stateless or stores data in shared volume
if docker ps | grep -q "$FAISS_CONTAINER"; then
    log "Checking FAISS for persistent data..."
    # Try to find FAISS data directory
    if docker exec "$FAISS_CONTAINER" test -d /data 2>/dev/null; then
        backup_container_data "$FAISS_CONTAINER" "faiss" "/data" || FAISS_SUCCESS=0
        FAISS_SUCCESS=1
    else
        log "FAISS appears to be stateless, skipping data backup"
        FAISS_SUCCESS=1
    fi
else
    log "WARNING: FAISS container not running"
    FAISS_SUCCESS=0
fi

# Rotate old backups
log "Rotating backups older than $RETENTION_DAYS days..."
DELETED_COUNT=$(find "$BACKUP_DIR" -name "*_*.tar.gz" -type f -mtime +${RETENTION_DAYS} -delete -print | wc -l)
find "$BACKUP_DIR" -name "*.sha256" -type f -mtime +${RETENTION_DAYS} -delete
log "Deleted $DELETED_COUNT old backup(s)"

# Summary
TOTAL_BACKUPS=$(find "$BACKUP_DIR" -name "*_*.tar.gz" -type f | wc -l)
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
SUCCESS_COUNT=$((CHROMADB_SUCCESS + QDRANT_SUCCESS + FAISS_SUCCESS))

log "Vector databases backup complete"
log "ChromaDB: $([ $CHROMADB_SUCCESS -eq 1 ] && echo 'SUCCESS' || echo 'FAILED')"
log "Qdrant: $([ $QDRANT_SUCCESS -eq 1 ] && echo 'SUCCESS' || echo 'FAILED')"
log "FAISS: $([ $FAISS_SUCCESS -eq 1 ] && echo 'SUCCESS' || echo 'FAILED')"
log "Total backups: $TOTAL_BACKUPS, Total size: $TOTAL_SIZE"

# Exit with error if no backups succeeded
if [ $SUCCESS_COUNT -eq 0 ]; then
    log "ERROR: All vector database backups failed"
    exit 1
fi

log "Vector databases backup completed (${SUCCESS_COUNT}/3 successful)"
exit 0
