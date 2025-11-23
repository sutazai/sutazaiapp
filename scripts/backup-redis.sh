#!/bin/bash
# Redis Backup Script for SutazAI Platform
# RDB snapshot and AOF persistence backup
# Location: /opt/sutazaiapp/scripts/backup-redis.sh

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/opt/sutazaiapp/backups/redis}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
CONTAINER_NAME="${CONTAINER_NAME:-sutazai-redis}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="redis_${TIMESTAMP}"
LOG_FILE="${BACKUP_DIR}/backup.log"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Starting Redis backup"

# Check if container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    log "ERROR: Container $CONTAINER_NAME is not running"
    exit 1
fi

# Trigger Redis BGSAVE
log "Triggering Redis BGSAVE..."
docker exec "$CONTAINER_NAME" redis-cli BGSAVE >/dev/null

# Wait for BGSAVE to complete
log "Waiting for BGSAVE to complete..."
RETRIES=30
while [ $RETRIES -gt 0 ]; do
    LASTSAVE=$(docker exec "$CONTAINER_NAME" redis-cli LASTSAVE)
    sleep 2
    NEWSAVE=$(docker exec "$CONTAINER_NAME" redis-cli LASTSAVE)
    if [ "$NEWSAVE" -gt "$LASTSAVE" ]; then
        log "BGSAVE completed"
        break
    fi
    RETRIES=$((RETRIES - 1))
done

if [ $RETRIES -eq 0 ]; then
    log "WARNING: BGSAVE timeout, proceeding with existing snapshot"
fi

# Create backup directory for this backup
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"
mkdir -p "$BACKUP_PATH"

# Copy RDB file
log "Copying RDB snapshot..."
if docker exec "$CONTAINER_NAME" test -f /data/dump.rdb; then
    docker cp "$CONTAINER_NAME:/data/dump.rdb" "$BACKUP_PATH/dump.rdb"
    log "RDB snapshot copied"
else
    log "WARNING: RDB file not found"
fi

# Copy AOF file if exists
log "Checking for AOF file..."
if docker exec "$CONTAINER_NAME" test -f /data/appendonly.aof 2>/dev/null; then
    docker cp "$CONTAINER_NAME:/data/appendonly.aof" "$BACKUP_PATH/appendonly.aof"
    log "AOF file copied"
else
    log "AOF file not found (may not be enabled)"
fi

# Get Redis info
docker exec "$CONTAINER_NAME" redis-cli INFO > "$BACKUP_PATH/redis_info.txt"

# Compress backup
log "Compressing backup..."
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" -C "$BACKUP_DIR" "$BACKUP_NAME"
rm -rf "$BACKUP_PATH"

BACKUP_SIZE=$(du -h "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" | cut -f1)
log "Backup created: ${BACKUP_NAME}.tar.gz (size: $BACKUP_SIZE)"

# Create checksum
CHECKSUM=$(sha256sum "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" | cut -d' ' -f1)
echo "$CHECKSUM  ${BACKUP_NAME}.tar.gz" > "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz.sha256"
log "Checksum: $CHECKSUM"

# Rotate old backups
log "Rotating backups older than $RETENTION_DAYS days..."
DELETED_COUNT=$(find "$BACKUP_DIR" -name "redis_*.tar.gz" -type f -mtime +${RETENTION_DAYS} -delete -print | wc -l)
find "$BACKUP_DIR" -name "*.sha256" -type f -mtime +${RETENTION_DAYS} -delete
log "Deleted $DELETED_COUNT old backup(s)"

# Summary
TOTAL_BACKUPS=$(find "$BACKUP_DIR" -name "redis_*.tar.gz" -type f | wc -l)
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
log "Backup complete. Total backups: $TOTAL_BACKUPS, Total size: $TOTAL_SIZE"

log "Redis backup completed successfully"
exit 0
