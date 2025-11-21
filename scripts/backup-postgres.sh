#!/bin/bash
# PostgreSQL Backup Script for SutazAI Platform
# Automated database backup with rotation and verification
# Location: /opt/sutazaiapp/scripts/backup-postgres.sh

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/opt/sutazaiapp/backups/postgres}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
CONTAINER_NAME="${CONTAINER_NAME:-sutazai-postgres}"
DB_NAME="${DB_NAME:-jarvis_ai}"
DB_USER="${DB_USER:-jarvis}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="postgres_${DB_NAME}_${TIMESTAMP}.sql.gz"
LOG_FILE="${BACKUP_DIR}/backup.log"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Starting PostgreSQL backup for database: $DB_NAME"

# Check if container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    log "ERROR: Container $CONTAINER_NAME is not running"
    exit 1
fi

# Perform backup
log "Creating backup: $BACKUP_FILE"
if docker exec "$CONTAINER_NAME" pg_dump -U "$DB_USER" -d "$DB_NAME" | gzip > "${BACKUP_DIR}/${BACKUP_FILE}"; then
    log "Backup created successfully: $BACKUP_FILE"
    
    # Get backup size
    BACKUP_SIZE=$(du -h "${BACKUP_DIR}/${BACKUP_FILE}" | cut -f1)
    log "Backup size: $BACKUP_SIZE"
    
    # Verify backup
    log "Verifying backup integrity..."
    if gunzip -t "${BACKUP_DIR}/${BACKUP_FILE}" 2>/dev/null; then
        log "Backup verification successful"
        
        # Create checksum
        CHECKSUM=$(sha256sum "${BACKUP_DIR}/${BACKUP_FILE}" | cut -d' ' -f1)
        echo "$CHECKSUM  ${BACKUP_FILE}" > "${BACKUP_DIR}/${BACKUP_FILE}.sha256"
        log "Checksum created: $CHECKSUM"
    else
        log "ERROR: Backup verification failed"
        exit 1
    fi
else
    log "ERROR: Backup creation failed"
    exit 1
fi

# Rotate old backups
log "Rotating backups older than $RETENTION_DAYS days..."
DELETED_COUNT=$(find "$BACKUP_DIR" -name "postgres_*.sql.gz" -type f -mtime +${RETENTION_DAYS} -delete -print | wc -l)
if [ "$DELETED_COUNT" -gt 0 ]; then
    log "Deleted $DELETED_COUNT old backup(s)"
else
    log "No old backups to delete"
fi

# Also delete old checksums
find "$BACKUP_DIR" -name "*.sha256" -type f -mtime +${RETENTION_DAYS} -delete

# Summary
TOTAL_BACKUPS=$(find "$BACKUP_DIR" -name "postgres_*.sql.gz" -type f | wc -l)
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
log "Backup complete. Total backups: $TOTAL_BACKUPS, Total size: $TOTAL_SIZE"

# Send metrics to Prometheus pushgateway if available
if command -v curl &> /dev/null && [ -n "${PUSHGATEWAY_URL:-}" ]; then
    cat <<EOF | curl --data-binary @- "${PUSHGATEWAY_URL}/metrics/job/postgres_backup"
# TYPE postgres_backup_success gauge
postgres_backup_success 1
# TYPE postgres_backup_size_bytes gauge
postgres_backup_size_bytes $(stat -f%z "${BACKUP_DIR}/${BACKUP_FILE}" 2>/dev/null || stat -c%s "${BACKUP_DIR}/${BACKUP_FILE}")
# TYPE postgres_backup_duration_seconds gauge
postgres_backup_duration_seconds $SECONDS
# TYPE postgres_backup_total_count gauge
postgres_backup_total_count $TOTAL_BACKUPS
EOF
    log "Metrics sent to Pushgateway"
fi

log "PostgreSQL backup completed successfully"
exit 0
