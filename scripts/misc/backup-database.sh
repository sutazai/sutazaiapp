#!/bin/bash
# SutazAI Database Backup Script
# Comprehensive PostgreSQL backup with encryption and verification

set -euo pipefail

# Configuration
BACKUP_DIR="/opt/sutazaiapp/backups/database"
LOG_DIR="/opt/sutazaiapp/logs/backup"
POSTGRES_CONTAINER="sutazai-postgres"
DATABASE_NAME="sutazai"
POSTGRES_USER="sutazai"
RETENTION_DAYS=30
COMPRESSION_LEVEL=6
ENCRYPTION_KEY_FILE="/opt/sutazaiapp/secrets/backup_encryption_key"

# Create directories
mkdir -p "$BACKUP_DIR" "$LOG_DIR"

# Logging setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/backup_database_$TIMESTAMP.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

# Health checks
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        error_exit "Docker is not running"
    fi
    
    # Check if PostgreSQL container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^$POSTGRES_CONTAINER$"; then
        error_exit "PostgreSQL container $POSTGRES_CONTAINER is not running"
    fi
    
    # Check available disk space (require at least 5GB)
    AVAILABLE_SPACE=$(df "$BACKUP_DIR" | awk 'NR==2 {print $4}')
    REQUIRED_SPACE=5242880  # 5GB in KB
    
    if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
        error_exit "Insufficient disk space. Available: ${AVAILABLE_SPACE}KB, Required: ${REQUIRED_SPACE}KB"
    fi
    
    # Check if encryption key exists
    if [ ! -f "$ENCRYPTION_KEY_FILE" ]; then
        log "WARNING: Encryption key not found. Creating new key..."
        openssl rand -base64 32 > "$ENCRYPTION_KEY_FILE"
        chmod 600 "$ENCRYPTION_KEY_FILE"
    fi
    
    log "Prerequisites check completed successfully"
}

# Perform database backup
perform_backup() {
    log "Starting database backup..."
    
    local backup_file="$BACKUP_DIR/sutazai_db_backup_$TIMESTAMP.sql"
    local compressed_file="$backup_file.gz"
    local encrypted_file="$compressed_file.enc"
    
    # Create SQL dump
    log "Creating database dump..."
    docker exec "$POSTGRES_CONTAINER" pg_dump \
        -U "$POSTGRES_USER" \
        -d "$DATABASE_NAME" \
        --verbose \
        --no-password \
        --format=custom \
        --compress="$COMPRESSION_LEVEL" \
        --no-privileges \
        --no-owner > "$backup_file" 2>>"$LOG_FILE"
    
    if [ ! -s "$backup_file" ]; then
        error_exit "Database dump failed or is empty"
    fi
    
    # Compress the backup
    log "Compressing backup..."
    gzip -"$COMPRESSION_LEVEL" "$backup_file"
    
    # Encrypt the compressed backup
    log "Encrypting backup..."
    openssl enc -aes-256-cbc -salt -in "$compressed_file" -out "$encrypted_file" -pass file:"$ENCRYPTION_KEY_FILE"
    
    # Remove unencrypted file
    rm -f "$compressed_file"
    
    # Calculate and store checksum
    log "Calculating checksum..."
    sha256sum "$encrypted_file" > "$encrypted_file.sha256"
    
    log "Backup created: $encrypted_file"
    log "Backup size: $(du -h "$encrypted_file" | cut -f1)"
}

# Verify backup integrity
verify_backup() {
    log "Verifying backup integrity..."
    local latest_backup=$(find "$BACKUP_DIR" -name "sutazai_db_backup_*.sql.gz.enc" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -z "$latest_backup" ]; then
        error_exit "No backup file found for verification"
    fi
    
    # Verify checksum
    log "Verifying checksum..."
    if ! sha256sum -c "$latest_backup.sha256" >/dev/null 2>&1; then
        error_exit "Backup checksum verification failed"
    fi
    
    # Test decryption
    log "Testing decryption..."
    if ! openssl enc -aes-256-cbc -d -in "$latest_backup" -pass file:"$ENCRYPTION_KEY_FILE" >/dev/null 2>&1; then
        error_exit "Backup decryption test failed"
    fi
    
    log "Backup verification completed successfully"
}

# Clean up old backups
cleanup_old_backups() {
    log "Cleaning up old backups (retention: $RETENTION_DAYS days)..."
    
    find "$BACKUP_DIR" -name "sutazai_db_backup_*.sql.gz.enc" -type f -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR" -name "sutazai_db_backup_*.sql.gz.enc.sha256" -type f -mtime +$RETENTION_DAYS -delete
    
    local remaining_backups=$(find "$BACKUP_DIR" -name "sutazai_db_backup_*.sql.gz.enc" -type f | wc -l)
    log "Cleanup completed. Remaining backup files: $remaining_backups"
}

# Export backup metrics to Prometheus
export_metrics() {
    local metrics_file="/opt/sutazaiapp/metrics/backup_metrics.prom"
    mkdir -p "$(dirname "$metrics_file")"
    
    cat > "$metrics_file" << EOF
# HELP sutazai_backup_last_success_timestamp_seconds Last successful backup timestamp
# TYPE sutazai_backup_last_success_timestamp_seconds gauge
sutazai_backup_last_success_timestamp_seconds{type="database"} $(date +%s)

# HELP sutazai_backup_duration_seconds Backup duration in seconds
# TYPE sutazai_backup_duration_seconds gauge
sutazai_backup_duration_seconds{type="database"} $BACKUP_DURATION

# HELP sutazai_backup_size_bytes Backup file size in bytes
# TYPE sutazai_backup_size_bytes gauge
sutazai_backup_size_bytes{type="database"} $BACKUP_SIZE
EOF
}

# Send notifications
send_notification() {
    local status=$1
    local message=$2
    
    # Slack notification (if webhook URL is configured)
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"SutazAI Database Backup $status: $message\"}" \
            "$SLACK_WEBHOOK_URL" >/dev/null 2>&1 || true
    fi
    
    # Email notification (if configured)
    if command -v mail >/dev/null 2>&1 && [ -n "${BACKUP_EMAIL:-}" ]; then
        echo "$message" | mail -s "SutazAI Database Backup $status" "$BACKUP_EMAIL" || true
    fi
    
    log "Notification sent: $status - $message"
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    log "Starting SutazAI database backup process"
    
    # Trap errors and cleanup
    trap 'error_exit "Backup process interrupted"' INT TERM
    
    check_prerequisites
    perform_backup
    verify_backup
    cleanup_old_backups
    
    local end_time=$(date +%s)
    BACKUP_DURATION=$((end_time - start_time))
    
    # Get backup file size
    local latest_backup=$(find "$BACKUP_DIR" -name "sutazai_db_backup_*.sql.gz.enc" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    BACKUP_SIZE=$(stat -c%s "$latest_backup" 2>/dev/null || echo 0)
    
    export_metrics
    
    log "Database backup process completed successfully in ${BACKUP_DURATION} seconds"
    send_notification "SUCCESS" "Database backup completed in ${BACKUP_DURATION} seconds. Size: $(du -h "$latest_backup" | cut -f1)"
}

# Error handling
handle_error() {
    local exit_code=$?
    log "Backup process failed with exit code: $exit_code"
    send_notification "FAILED" "Database backup failed with exit code: $exit_code. Check logs: $LOG_FILE"
    exit $exit_code
}

trap handle_error ERR

# Execute main function
main "$@"