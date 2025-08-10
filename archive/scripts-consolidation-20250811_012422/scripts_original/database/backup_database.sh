#!/bin/bash
# SutazAI Database Backup Script
# Automated PostgreSQL backup with retention policy
# Author: DBA Administrator
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

CONTAINER_NAME="sutazai-postgres"
DB_NAME="sutazai"
DB_USER="sutazai"
BACKUP_DIR="/opt/sutazaiapp/backups/database"
RETENTION_DAYS=30
DATE=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="sutazai_backup_${DATE}.sql"
LOG_FILE="/opt/sutazaiapp/logs/backup_$(date +"%Y%m%d").log"

# Create directories if they don't exist
mkdir -p "$BACKUP_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to create backup
create_backup() {
    log "Starting backup of database: $DB_NAME"
    
    # Test database connectivity first
    if ! docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -c '\l' >/dev/null 2>&1; then
        log "ERROR: Cannot connect to database"
        exit 1
    fi
    
    # Create the backup
    if docker exec "$CONTAINER_NAME" pg_dump -U "$DB_USER" -d "$DB_NAME" \
        --verbose --clean --no-owner --no-privileges > "$BACKUP_DIR/$BACKUP_FILE"; then
        log "SUCCESS: Backup created at $BACKUP_DIR/$BACKUP_FILE"
        
        # Compress the backup
        gzip "$BACKUP_DIR/$BACKUP_FILE"
        log "SUCCESS: Backup compressed to $BACKUP_DIR/$BACKUP_FILE.gz"
        
        # Verify the backup
        if gunzip -t "$BACKUP_DIR/$BACKUP_FILE.gz" 2>/dev/null; then
            log "SUCCESS: Backup verification passed"
            
            # Get backup size
            BACKUP_SIZE=$(du -h "$BACKUP_DIR/$BACKUP_FILE.gz" | cut -f1)
            log "Backup size: $BACKUP_SIZE"
            
            # Record backup in database
            record_backup_success "$BACKUP_FILE.gz" "$BACKUP_SIZE"
        else
            log "ERROR: Backup verification failed"
            exit 1
        fi
    else
        log "ERROR: Backup creation failed"
        exit 1
    fi
}

# Record backup success in database
record_backup_success() {
    local backup_file="$1"
    local backup_size="$2"
    
    docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -c "
        INSERT INTO system_alerts (alert_type, severity, title, description, source, status, metadata)
        VALUES (
            'backup_success',
            'info',
            'Database Backup Completed',
            'Automated backup completed successfully',
            'backup_script',
            'resolved',
            '{\"backup_file\": \"$backup_file\", \"backup_size\": \"$backup_size\", \"backup_date\": \"$(date -Iseconds)\"}}'
        );" >/dev/null 2>&1 || log "WARNING: Could not record backup success in database"
}

# Function to cleanup old backups
cleanup_old_backups() {
    log "Cleaning up backups older than $RETENTION_DAYS days"
    
    if find "$BACKUP_DIR" -name "sutazai_backup_*.sql.gz" -mtime +$RETENTION_DAYS -delete 2>/dev/null; then
        local deleted_count=$(find "$BACKUP_DIR" -name "sutazai_backup_*.sql.gz" -mtime +$RETENTION_DAYS 2>/dev/null | wc -l)
        log "Cleaned up old backups (retention: $RETENTION_DAYS days)"
    else
        log "No old backups to clean up"
    fi
}

# Function to test backup restoration
test_restore() {
    local test_db="sutazai_restore_test"
    
    log "Testing backup restore capability"
    
    # Create test database
    if docker exec "$CONTAINER_NAME" createdb -U "$DB_USER" "$test_db" 2>/dev/null; then
        log "Created test database: $test_db"
        
        # Find the latest backup
        local latest_backup=$(ls -t "$BACKUP_DIR"/sutazai_backup_*.sql.gz 2>/dev/null | head -1)
        
        if [[ -n "$latest_backup" ]]; then
            # Restore to test database
            if gunzip -c "$latest_backup" | docker exec -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$test_db" >/dev/null 2>&1; then
                log "SUCCESS: Backup restore test passed"
                
                # Verify key tables exist
                local table_count=$(docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$test_db" -t -c "
                    SELECT count(*) FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE';" 2>/dev/null | tr -d ' ')
                
                log "Restored database has $table_count tables"
                
                if [[ "$table_count" -ge 6 ]]; then
                    log "SUCCESS: Key tables verified in restored database"
                else
                    log "WARNING: Restored database may be incomplete"
                fi
            else
                log "ERROR: Backup restore test failed"
            fi
        else
            log "WARNING: No backup file found for restore test"
        fi
        
        # Cleanup test database
        docker exec "$CONTAINER_NAME" dropdb -U "$DB_USER" "$test_db" 2>/dev/null || true
        log "Cleaned up test database"
    else
        log "WARNING: Could not create test database for restore test"
    fi
}

# Main execution
main() {
    log "=== SutazAI Database Backup Started ==="
    
    # Check if container is running
    if ! docker ps --filter name="$CONTAINER_NAME" --filter status=running --quiet | grep -q .; then
        log "ERROR: Container $CONTAINER_NAME is not running"
        exit 1
    fi
    
    create_backup
    cleanup_old_backups
    
    # Test restore functionality (optional - can be disabled for performance)
    if [[ "${SKIP_RESTORE_TEST:-false}" != "true" ]]; then
        test_restore
    fi
    
    # Display backup statistics
    local backup_count=$(find "$BACKUP_DIR" -name "sutazai_backup_*.sql.gz" | wc -l)
    local total_size=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)
    
    log "=== Backup Statistics ==="
    log "Total backups: $backup_count"
    log "Total backup size: $total_size"
    log "Retention policy: $RETENTION_DAYS days"
    log "=== SutazAI Database Backup Completed ==="
}

# Run main function
main "$@"