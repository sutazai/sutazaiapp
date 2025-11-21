#!/bin/bash
# Master Backup Script for SutazAI Platform
# Orchestrates all database backups and manages global backup operations
# Location: /opt/sutazaiapp/scripts/backup-all.sh

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_BASE_DIR="${BACKUP_DIR:-/opt/sutazaiapp/backups}"
LOG_FILE="${BACKUP_BASE_DIR}/backup-all.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_BASE_DIR"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "========================================="
log "Starting complete system backup"
log "Timestamp: $TIMESTAMP"
log "========================================="

# Initialize counters
TOTAL_BACKUPS=0
SUCCESSFUL_BACKUPS=0
FAILED_BACKUPS=0

# Function to run backup script
run_backup() {
    local script=$1
    local name=$2
    
    log "Running $name backup..."
    TOTAL_BACKUPS=$((TOTAL_BACKUPS + 1))
    
    if bash "$script"; then
        log "✓ $name backup completed successfully"
        SUCCESSFUL_BACKUPS=$((SUCCESSFUL_BACKUPS + 1))
        return 0
    else
        log "✗ $name backup failed"
        FAILED_BACKUPS=$((FAILED_BACKUPS + 1))
        return 1
    fi
}

# Make backup scripts executable
chmod +x "${SCRIPT_DIR}"/backup-*.sh 2>/dev/null || true

# Run all backup scripts
run_backup "${SCRIPT_DIR}/backup-postgres.sh" "PostgreSQL"
run_backup "${SCRIPT_DIR}/backup-neo4j.sh" "Neo4j"
run_backup "${SCRIPT_DIR}/backup-redis.sh" "Redis"
run_backup "${SCRIPT_DIR}/backup-vectors.sh" "Vector Databases"

# Calculate total backup size
TOTAL_SIZE=$(du -sh "$BACKUP_BASE_DIR" | cut -f1)

# Generate backup report
REPORT_FILE="${BACKUP_BASE_DIR}/backup_report_${TIMESTAMP}.txt"
cat > "$REPORT_FILE" <<EOF
SutazAI Platform Backup Report
Generated: $(date)
========================================

Backup Summary:
- Total backup operations: $TOTAL_BACKUPS
- Successful: $SUCCESSFUL_BACKUPS
- Failed: $FAILED_BACKUPS
- Total backup size: $TOTAL_SIZE

Individual Backups:
- PostgreSQL: $(find "${BACKUP_BASE_DIR}/postgres" -name "postgres_*.sql.gz" -type f -mtime -1 | wc -l) new backup(s)
- Neo4j: $(find "${BACKUP_BASE_DIR}/neo4j" -name "neo4j_*" -type f -mtime -1 ! -name "*.log" ! -name "*.sha256" | wc -l) new backup(s)
- Redis: $(find "${BACKUP_BASE_DIR}/redis" -name "redis_*.tar.gz" -type f -mtime -1 | wc -l) new backup(s)
- Vector DBs: $(find "${BACKUP_BASE_DIR}/vectors" -name "*_*.tar.gz" -type f -mtime -1 | wc -l) new backup(s)

Backup Locations:
- Base directory: $BACKUP_BASE_DIR
- PostgreSQL: ${BACKUP_BASE_DIR}/postgres
- Neo4j: ${BACKUP_BASE_DIR}/neo4j
- Redis: ${BACKUP_BASE_DIR}/redis
- Vectors: ${BACKUP_BASE_DIR}/vectors

System Status:
- Running containers: $(docker ps --format '{{.Names}}' | wc -l)
- Disk usage: $(df -h "$BACKUP_BASE_DIR" | tail -1 | awk '{print $5 " used"}')

Notes:
- Retention period: ${RETENTION_DAYS:-7} days
- Automatic rotation enabled
- Checksums generated for all backups

========================================
EOF

log "========================================="
log "Backup Summary:"
log "- Successful: $SUCCESSFUL_BACKUPS/$TOTAL_BACKUPS"
log "- Failed: $FAILED_BACKUPS"
log "- Total size: $TOTAL_SIZE"
log "- Report: $REPORT_FILE"
log "========================================="

# Send notification if configured
if [ -n "${NOTIFICATION_WEBHOOK:-}" ]; then
    curl -X POST "$NOTIFICATION_WEBHOOK" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"SutazAI Backup: $SUCCESSFUL_BACKUPS/$TOTAL_BACKUPS successful, Size: $TOTAL_SIZE\"}" \
        2>/dev/null || log "Failed to send notification"
fi

# Exit with error if any backups failed
if [ $FAILED_BACKUPS -gt 0 ]; then
    log "ERROR: $FAILED_BACKUPS backup(s) failed"
    exit 1
fi

log "All backups completed successfully"
exit 0
