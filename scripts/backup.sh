#!/bin/bash
BACKUP_DIR="${BASE_DIR}/backups/$(date +%Y-%m-%d)"
mkdir -p "$BACKUP_DIR"

# Model registry backup
tar -czf "$BACKUP_DIR/models.tar.gz" "${MODEL_DIR}/validated"

# Database backup
docker exec postgres pg_dumpall -U ai_user | gzip > "$BACKUP_DIR/db_backup.sql.gz"

# Encrypt backups
gpg --batch --yes --encrypt --recipient "Backup Key" "$BACKUP_DIR"/*.tar.gz 