#!/bin/bash
# Verify backup integrity
BACKUP_FILE=$(ls -t /var/backups/sutazai | head -1)
tar -tzf $BACKUP_FILE > /dev/null && echo "Backup is valid" || echo "Backup is corrupted" 