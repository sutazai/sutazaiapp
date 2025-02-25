#!/bin/bash
# Secure Synchronization Wrapper
# Performs dry-run first and maintains versioned backups

SOURCE="/"
DEST="/mnt/backup_drive"
EXCLUDE_FILE="/etc/backup_excludes.conf"

rsync --dry-run -avh --checksum --delete --backup --backup-dir="../versions/$(date +%F)" \
  --exclude-from="$EXCLUDE_FILE" --log-file="/var/log/rsync/$(date +%F).log" \
  "$SOURCE" "$DEST"

read -p "Dry run complete. Review logs and press Y to continue: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  rsync -avh --checksum --delete --backup --backup-dir="../versions/$(date +%F)" \
    --exclude-from="$EXCLUDE_FILE" --log-file="/var/log/rsync/$(date +%F).log" \
    "$SOURCE" "$DEST"
fi 