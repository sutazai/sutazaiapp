#!/bin/bash
# Emergency Rollback Script for Script Consolidation
set -euo pipefail

echo "=== EMERGENCY ROLLBACK INITIATED - $(date) ==="
echo "Backing up current state..."
cp -r /opt/sutazaiapp/scripts /opt/sutazaiapp/scripts.rollback.backup 2>/dev/null || true

echo "Restoring original scripts from backup..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_SCRIPTS="${SCRIPT_DIR}/scripts_original"

if [ ! -d "$BACKUP_SCRIPTS" ]; then
    echo "ERROR: Backup directory not found at $BACKUP_SCRIPTS"
    exit 1
fi

rm -rf /opt/sutazaiapp/scripts
cp -r "$BACKUP_SCRIPTS" /opt/sutazaiapp/scripts

# Restore executable permissions
find /opt/sutazaiapp/scripts -type f -name "*.sh" -exec chmod +x {} \;

echo "Verifying rollback..."
ORIGINAL_COUNT=$(find "$BACKUP_SCRIPTS" -type f -name "*.sh" | wc -l)
RESTORED_COUNT=$(find /opt/sutazaiapp/scripts -type f -name "*.sh" | wc -l)

if [ "$ORIGINAL_COUNT" -eq "$RESTORED_COUNT" ]; then
    echo "SUCCESS: Rollback complete! $RESTORED_COUNT scripts restored."
else
    echo "WARNING: Script count mismatch. Original: $ORIGINAL_COUNT, Restored: $RESTORED_COUNT"
fi

echo "=== ROLLBACK COMPLETED ==="