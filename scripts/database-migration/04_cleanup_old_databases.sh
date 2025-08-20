#!/bin/bash
# Cleanup script to remove old SQLite databases after successful migration
# This should only be run AFTER validation confirms successful migration

set -e

# Safety check - require explicit confirmation
if [ "$1" != "--confirm-cleanup" ]; then
    echo "=== DATABASE CLEANUP WARNING ==="
    echo ""
    echo "This script will PERMANENTLY DELETE all SQLite database files"
    echo "after moving them to an archive directory."
    echo ""
    echo "BEFORE RUNNING THIS:"
    echo "1. Ensure migration is complete (run 02_migrate_sqlite_to_postgres.py)"
    echo "2. Ensure validation passed (run 03_validate_migration.py)"
    echo "3. Ensure backup exists (run 00_backup_all_databases.sh)"
    echo "4. Test application with new PostgreSQL database"
    echo ""
    echo "To proceed with cleanup, run:"
    echo "  $0 --confirm-cleanup"
    echo ""
    exit 1
fi

ARCHIVE_DIR="/opt/sutazaiapp/archives/databases/pre_migration_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$ARCHIVE_DIR/cleanup.log"

echo "=== Database Cleanup Script ==="
echo "Starting cleanup at $(date)"
echo "Archive directory: $ARCHIVE_DIR"

# Create archive directory
mkdir -p "$ARCHIVE_DIR"

# Start logging
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

# Check if validation report exists and passed
VALIDATION_REPORT="/opt/sutazaiapp/scripts/database-migration/validation_report.json"
if [ -f "$VALIDATION_REPORT" ]; then
    echo "Checking validation report..."
    SUCCESS_RATE=$(python3 -c "import json; data=json.load(open('$VALIDATION_REPORT')); print(data['validation_summary']['success_rate'])")
    
    if (( $(echo "$SUCCESS_RATE < 100" | bc -l) )); then
        echo "ERROR: Validation did not pass 100% (success rate: $SUCCESS_RATE%)"
        echo "Please fix issues before cleanup"
        exit 1
    fi
    echo "✓ Validation passed with 100% success rate"
else
    echo "WARNING: No validation report found at $VALIDATION_REPORT"
    echo "Run 03_validate_migration.py first"
    exit 1
fi

echo ""
echo "Archiving SQLite databases..."

# Archive all memory.db files
count=0
for db in $(find /opt/sutazaiapp -name "memory.db" -type f 2>/dev/null); do
    # Skip if in backup or archive directory
    if [[ "$db" == *"/backups/"* ]] || [[ "$db" == *"/archives/"* ]]; then
        echo "Skipping backup/archive: $db"
        continue
    fi
    
    # Create relative path structure in archive
    relative_path=${db#/opt/sutazaiapp/}
    archive_path="$ARCHIVE_DIR/$relative_path"
    archive_dir=$(dirname "$archive_path")
    
    mkdir -p "$archive_dir"
    
    echo "Archiving: $db"
    echo "       to: $archive_path"
    
    # Move file to archive
    mv "$db" "$archive_path"
    
    if [ -f "$archive_path" ]; then
        echo "        ✓ Archived successfully"
        ((count++))
        
        # Remove empty .swarm directory if exists
        swarm_dir=$(dirname "$db")
        if [ "$(basename "$swarm_dir")" = ".swarm" ]; then
            if [ -z "$(ls -A "$swarm_dir")" ]; then
                rmdir "$swarm_dir"
                echo "        ✓ Removed empty .swarm directory"
            fi
        fi
    else
        echo "        ✗ Archive failed!"
        exit 1
    fi
    echo ""
done

echo "Archived $count memory.db files"
echo ""

# Archive extended memory database
extended_db="/opt/sutazaiapp/data/mcp/extended-memory/extended_memory.db"
if [ -f "$extended_db" ]; then
    echo "Archiving extended memory database..."
    archive_path="$ARCHIVE_DIR/extended_memory/extended_memory.db"
    mkdir -p "$(dirname "$archive_path")"
    mv "$extended_db" "$archive_path"
    echo "        ✓ Extended memory archived"
    
    # Create placeholder file with migration notice
    cat > "$extended_db.MIGRATED" <<EOF
This database has been migrated to PostgreSQL.
Original file archived at: $archive_path
Migration date: $(date)
EOF
fi

echo ""
echo "Creating archive manifest..."

# Create manifest
cat > "$ARCHIVE_DIR/manifest.json" <<EOF
{
  "cleanup_date": "$(date -Iseconds)",
  "archive_directory": "$ARCHIVE_DIR",
  "databases_archived": $count,
  "total_size": "$(du -sh "$ARCHIVE_DIR" | cut -f1)",
  "postgresql_target": {
    "host": "localhost",
    "port": 10000,
    "database": "sutazai",
    "table": "unified_memory"
  },
  "files_archived": [
$(find "$ARCHIVE_DIR" -type f \( -name "*.db" -o -name "*.sqlite" \) | while read f; do
    echo "    \"${f#$ARCHIVE_DIR/}\","
done | sed '$ s/,$//')
  ]
}
EOF

# Create rollback script
cat > "$ARCHIVE_DIR/rollback.sh" <<'ROLLBACK_SCRIPT'
#!/bin/bash
# Rollback script to restore SQLite databases

set -e

if [ "$1" != "--force" ]; then
    echo "This will restore all SQLite databases from archive."
    echo "WARNING: This will undo the migration to PostgreSQL!"
    echo ""
    echo "To proceed, run: $0 --force"
    exit 1
fi

ARCHIVE_DIR="$(dirname "$0")"
echo "Restoring from archive: $ARCHIVE_DIR"

# Restore all databases to original locations
for archive_file in $(find "$ARCHIVE_DIR" -name "*.db" -o -name "*.sqlite" -type f 2>/dev/null); do
    # Skip manifest and scripts
    if [[ "$archive_file" == *"manifest.json" ]] || [[ "$archive_file" == *".sh" ]]; then
        continue
    fi
    
    relative_path=${archive_file#$ARCHIVE_DIR/}
    
    # Handle extended memory specially
    if [[ "$relative_path" == "extended_memory/extended_memory.db" ]]; then
        target_path="/opt/sutazaiapp/data/mcp/extended-memory/extended_memory.db"
    else
        target_path="/opt/sutazaiapp/$relative_path"
    fi
    
    echo "Restoring: $target_path"
    mkdir -p "$(dirname "$target_path")"
    cp -p "$archive_file" "$target_path"
done

echo "Rollback complete!"
echo "Remember to update application configuration to use SQLite again"
ROLLBACK_SCRIPT

chmod +x "$ARCHIVE_DIR/rollback.sh"

echo ""
echo "=== Cleanup Complete ==="
echo "Archive location: $ARCHIVE_DIR"
echo "Manifest file: $ARCHIVE_DIR/manifest.json"
echo "Rollback script: $ARCHIVE_DIR/rollback.sh"
echo ""
echo "Total archived size: $(du -sh "$ARCHIVE_DIR" | cut -f1)"
echo ""
echo "IMPORTANT: The old SQLite databases have been archived."
echo "The application should now be using PostgreSQL."
echo "If you need to rollback, run: $ARCHIVE_DIR/rollback.sh --force"
echo ""
echo "Cleanup completed at $(date)"