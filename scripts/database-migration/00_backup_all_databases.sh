#!/bin/bash
# Backup all SQLite databases before migration
# Creates timestamped backup directory with all databases

set -e

BACKUP_DIR="/opt/sutazaiapp/backups/databases/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$BACKUP_DIR/backup.log"

echo "=== Database Backup Script ==="
echo "Starting backup at $(date)"
echo "Backup directory: $BACKUP_DIR"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Start logging
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo ""
echo "Finding all SQLite databases..."

# Find and backup all memory.db files (excluding backups and scripts/database-migration)
count=0
for db in $(find /opt/sutazaiapp -name "memory.db" -type f -not -path "*/backups/*" -not -path "*/scripts/database-migration/*" 2>/dev/null); do
    # Create relative path structure in backup
    relative_path=${db#/opt/sutazaiapp/}
    backup_path="$BACKUP_DIR/memory_databases/$relative_path"
    backup_dir=$(dirname "$backup_path")
    
    mkdir -p "$backup_dir"
    
    echo "Backing up: $db"
    echo "        to: $backup_path"
    
    # Copy with preservation of all attributes
    cp -p "$db" "$backup_path"
    
    # Verify the backup
    if [ -f "$backup_path" ]; then
        original_size=$(stat -c%s "$db")
        backup_size=$(stat -c%s "$backup_path")
        
        if [ "$original_size" -eq "$backup_size" ]; then
            echo "        ✓ Verified (size: $original_size bytes)"
            ((count++))
        else
            echo "        ✗ Size mismatch! Original: $original_size, Backup: $backup_size"
            exit 1
        fi
    else
        echo "        ✗ Backup failed!"
        exit 1
    fi
    echo ""
done

echo "Backed up $count memory.db files"
echo ""

# Backup extended memory database
extended_db="/opt/sutazaiapp/data/mcp/extended-memory/extended_memory.db"
if [ -f "$extended_db" ]; then
    echo "Backing up extended memory database..."
    backup_path="$BACKUP_DIR/extended_memory/extended_memory.db"
    mkdir -p "$(dirname "$backup_path")"
    cp -p "$extended_db" "$backup_path"
    echo "        ✓ Extended memory backed up"
fi

# Backup N8N database
n8n_db="/opt/sutazaiapp/data/n8n/database.sqlite"
if [ -f "$n8n_db" ]; then
    echo "Backing up N8N database..."
    backup_path="$BACKUP_DIR/application_databases/n8n_database.sqlite"
    mkdir -p "$(dirname "$backup_path")"
    cp -p "$n8n_db" "$backup_path"
    echo "        ✓ N8N database backed up"
fi

# Backup Flowise database
flowise_db="/opt/sutazaiapp/data/flowise/database.sqlite"
if [ -f "$flowise_db" ]; then
    echo "Backing up Flowise database..."
    backup_path="$BACKUP_DIR/application_databases/flowise_database.sqlite"
    mkdir -p "$(dirname "$backup_path")"
    cp -p "$flowise_db" "$backup_path"
    echo "        ✓ Flowise database backed up"
fi

echo ""
echo "Creating backup manifest..."

# Create manifest file
cat > "$BACKUP_DIR/manifest.json" <<EOF
{
  "backup_date": "$(date -Iseconds)",
  "backup_directory": "$BACKUP_DIR",
  "databases_backed_up": $count,
  "total_size": "$(du -sh "$BACKUP_DIR" | cut -f1)",
  "files": [
$(find "$BACKUP_DIR" -type f -name "*.db" -o -name "*.sqlite" | while read f; do
    echo "    \"${f#$BACKUP_DIR/}\","
done | sed '$ s/,$//')
  ]
}
EOF

# Create restore script
cat > "$BACKUP_DIR/restore.sh" <<'RESTORE_SCRIPT'
#!/bin/bash
# Restore script for database backup

set -e

if [ "$1" != "--force" ]; then
    echo "This will restore all databases from this backup."
    echo "WARNING: This will overwrite existing databases!"
    echo ""
    echo "To proceed, run: $0 --force"
    exit 1
fi

BACKUP_DIR="$(dirname "$0")"
echo "Restoring from: $BACKUP_DIR"

# Restore memory databases
for backup_file in $(find "$BACKUP_DIR/memory_databases" -name "memory.db" -type f 2>/dev/null); do
    relative_path=${backup_file#$BACKUP_DIR/memory_databases/}
    target_path="/opt/sutazaiapp/$relative_path"
    
    echo "Restoring: $target_path"
    mkdir -p "$(dirname "$target_path")"
    cp -p "$backup_file" "$target_path"
done

# Restore extended memory
if [ -f "$BACKUP_DIR/extended_memory/extended_memory.db" ]; then
    echo "Restoring extended memory database..."
    cp -p "$BACKUP_DIR/extended_memory/extended_memory.db" "/opt/sutazaiapp/data/mcp/extended-memory/extended_memory.db"
fi

# Restore application databases
if [ -f "$BACKUP_DIR/application_databases/n8n_database.sqlite" ]; then
    echo "Restoring N8N database..."
    cp -p "$BACKUP_DIR/application_databases/n8n_database.sqlite" "/opt/sutazaiapp/data/n8n/database.sqlite"
fi

if [ -f "$BACKUP_DIR/application_databases/flowise_database.sqlite" ]; then
    echo "Restoring Flowise database..."
    cp -p "$BACKUP_DIR/application_databases/flowise_database.sqlite" "/opt/sutazaiapp/data/flowise/database.sqlite"
fi

echo "Restore complete!"
RESTORE_SCRIPT

chmod +x "$BACKUP_DIR/restore.sh"

echo ""
echo "=== Backup Complete ==="
echo "Backup location: $BACKUP_DIR"
echo "Manifest file: $BACKUP_DIR/manifest.json"
echo "Restore script: $BACKUP_DIR/restore.sh"
echo ""
echo "Total backup size: $(du -sh "$BACKUP_DIR" | cut -f1)"
echo "Backup completed at $(date)"