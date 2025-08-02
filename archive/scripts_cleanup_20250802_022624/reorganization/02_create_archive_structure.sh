#!/bin/bash
# SutazAI Archive Structure Creation Script
# Creates organized archive directories

set -euo pipefail

# Configuration
ARCHIVE_ROOT="/opt/sutazaiapp/archive/reorganization_$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="/opt/sutazaiapp/logs/reorganization.log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE" >&2
}

# Create comprehensive archive structure
create_archive_structure() {
    log "Creating archive structure at: $ARCHIVE_ROOT"
    
    # Main archive categories
    mkdir -p "$ARCHIVE_ROOT"/{
        duplicates/{
            monitoring,
            deployment,
            testing,
            configuration,
            utilities
        },
        obsolete/{
            old_versions,
            deprecated,
            unused,
            experimental
        },
        redundant/{
            scripts,
            configs,
            documentation,
            workflows
        },
        legacy/{
            old_implementations,
            archived_features,
            deprecated_apis,
            backup_configs
        }
    }
    
    # Create detailed subdirectories
    mkdir -p "$ARCHIVE_ROOT/duplicates/monitoring"/{
        system_monitors,
        dashboard_scripts,
        health_checks,
        log_processors
    }
    
    mkdir -p "$ARCHIVE_ROOT/duplicates/deployment"/{
        old_deploy_scripts,
        backup_deployments,
        test_deployments,
        partial_implementations
    }
    
    mkdir -p "$ARCHIVE_ROOT/duplicates/testing"/{
        old_test_scripts,
        experimental_tests,
        broken_tests,
        performance_tests
    }
    
    mkdir -p "$ARCHIVE_ROOT/duplicates/configuration"/{
        duplicate_configs,
        old_agent_configs,
        experimental_settings,
        backup_configurations
    }
    
    mkdir -p "$ARCHIVE_ROOT/duplicates/utilities"/{
        helper_scripts,
        maintenance_tools,
        cleanup_scripts,
        migration_tools
    }
    
    # Create inventory files
    mkdir -p "$ARCHIVE_ROOT/inventory"
    
    log "Archive structure created successfully"
}

# Create archive documentation
create_archive_documentation() {
    log "Creating archive documentation..."
    
    cat > "$ARCHIVE_ROOT/README.md" << 'EOF'
# SutazAI Archive Directory

This directory contains files moved during the codebase reorganization to maintain system stability while reducing clutter.

## Structure

### duplicates/
Files that have multiple copies or serve similar functions:
- **monitoring/**: Duplicate monitoring and dashboard scripts
- **deployment/**: Multiple deployment script versions
- **testing/**: Redundant test scripts and utilities
- **configuration/**: Duplicate configuration files
- **utilities/**: Helper scripts with overlapping functionality

### obsolete/
Files that are no longer needed:
- **old_versions/**: Previous versions of scripts
- **deprecated/**: Deprecated functionality
- **unused/**: Scripts not referenced anywhere
- **experimental/**: Experimental code that didn't make it to production

### redundant/
Files that duplicate functionality:
- **scripts/**: Scripts with overlapping purposes
- **configs/**: Configuration files with similar settings
- **documentation/**: Duplicate documentation
- **workflows/**: Redundant workflow definitions

### legacy/
Historical files kept for reference:
- **old_implementations/**: Previous implementation approaches
- **archived_features/**: Features that were removed
- **deprecated_apis/**: Old API implementations
- **backup_configs/**: Historical configuration backups

## File Movement Log

All file movements are logged in `inventory/movement_log.txt`

## Safety Measures

- All files are moved, not deleted
- Original timestamps preserved
- Full restoration available via backup
- System tested after each movement phase

## Restoration

If any archived file is needed:
1. Copy from appropriate archive subdirectory
2. Update any references in active code
3. Test system functionality
EOF

    cat > "$ARCHIVE_ROOT/inventory/movement_log.txt" << 'EOF'
# SutazAI File Movement Log
# Tracks all files moved during reorganization

# Format: TIMESTAMP|SOURCE|DESTINATION|REASON|STATUS
# Example: 2024-01-20 10:30:00|/scripts/old_script.sh|/archive/duplicates/scripts/|Duplicate functionality|MOVED

EOF

    log "Archive documentation created"
}

# Create restoration utilities
create_restoration_utilities() {
    log "Creating restoration utilities..."
    
    cat > "$ARCHIVE_ROOT/restore_file.sh" << 'EOF'
#!/bin/bash
# Restore a specific file from archive

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <archived_file_path> <destination_path>"
    echo "Example: $0 duplicates/monitoring/old_monitor.sh /opt/sutazaiapp/scripts/"
    exit 1
fi

ARCHIVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVED_FILE="$1"
DESTINATION="$2"
LOG_FILE="$ARCHIVE_ROOT/inventory/restoration_log.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if archived file exists
if [ ! -f "$ARCHIVE_ROOT/$ARCHIVED_FILE" ]; then
    echo "Error: Archived file not found: $ARCHIVED_FILE"
    exit 1
fi

# Create destination directory if needed
mkdir -p "$(dirname "$DESTINATION")"

# Copy file back
cp "$ARCHIVE_ROOT/$ARCHIVED_FILE" "$DESTINATION"

log "RESTORED|$ARCHIVED_FILE|$DESTINATION|Manual restoration|SUCCESS"

echo "✅ File restored: $ARCHIVED_FILE -> $DESTINATION"
EOF

    chmod +x "$ARCHIVE_ROOT/restore_file.sh"
    
    cat > "$ARCHIVE_ROOT/list_archived.sh" << 'EOF'
#!/bin/bash
# List all archived files by category

ARCHIVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "📁 SutazAI Archived Files"
echo "========================="

for category in duplicates obsolete redundant legacy; do
    if [ -d "$ARCHIVE_ROOT/$category" ]; then
        echo
        echo "📂 ${category^^}"
        echo "$(echo "$category" | tr '[:lower:]' '-')"
        find "$ARCHIVE_ROOT/$category" -type f | sed 's|'"$ARCHIVE_ROOT/$category"'/||' | sort | sed 's/^/  - /'
    fi
done

echo
echo "📊 Summary:"
echo "  Total archived files: $(find "$ARCHIVE_ROOT" -type f ! -path "*/inventory/*" ! -name "*.md" ! -name "*.sh" | wc -l)"
echo "  Archive size: $(du -sh "$ARCHIVE_ROOT" | cut -f1)"
EOF

    chmod +x "$ARCHIVE_ROOT/list_archived.sh"
    
    log "Restoration utilities created"
}

# Create archive management tools
create_management_tools() {
    log "Creating archive management tools..."
    
    cat > "$ARCHIVE_ROOT/archive_stats.sh" << 'EOF'
#!/bin/bash
# Generate archive statistics

ARCHIVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "📈 SutazAI Archive Statistics"
echo "============================="
echo

# Size statistics
echo "📏 Size Information:"
echo "  Total archive size: $(du -sh "$ARCHIVE_ROOT" | cut -f1)"
echo "  Duplicates: $(du -sh "$ARCHIVE_ROOT/duplicates" 2>/dev/null | cut -f1 || echo "0B")"
echo "  Obsolete: $(du -sh "$ARCHIVE_ROOT/obsolete" 2>/dev/null | cut -f1 || echo "0B")"
echo "  Redundant: $(du -sh "$ARCHIVE_ROOT/redundant" 2>/dev/null | cut -f1 || echo "0B")"
echo "  Legacy: $(du -sh "$ARCHIVE_ROOT/legacy" 2>/dev/null | cut -f1 || echo "0B")"
echo

# File count statistics
echo "📊 File Counts:"
for category in duplicates obsolete redundant legacy; do
    if [ -d "$ARCHIVE_ROOT/$category" ]; then
        count=$(find "$ARCHIVE_ROOT/$category" -type f | wc -l)
        echo "  $category: $count files"
    fi
done
echo

# File type breakdown
echo "📝 File Types:"
find "$ARCHIVE_ROOT" -type f ! -path "*/inventory/*" ! -name "*.md" ! -name "*.sh" | sed 's/.*\.//' | sort | uniq -c | sort -nr | head -10 | while read count ext; do
    echo "  .$ext: $count files"
done
echo

# Recent additions
echo "🕒 Recent Additions (last 24 hours):"
find "$ARCHIVE_ROOT" -type f ! -path "*/inventory/*" ! -name "*.md" ! -name "*.sh" -mtime -1 | head -10 | sed 's|'"$ARCHIVE_ROOT"'/||' | sed 's/^/  - /'
EOF

    chmod +x "$ARCHIVE_ROOT/archive_stats.sh"
    
    log "Archive management tools created"
}

# Main function
main() {
    log "Creating SutazAI archive structure..."
    
    # Create the structure
    create_archive_structure
    create_archive_documentation
    create_restoration_utilities
    create_management_tools
    
    # Set proper permissions
    chmod -R 755 "$ARCHIVE_ROOT"
    
    log "Archive structure created successfully: $ARCHIVE_ROOT"
    
    echo "✅ Archive structure created: $ARCHIVE_ROOT"
    echo "📖 Documentation: $ARCHIVE_ROOT/README.md"
    echo "🔧 Utilities: $ARCHIVE_ROOT/*.sh"
    
    # Export archive path for next script
    echo "export SUTAZAI_ARCHIVE_ROOT='$ARCHIVE_ROOT'" > /tmp/sutazai_archive_path.env
}

# Run main function
main "$@"