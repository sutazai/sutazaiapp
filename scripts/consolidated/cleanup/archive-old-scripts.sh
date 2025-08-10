#!/bin/bash
#
# Archive Old Scripts - Safe Cleanup Operation
# Safely archives the 276+ redundant scripts after consolidation
#
# Author: Shell Automation Specialist
# Version: 1.0.0
# Date: 2025-08-11
#
# SAFETY FIRST: This script archives (moves) old scripts rather than deleting them
# This allows for easy recovery if any functionality is discovered to be missing.
#

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly ARCHIVE_DIR="${PROJECT_ROOT}/archive/old_scripts_${TIMESTAMP}"
readonly LOG_FILE="${PROJECT_ROOT}/logs/script_cleanup_${TIMESTAMP}.log"

# Create directories
mkdir -p "$ARCHIVE_DIR" "$(dirname "$LOG_FILE")"

# Logging
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

# Configuration
DRY_RUN="${DRY_RUN:-false}"
PRESERVE_LIST=(
    # Preserve our consolidated scripts
    "scripts/consolidated"
    
    # Preserve essential infrastructure scripts
    "backend/scripts/db/apply_uuid_schema.sh"
    "backend/scripts/db/execute_uuid_migration.sh"
    "backend/start_optimized.sh"
    
    # Preserve entrypoint scripts (these are used by containers)
    "auth/jwt-service/entrypoint.sh"
    "auth/rbac-engine/entrypoint.sh"
    "auth/service-account-manager/entrypoint.sh"
)

# Scripts that should NOT be archived (critical infrastructure)
should_preserve_script() {
    local script_path="$1"
    
    # Check against preserve list
    for preserve_pattern in "${PRESERVE_LIST[@]}"; do
        if [[ "$script_path" == *"$preserve_pattern"* ]]; then
            return 0  # Preserve this script
        fi
    done
    
    return 1  # Archive this script
}

# Archive a script safely
archive_script() {
    local script_path="$1"
    local relative_path="${script_path#${PROJECT_ROOT}/}"
    local archive_path="${ARCHIVE_DIR}/${relative_path}"
    
    # Create directory structure in archive
    mkdir -p "$(dirname "$archive_path")"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would archive $relative_path"
    else
        # Move the script to archive
        mv "$script_path" "$archive_path"
        log_info "Archived: $relative_path"
    fi
}

# Main archive operation
main() {
    log_info "Starting safe script cleanup operation..."
    log_info "Archive location: $ARCHIVE_DIR"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN MODE - No files will be moved"
    fi
    
    # Find all shell scripts (excluding node_modules and our consolidated scripts)
    local all_scripts=()
    while IFS= read -r -d '' script; do
        all_scripts+=("$script")
    done < <(find "$PROJECT_ROOT" -name "*.sh" -type f -not -path "*/node_modules/*" -not -path "*/archive/*" -print0)
    
    local total_scripts=${#all_scripts[@]}
    local preserved_scripts=0
    local archived_scripts=0
    
    log_info "Found $total_scripts shell scripts to evaluate"
    
    # Process each script
    for script_path in "${all_scripts[@]}"; do
        if should_preserve_script "$script_path"; then
            preserved_scripts=$((preserved_scripts + 1))
            local relative_path="${script_path#${PROJECT_ROOT}/}"
            log_info "PRESERVED: $relative_path"
        else
            archive_script "$script_path"
            archived_scripts=$((archived_scripts + 1))
        fi
    done
    
    # Create archive manifest
    create_archive_manifest "$archived_scripts" "$preserved_scripts" "$total_scripts"
    
    # Summary
    log_success "Script cleanup completed!"
    log_info "Total scripts processed: $total_scripts"
    log_info "Scripts preserved: $preserved_scripts" 
    log_info "Scripts archived: $archived_scripts"
    log_info "Archive location: $ARCHIVE_DIR"
    
    if [[ $archived_scripts -gt 0 ]]; then
        echo ""
        echo "✅ SUCCESS: Archived $archived_scripts redundant scripts"
        echo "📁 Archive location: $ARCHIVE_DIR"
        echo "📋 Manifest: ${ARCHIVE_DIR}/archive_manifest.json"
        echo ""
        echo "🔄 To restore archived scripts if needed:"
        echo "   cp -r $ARCHIVE_DIR/* $PROJECT_ROOT/"
        echo ""
    fi
}

# Create archive manifest
create_archive_manifest() {
    local archived_count="$1"
    local preserved_count="$2"
    local total_count="$3"
    
    local manifest_file="${ARCHIVE_DIR}/archive_manifest.json"
    
    cat > "$manifest_file" << EOF
{
    "archive_timestamp": "$TIMESTAMP",
    "archive_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "operation": "SutazAI Script Consolidation Cleanup",
    "consolidation_report": "SHELL_SCRIPT_CONSOLIDATION_REPORT.md",
    "summary": {
        "total_scripts_found": $total_count,
        "scripts_archived": $archived_count,
        "scripts_preserved": $preserved_count,
        "consolidation_achieved": "$(echo "scale=1; $archived_count * 100 / $total_count" | bc -l)%"
    },
    "consolidated_controllers": [
        "scripts/consolidated/master.sh",
        "scripts/consolidated/deployment/master-deploy.sh", 
        "scripts/consolidated/monitoring/master-monitor.sh",
        "scripts/consolidated/maintenance/master-maintenance.sh",
        "scripts/consolidated/testing/master-test.sh",
        "scripts/consolidated/security/master-security.sh"
    ],
    "restoration_command": "cp -r $ARCHIVE_DIR/* $PROJECT_ROOT/",
    "verification_steps": [
        "Test consolidated scripts: ./scripts/consolidated/master.sh status",
        "Check system health: ./scripts/consolidated/master.sh health",
        "Validate security: ./scripts/consolidated/master.sh security validate"
    ]
}
EOF
    
    log_success "Archive manifest created: $manifest_file"
}

# Help usage
show_usage() {
    cat << 'EOF'
Archive Old Scripts - Safe Cleanup Operation

USAGE:
    ./archive-old-scripts.sh [OPTIONS]

OPTIONS:
    --dry-run       Show what would be archived without moving files
    --help          Show this help message

DESCRIPTION:
    Safely archives redundant scripts after consolidation to 6 master controllers.
    Scripts are moved (not deleted) to allow for easy recovery if needed.

PRESERVED SCRIPTS:
    - All consolidated controllers (scripts/consolidated/*)
    - Critical infrastructure scripts (database, entrypoints)
    - Container-required scripts

SAFETY FEATURES:
    - Archives (moves) rather than deletes scripts
    - Creates detailed manifest for restoration
    - Preserves critical infrastructure scripts
    - Dry-run mode for validation

RESTORATION:
    If needed, restore scripts with:
    cp -r /archive/old_scripts_TIMESTAMP/* /opt/sutazaiapp/

EOF
}

# Parse command line
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Execute main function
main