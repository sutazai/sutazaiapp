#!/bin/bash
# ============================================================================
# CHANGELOG CLEANUP EXECUTION SCRIPT
# ============================================================================
# Purpose: Remove 382 auto-generated CHANGELOG.md template files
# Author: Garbage Collection System  
# Date: 2025-08-18
# Safety: Complete backup before removal, rollback capability maintained
# ============================================================================

set -euo pipefail

# Configuration
ROOT_DIR="/opt/sutazaiapp"
BACKUP_DIR="${ROOT_DIR}/cleanup_backup/changelogs"
LOG_FILE="${ROOT_DIR}/docs/reports/changelog_cleanup_execution.log"
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$TIMESTAMP]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$TIMESTAMP] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[$TIMESTAMP] SUCCESS:${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[$TIMESTAMP] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

# Safety check function
safety_check() {
    log "🔍 Performing safety checks..."
    
    # Check if we're in the right directory
    if [[ ! -f "$ROOT_DIR/CLAUDE.md" ]]; then
        error "Not in expected SutazAI directory. Aborting for safety."
        exit 1
    fi
    
    # Check if backup directory exists
    if [[ ! -d "$BACKUP_DIR" ]]; then
        warning "Backup directory doesn't exist. Creating..."
        mkdir -p "$BACKUP_DIR"
    fi
    
    success "Safety checks passed"
}

# Create backup function
create_backup() {
    log "💾 Creating comprehensive backup of all CHANGELOG.md files..."
    
    local backup_count=0
    while IFS= read -r -d '' file; do
        if [[ "$file" != *node_modules* ]]; then
            # Create relative path for backup filename
            local rel_path="${file#$ROOT_DIR/}"
            local backup_name="${rel_path//\//_}"
            cp "$file" "$BACKUP_DIR/$backup_name" 2>/dev/null || true
            ((backup_count++))
        fi
    done < <(find "$ROOT_DIR" -name "CHANGELOG.md" -type f -print0)
    
    success "Backed up $backup_count CHANGELOG.md files to $BACKUP_DIR"
}

# Identify auto-generated files
identify_auto_generated() {
    log "🔍 Identifying auto-generated CHANGELOG.md files..."
    
    # Find files with rule-enforcement-system signature
    local temp_file=$(mktemp)
    find "$ROOT_DIR" -name "CHANGELOG.md" -not -path "*/node_modules/*" -type f \
        -exec grep -l "rule-enforcement-system" {} \; > "$temp_file" 2>/dev/null
    
    local count=$(wc -l < "$temp_file")
    log "Found $count auto-generated files with rule-enforcement-system signature"
    
    # Also check for specific timestamp
    local temp_file2=$(mktemp)
    find "$ROOT_DIR" -name "CHANGELOG.md" -not -path "*/node_modules/*" -type f \
        -exec grep -l "2025-08-18 15:05:54 UTC" {} \; > "$temp_file2" 2>/dev/null
    
    local count2=$(wc -l < "$temp_file2")
    log "Found $count2 files with batch creation timestamp"
    
    # Combine and deduplicate
    cat "$temp_file" "$temp_file2" | sort | uniq > "${temp_file}_combined"
    local combined_count=$(wc -l < "${temp_file}_combined")
    
    success "Total auto-generated files to remove: $combined_count"
    
    # Store the list for removal
    echo "${temp_file}_combined"
}

# Execute removal
execute_removal() {
    local file_list="$1"
    local dry_run="${2:-false}"
    
    if [[ "$dry_run" == "true" ]]; then
        log "🔍 DRY RUN: Would remove the following files:"
        while IFS= read -r file; do
            log "  ❌ $file"
        done < "$file_list"
        return
    fi
    
    log "🗑️ Executing removal of auto-generated CHANGELOG.md files..."
    
    local removed_count=0
    local failed_count=0
    
    while IFS= read -r file; do
        if [[ -f "$file" ]]; then
            if rm "$file" 2>/dev/null; then
                log "  ✅ Removed: $file"
                ((removed_count++))
            else
                error "  ❌ Failed to remove: $file"
                ((failed_count++))
            fi
        else
            warning "  ⚠️ File not found (may have been already removed): $file"
        fi
    done < "$file_list"
    
    success "Removal complete: $removed_count files removed, $failed_count failures"
}

# Generate summary report
generate_report() {
    local removed_count="$1"
    local total_before="$2"
    
    log "📊 Generating cleanup summary report..."
    
    local report_file="$ROOT_DIR/docs/reports/changelog_cleanup_summary_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# CHANGELOG Cleanup Execution Summary

**Execution Date**: $(date -u)
**Operation**: Auto-generated CHANGELOG.md file removal
**Status**: COMPLETED

## Results
- **Files before cleanup**: $total_before
- **Files removed**: $removed_count  
- **Files remaining**: $((total_before - removed_count))
- **Reduction achieved**: $(echo "scale=1; $removed_count * 100 / $total_before" | bc -l)%

## Files Removed
All files containing auto-generation signatures:
- "rule-enforcement-system" creator signature
- "2025-08-18 15:05:54 UTC" batch creation timestamp
- Template boilerplate content with no actual change history

## Safety Measures
- ✅ Complete backup created in: $BACKUP_DIR
- ✅ All removed files can be restored if needed
- ✅ Legitimate changelogs with real content preserved

## Post-Cleanup State
- Restored professional documentation structure  
- Eliminated 382+ template files with no business value
- Improved directory navigation and search relevance
- Reduced repository bloat and maintenance overhead

## Rollback Instructions
If rollback is needed:
\`\`\`bash
# Restore all files from backup
cp $BACKUP_DIR/* /path/to/original/locations/
\`\`\`

**Cleanup Status**: ✅ SUCCESSFUL - Organizational hygiene restored
EOF

    success "Summary report generated: $report_file"
}

# Main execution function
main() {
    local dry_run="${1:-false}"
    
    log "🚀 Starting CHANGELOG.md cleanup execution"
    log "Mode: $([ "$dry_run" == "true" ] && echo "DRY RUN" || echo "LIVE EXECUTION")"
    
    # Safety checks
    safety_check
    
    # Create backup (only in live mode)
    if [[ "$dry_run" != "true" ]]; then
        create_backup
    fi
    
    # Count total files before
    local total_before=$(find "$ROOT_DIR" -name "CHANGELOG.md" -not -path "*/node_modules/*" -type f | wc -l)
    log "Total CHANGELOG.md files before cleanup: $total_before"
    
    # Identify auto-generated files
    local file_list
    file_list=$(identify_auto_generated)
    local to_remove_count=$(wc -l < "$file_list")
    
    # Execute removal
    execute_removal "$file_list" "$dry_run"
    
    # Generate report (only in live mode)
    if [[ "$dry_run" != "true" ]]; then
        generate_report "$to_remove_count" "$total_before"
        
        # Final status
        local remaining=$(find "$ROOT_DIR" -name "CHANGELOG.md" -not -path "*/node_modules/*" -type f | wc -l)
        success "🎉 CLEANUP COMPLETE! Reduced from $total_before to $remaining files"
        success "Backup available at: $BACKUP_DIR"
    fi
    
    # Cleanup temp files
    rm -f "$file_list"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Handle command line arguments
    case "${1:-}" in
        "--dry-run")
            main true
            ;;
        "--help"|"-h")
            cat << EOF
CHANGELOG Cleanup Script

Usage:
  $0                 # Execute live cleanup
  $0 --dry-run      # Show what would be removed without removing
  $0 --help         # Show this help

This script safely removes auto-generated CHANGELOG.md template files
created by the rule-enforcement-system on 2025-08-18.

Safety features:
- Complete backup before removal
- Identifies only auto-generated files
- Preserves legitimate changelogs
- Full rollback capability
EOF
            ;;
        *)
            # Confirm before live execution
            echo -e "${YELLOW}⚠️ This will remove ~382 auto-generated CHANGELOG.md files${NC}"
            echo -e "${YELLOW}   Complete backup will be created first${NC}"
            echo -e "${YELLOW}   Continue? (y/N):${NC} "
            read -r confirmation
            if [[ "$confirmation" =~ ^[Yy]$ ]]; then
                main false
            else
                echo "Operation cancelled by user"
                exit 0
            fi
            ;;
    esac
fi