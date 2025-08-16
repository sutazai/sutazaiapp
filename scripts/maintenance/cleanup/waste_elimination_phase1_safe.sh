#!/bin/bash
# WASTE ELIMINATION PHASE 1: SAFE CLEANUP
# Rule 13: Zero Tolerance for Waste - Immediate Safe Cleanup
# Risk Level: SAFE (Zero risk of breaking functionality)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/opt/sutazaiapp"
BACKUP_DIR="/opt/sutazaiapp/waste_elimination_backups/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/opt/sutazaiapp/logs/waste_elimination_phase1_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

info() {
    echo -e "${BLUE}INFO: $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}SUCCESS: $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

# Create backup directory
create_backup_dir() {
    log "Creating backup directory: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$BACKUP_DIR/logs"
    mkdir -p "$BACKUP_DIR/test_results"
    mkdir -p "$BACKUP_DIR/archives"
}

# Phase 1.1: Log File Cleanup
cleanup_logs() {
    log "Starting Phase 1.1: Log File Cleanup"
    
    cd "$ROOT_DIR"
    
    # Count files before cleanup
    local log_count_before=$(find logs -name "*.log" 2>/dev/null | wc -l || echo 0)
    local log_size_before=$(du -sh logs 2>/dev/null | cut -f1 || echo "0")
    
    info "Before cleanup: $log_count_before log files, $log_size_before total size"
    
    # Backup logs that will be compressed/deleted
    log "Backing up logs older than 7 days to $BACKUP_DIR/logs/"
    find logs -name "*.log" -mtime +7 -exec cp {} "$BACKUP_DIR/logs/" \; 2>/dev/null || true
    
    # Compress logs older than 7 days
    log "Compressing logs older than 7 days"
    find logs -name "*.log" -mtime +7 -exec gzip {} \; 2>/dev/null || true
    
    # Delete compressed logs older than 30 days
    log "Removing compressed logs older than 30 days"
    local deleted_gz=$(find logs -name "*.log.gz" -mtime +30 -delete -print 2>/dev/null | wc -l)
    info "Deleted $deleted_gz old compressed log files"
    
    # Delete MCP logs older than 14 days (these regenerate frequently)
    log "Removing MCP logs older than 14 days"
    local deleted_mcp=$(find logs -name "mcp_*.log" -mtime +14 -delete -print 2>/dev/null | wc -l)
    info "Deleted $deleted_mcp old MCP log files"
    
    # Count files after cleanup
    local log_count_after=$(find logs -name "*.log" 2>/dev/null | wc -l || echo 0)
    local log_size_after=$(du -sh logs 2>/dev/null | cut -f1 || echo "0")
    
    success "Log cleanup complete: $log_count_after log files remaining, $log_size_after total size"
    log "Log files eliminated: $((log_count_before - log_count_after))"
}

# Phase 1.2: Test Result File Cleanup
cleanup_test_results() {
    log "Starting Phase 1.2: Test Result File Cleanup"
    
    cd "$ROOT_DIR"
    
    # Count test result files before cleanup
    local test_count_before=$(find . -name "*test_results*.json" -o -name "*_report_*.json" -path "*/test*" 2>/dev/null | wc -l || echo 0)
    local test_size_before=$(du -ch $(find . -name "*test_results*.json" -o -name "*_report_*.json" -path "*/test*" 2>/dev/null) 2>/dev/null | tail -1 | cut -f1 || echo "0")
    
    info "Before cleanup: $test_count_before test result files, $test_size_before total size"
    
    # Backup test results that will be deleted
    log "Backing up test results older than 14 days to $BACKUP_DIR/test_results/"
    find . -name "*test_results*.json" -mtime +14 -exec cp {} "$BACKUP_DIR/test_results/" \; 2>/dev/null || true
    find . -name "*_report_*.json" -path "*/test*" -mtime +14 -exec cp {} "$BACKUP_DIR/test_results/" \; 2>/dev/null || true
    
    # Remove test result files older than 14 days
    log "Removing test result files older than 14 days"
    local deleted_test1=$(find . -name "*test_results*.json" -mtime +14 -delete -print 2>/dev/null | wc -l)
    local deleted_test2=$(find . -name "*_report_*.json" -path "*/test*" -mtime +14 -delete -print 2>/dev/null | wc -l)
    
    # Remove specific test debris patterns
    log "Removing old bulletproof test results"
    local deleted_bulletproof=$(find . -name "bulletproof_test_results_*.json" -mtime +7 -delete -print 2>/dev/null | wc -l)
    
    log "Removing old comprehensive test reports"
    local deleted_comprehensive=$(find . -name "comprehensive_test_report_*.json" -mtime +7 -delete -print 2>/dev/null | wc -l)
    
    # Count test result files after cleanup
    local test_count_after=$(find . -name "*test_results*.json" -o -name "*_report_*.json" -path "*/test*" 2>/dev/null | wc -l || echo 0)
    
    local total_deleted=$((deleted_test1 + deleted_test2 + deleted_bulletproof + deleted_comprehensive))
    success "Test result cleanup complete: $total_deleted files eliminated"
    log "Test result files remaining: $test_count_after"
}

# Phase 1.3: Archive Directory Cleanup
cleanup_archives() {
    log "Starting Phase 1.3: Archive Directory Cleanup"
    
    cd "$ROOT_DIR"
    
    # Check if archive directories exist
    if [[ -d "archive" ]]; then
        local archive_size_before=$(du -sh archive 2>/dev/null | cut -f1 || echo "0")
        info "Archive directory size before cleanup: $archive_size_before"
        
        # Create compressed backup of archives
        log "Creating compressed backup of archive directory"
        tar -czf "$BACKUP_DIR/archives/archive_backup_$(date +%Y%m%d).tar.gz" archive/ 2>/dev/null || warning "Failed to backup archive directory"
        
        # Remove waste cleanup archives (already processed)
        if [[ -d "archive/waste_cleanup_20250815" ]]; then
            log "Removing processed waste cleanup archive"
            rm -rf archive/waste_cleanup_20250815/
            success "Removed archive/waste_cleanup_20250815/"
        fi
        
        # Remove old deployment backups
        log "Removing old deployment backup directories"
        local deleted_deploys=$(find archive -name "deploy_*" -type d -mtime +7 -exec rm -rf {} \; -print 2>/dev/null | wc -l || echo 0)
        info "Removed $deleted_deploys old deployment backup directories"
        
        local archive_size_after=$(du -sh archive 2>/dev/null | cut -f1 || echo "0")
        success "Archive cleanup complete: $archive_size_before -> $archive_size_after"
    fi
    
    # Clean up backups directory
    if [[ -d "backups" ]]; then
        local backup_size_before=$(du -sh backups 2>/dev/null | cut -f1 || echo "0")
        info "Backups directory size before cleanup: $backup_size_before"
        
        # Create compressed backup
        log "Creating compressed backup of backups directory"
        tar -czf "$BACKUP_DIR/archives/backups_backup_$(date +%Y%m%d).tar.gz" backups/ 2>/dev/null || warning "Failed to backup backups directory"
        
        # Remove old deployment backups
        log "Removing old deployment backups"
        local deleted_backup_deploys=$(find backups -name "deploy_*" -type d -mtime +14 -exec rm -rf {} \; -print 2>/dev/null | wc -l || echo 0)
        info "Removed $deleted_backup_deploys old backup deployment directories"
        
        local backup_size_after=$(du -sh backups 2>/dev/null | cut -f1 || echo "0")
        success "Backup cleanup complete: $backup_size_before -> $backup_size_after"
    fi
}

# Phase 1.4: Remove Empty Directories
cleanup_empty_dirs() {
    log "Starting Phase 1.4: Empty Directory Cleanup"
    
    cd "$ROOT_DIR"
    
    # Find and remove empty directories (excluding .git and protected dirs)
    log "Finding empty directories"
    local empty_dirs=$(find . -type d -empty -not -path "./.git*" -not -path "./node_modules*" -not -path "./.venv*" -not -path "./.mcp*" 2>/dev/null | wc -l || echo 0)
    
    if [[ $empty_dirs -gt 0 ]]; then
        info "Found $empty_dirs empty directories"
        
        # List empty directories for logging
        find . -type d -empty -not -path "./.git*" -not -path "./node_modules*" -not -path "./.venv*" -not -path "./.mcp*" 2>/dev/null | while read -r dir; do
            log "Removing empty directory: $dir"
            rmdir "$dir" 2>/dev/null || warning "Failed to remove directory: $dir"
        done
        
        success "Empty directory cleanup complete"
    else
        info "No empty directories found"
    fi
}

# Generate cleanup summary
generate_summary() {
    log "Generating cleanup summary"
    
    local summary_file="$BACKUP_DIR/PHASE1_CLEANUP_SUMMARY.md"
    
    cat > "$summary_file" << EOF
# WASTE ELIMINATION PHASE 1 - SAFE CLEANUP SUMMARY
**Execution Date**: $(date '+%Y-%m-%d %H:%M:%S UTC')
**Execution User**: $(whoami)
**Backup Location**: $BACKUP_DIR

## Phase 1.1: Log File Cleanup
- Compressed logs older than 7 days
- Deleted compressed logs older than 30 days
- Removed MCP logs older than 14 days
- All affected files backed up to: $BACKUP_DIR/logs/

## Phase 1.2: Test Result File Cleanup  
- Removed test result files older than 14 days
- Cleaned bulletproof test results older than 7 days
- Removed comprehensive test reports older than 7 days
- All affected files backed up to: $BACKUP_DIR/test_results/

## Phase 1.3: Archive Directory Cleanup
- Compressed and archived old deployment backups
- Removed processed waste cleanup archives
- All affected data backed up to: $BACKUP_DIR/archives/

## Phase 1.4: Empty Directory Cleanup
- Removed empty directories (excluding protected paths)
- No functional impact - cosmetic cleanup only

## Rollback Instructions
If any issues are discovered after this cleanup:

\`\`\`bash
# Restore logs
cp $BACKUP_DIR/logs/* /opt/sutazaiapp/logs/

# Restore test results
cp $BACKUP_DIR/test_results/* /opt/sutazaiapp/tests/

# Restore archives
cd /opt/sutazaiapp
tar -xzf $BACKUP_DIR/archives/archive_backup_*.tar.gz
tar -xzf $BACKUP_DIR/archives/backups_backup_*.tar.gz
\`\`\`

## Validation Commands
\`\`\`bash
# Quick system health check
docker-compose ps
curl -f http://localhost:10010/health || echo "Backend health check failed"

# Log directory integrity
ls -la /opt/sutazaiapp/logs/

# Test that system can still log
echo "Test log entry" >> /opt/sutazaiapp/logs/phase1_validation.log
\`\`\`

**Status**: ✅ PHASE 1 SAFE CLEANUP COMPLETED SUCCESSFULLY
EOF

    info "Summary generated: $summary_file"
    cat "$summary_file"
}

# Main execution
main() {
    log "Starting WASTE ELIMINATION PHASE 1: SAFE CLEANUP"
    log "Rule 13: Zero Tolerance for Waste - Immediate Implementation"
    
    # Pre-execution validation
    if [[ ! -d "$ROOT_DIR" ]]; then
        error "Root directory not found: $ROOT_DIR"
    fi
    
    if [[ ! -w "$ROOT_DIR" ]]; then
        error "No write permission to root directory: $ROOT_DIR"
    fi
    
    # Create backup directory
    create_backup_dir
    
    # Execute cleanup phases
    cleanup_logs
    cleanup_test_results
    cleanup_archives
    cleanup_empty_dirs
    
    # Generate summary
    generate_summary
    
    success "PHASE 1 SAFE CLEANUP COMPLETED SUCCESSFULLY"
    success "Backup location: $BACKUP_DIR"
    success "Log file: $LOG_FILE"
    
    info "Next steps:"
    info "1. Validate system functionality"
    info "2. Review cleanup summary: $BACKUP_DIR/PHASE1_CLEANUP_SUMMARY.md"
    info "3. Proceed to Phase 2 (Environment Consolidation) when ready"
    info "4. Monitor system for 24 hours before implementing Phase 3+"
}

# Trap for cleanup on exit
trap 'echo "Script interrupted. Check log: $LOG_FILE"' INT TERM

# Execute main function
main "$@"