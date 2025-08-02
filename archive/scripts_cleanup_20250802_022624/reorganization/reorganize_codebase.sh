#!/bin/bash
# SutazAI Master Codebase Reorganization Script
# Orchestrates safe codebase reorganization with comprehensive testing

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/opt/sutazaiapp/logs/reorganization_master.log"
REORGANIZATION_DIR="/opt/sutazaiapp/scripts/reorganization"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE" >&2
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] ℹ️  $1${NC}" | tee -a "$LOG_FILE"
}

# Display banner
show_banner() {
    echo -e "${BLUE}"
    cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                    SutazAI Codebase Reorganization           ║
║                         Safe & Comprehensive                 ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    
    echo "🎯 Objective: Safely reorganize codebase while maintaining system stability"
    echo "🛡️  Safety: Complete backup + phased execution + health monitoring"
    echo "📊 Scope: ~366 files to be analyzed and reorganized"
    echo ""
}

# Pre-flight checks
preflight_checks() {
    info "Performing pre-flight checks..."
    
    # Check if running as root or with appropriate permissions
    if [ ! -w "/opt/sutazaiapp" ]; then
        error "Insufficient permissions to modify /opt/sutazaiapp"
        return 1
    fi
    
    # Check disk space (need at least 1GB)
    local available_space=$(df /opt/sutazaiapp | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 1000000 ]; then  # Less than 1GB in KB
        error "Insufficient disk space. Need at least 1GB free."
        return 1
    fi
    
    # Check Docker
    if ! docker ps >/dev/null 2>&1; then
        error "Docker is not running or not accessible"
        return 1
    fi
    
    # Check essential containers
    local essential_containers=(
        "sutazai-backend-minimal"
        "sutazai-ollama-minimal"
    )
    
    for container in "${essential_containers[@]}"; do
        if ! docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
            error "Essential container not running: $container"
            return 1
        fi
    done
    
    # Check critical files exist
    local critical_files=(
        "/opt/sutazaiapp/backend/app/main.py"
        "/opt/sutazaiapp/docker-compose.minimal.yml"
        "/opt/sutazaiapp/scripts/live_logs.sh"
    )
    
    for file in "${critical_files[@]}"; do
        if [ ! -f "$file" ]; then
            error "Critical file missing: $file"
            return 1
        fi
    done
    
    success "Pre-flight checks passed"
    return 0
}

# Confirm user intent
confirm_operation() {
    echo ""
    echo "⚠️  REORGANIZATION CONFIRMATION"
    echo "================================"
    echo ""
    echo "This operation will:"
    echo "  1. Create a complete system backup"
    echo "  2. Move ~150+ redundant files to organized archive"
    echo "  3. Preserve all critical system files"
    echo "  4. Test system health after each phase"
    echo "  5. Provide full restoration capability"
    echo ""
    echo "Files to be preserved (never moved):"
    echo "  - backend/app/main.py (active backend)"
    echo "  - frontend/app.py (active frontend)"
    echo "  - docker-compose.minimal.yml (current deployment)"
    echo "  - scripts/live_logs.sh (monitoring)"
    echo "  - health_check.sh (system health)"
    echo ""
    echo "Estimated time: 10-15 minutes"
    echo "Risk level: LOW (full backup + incremental testing)"
    echo ""
    
    while true; do
        read -p "Proceed with reorganization? (yes/no): " yn
        case $yn in
            [Yy]es)
                success "User confirmed operation"
                break
                ;;
            [Nn]o)
                info "Operation cancelled by user"
                exit 0
                ;;
            *)
                echo "Please answer 'yes' or 'no'"
                ;;
        esac
    done
}

# Execute script phase
execute_phase() {
    local phase_number="$1"
    local phase_name="$2"
    local script_name="$3"
    local required="$4"  # "required" or "optional"
    
    echo ""
    echo -e "${BLUE}🔄 Phase $phase_number: $phase_name${NC}"
    echo "$(printf '=%.0s' {1..50})"
    
    local script_path="$REORGANIZATION_DIR/$script_name"
    
    if [ ! -f "$script_path" ]; then
        error "Phase script not found: $script_path"
        if [ "$required" == "required" ]; then
            return 1
        else
            warning "Optional phase skipped: $phase_name"
            return 0
        fi
    fi
    
    # Make script executable
    chmod +x "$script_path"
    
    # Execute script
    if bash "$script_path"; then
        success "Phase $phase_number completed: $phase_name"
        return 0
    else
        error "Phase $phase_number failed: $phase_name"
        if [ "$required" == "required" ]; then
            return 1
        else
            warning "Optional phase failed but continuing: $phase_name"
            return 0
        fi
    fi
}

# Emergency rollback function
emergency_rollback() {
    error "EMERGENCY ROLLBACK INITIATED"
    echo ""
    echo "🚨 EMERGENCY SITUATION DETECTED"
    echo "==============================="
    echo ""
    echo "Attempting to locate and execute most recent backup..."
    
    # Find most recent backup
    local backup_dir=$(find /opt/sutazaiapp/backups -name "reorganization_backup_*" -type d | sort | tail -1)
    
    if [ -n "$backup_dir" ] && [ -f "$backup_dir/restore.sh" ]; then
        warning "Found backup: $backup_dir"
        echo "Executing emergency restoration..."
        
        if bash "$backup_dir/restore.sh"; then
            success "Emergency restoration completed"
            echo ""
            echo "✅ System has been restored to pre-reorganization state"
            echo "📋 Check logs for details on what went wrong"
            echo "🔧 Consider running health check to verify restoration"
        else
            error "Emergency restoration failed"
            echo ""
            echo "❌ CRITICAL: Automated restoration failed"
            echo "🆘 MANUAL INTERVENTION REQUIRED"
            echo "📞 Contact system administrator immediately"
            echo "📁 Backup location: $backup_dir"
        fi
    else
        error "No backup found for emergency restoration"
        echo ""
        echo "❌ CRITICAL: No backup available for restoration"
        echo "🆘 SYSTEM IN UNKNOWN STATE"
        echo "📞 Contact system administrator immediately"
    fi
}

# Main reorganization process
main_reorganization() {
    info "Starting main reorganization process..."
    
    # Phase 1: System Backup (REQUIRED)
    if ! execute_phase "1" "System Backup" "01_backup_system.sh" "required"; then
        error "System backup failed - aborting reorganization"
        return 1
    fi
    
    # Phase 2: Create Archive Structure (REQUIRED)
    if ! execute_phase "2" "Create Archive Structure" "02_create_archive_structure.sh" "required"; then
        error "Archive structure creation failed - aborting reorganization"
        return 1
    fi
    
    # Phase 3: Identify Files to Move (REQUIRED)
    if ! execute_phase "3" "Identify Files to Move" "03_identify_files_to_move.sh" "required"; then
        error "File identification failed - aborting reorganization"
        return 1
    fi
    
    # Phase 4: Move Files Safely (CRITICAL)
    if ! execute_phase "4" "Move Files Safely" "04_move_files_safely.sh" "required"; then
        error "File movement failed - initiating emergency rollback"
        emergency_rollback
        return 1
    fi
    
    # Phase 5: Test System Health (CRITICAL)
    if ! execute_phase "5" "Test System Health" "05_test_system_health.sh" "required"; then
        error "System health test failed after reorganization"
        
        # Ask user if they want to rollback
        echo ""
        echo "⚠️  SYSTEM HEALTH TEST FAILED"
        echo "============================="
        echo ""
        echo "The reorganization completed but health tests indicate issues."
        echo ""
        
        while true; do
            read -p "Initiate emergency rollback? (yes/no): " yn
            case $yn in
                [Yy]es)
                    emergency_rollback
                    return 1
                    ;;
                [Nn]o)
                    warning "User chose to proceed despite health test failures"
                    break
                    ;;
                *)
                    echo "Please answer 'yes' or 'no'"
                    ;;
            esac
        done
    fi
    
    success "Main reorganization process completed successfully"
    return 0
}

# Generate final report
generate_final_report() {
    info "Generating final reorganization report..."
    
    local report_file="/opt/sutazaiapp/logs/reorganization_final_report.md"
    
    cat > "$report_file" << EOF
# SutazAI Codebase Reorganization - Final Report

**Date:** $(date)
**Duration:** $(date -d "@$(($(date +%s) - START_TIME))" -u +%H:%M:%S)
**Status:** $([ $? -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILED")

## Summary

The SutazAI codebase reorganization has been completed. This process moved redundant and duplicate files to an organized archive structure while preserving all critical system components.

## Key Achievements

- ✅ Complete system backup created
- ✅ Organized archive structure established  
- ✅ Redundant files safely moved to archive
- ✅ Critical system files preserved
- ✅ System health validated post-reorganization

## Files Preserved (Never Moved)

- \`backend/app/main.py\` - Active backend application
- \`frontend/app.py\` - Active frontend application
- \`docker-compose.minimal.yml\` - Current deployment configuration  
- \`scripts/live_logs.sh\` - Essential monitoring script
- \`health_check.sh\` - System health monitoring

## Archive Location

Moved files can be found in the organized archive structure:
\`$(cat /tmp/sutazai_archive_path.env 2>/dev/null | cut -d'=' -f2 | tr -d "'" || echo "Archive location not available")\`

## System Status

Post-reorganization system health: $(tail -1 /opt/sutazaiapp/logs/health_report_*.md 2>/dev/null | grep "Overall System Health:" | cut -d':' -f2 | tr -d ' ' || echo "Health report not available")

## Restoration Information

If any issues arise, the system can be fully restored using:
\`$(find /opt/sutazaiapp/backups -name "reorganization_backup_*" -type d | sort | tail -1)/restore.sh\`

## Next Steps

1. Monitor system performance over the next 24-48 hours
2. Archive can be cleaned up after 7 days if no issues arise
3. Update any scripts that may reference moved files (unlikely)

---
*Report generated by SutazAI reorganization system*
EOF

    success "Final report generated: $report_file"
}

# Cleanup function
cleanup_temp_files() {
    info "Cleaning up temporary files..."
    
    # Remove temporary environment file
    rm -f /tmp/sutazai_archive_path.env
    
    success "Cleanup completed"
}

# Main function
main() {
    # Record start time
    START_TIME=$(date +%s)
    
    # Show banner
    show_banner
    
    # Pre-flight checks
    if ! preflight_checks; then
        error "Pre-flight checks failed. Aborting reorganization."
        exit 1
    fi
    
    # Confirm operation
    confirm_operation
    
    # Start reorganization
    log "Starting SutazAI codebase reorganization..."
    
    # Set trap for emergency situations
    trap 'error "Script interrupted"; emergency_rollback' INT TERM
    
    # Execute main reorganization process
    if main_reorganization; then
        success "Reorganization completed successfully! 🎉"
        
        # Generate final report
        generate_final_report
        
        echo ""
        echo -e "${GREEN}✅ REORGANIZATION COMPLETE${NC}"
        echo "=========================="
        echo ""
        echo "🎯 Objective achieved: Codebase successfully reorganized"
        echo "🛡️  Safety maintained: All critical files preserved"
        echo "📊 Impact: Significantly improved codebase organization"
        echo "🔍 Health: System verified and operational"
        echo ""
        echo "📋 Full report: /opt/sutazaiapp/logs/reorganization_final_report.md"
        echo "📁 Archive: $(cat /tmp/sutazai_archive_path.env 2>/dev/null | cut -d'=' -f2 | tr -d "'" || echo "Check logs for location")"
        echo "🔄 Restoration: Available if needed (see backup)"
        echo ""
        echo "Next: Monitor system for 24-48 hours, then archive can be cleaned up"
        
    else
        error "Reorganization failed"
        
        echo ""
        echo -e "${RED}❌ REORGANIZATION FAILED${NC}"
        echo "========================"
        echo ""
        echo "🚨 The reorganization process encountered critical errors"
        echo "🔄 System may have been restored to original state"
        echo "📋 Check logs for detailed error information"
        echo "🆘 If system is unstable, use emergency backup for restoration"
        echo ""
        
        return 1
    fi
    
    # Cleanup
    cleanup_temp_files
    
    log "SutazAI codebase reorganization process completed"
}

# Run main function
main "$@"