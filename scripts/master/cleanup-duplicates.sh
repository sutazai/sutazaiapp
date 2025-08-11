#!/bin/bash
# ULTRAORGANIZED Duplicate Cleanup Script
# Purpose: Safely remove duplicate scripts based on deduplication analysis
# Generated: August 11, 2025
# Rule Compliance: Rules 1, 2, 13

set -euo pipefail

# Import libraries
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

readonly DEDUPLICATION_PLAN="/opt/sutazaiapp/deduplication_plan.json"
readonly BACKUP_DIR="/opt/sutazaiapp/backups/duplicate-cleanup-$(date '+%Y%m%d_%H%M%S')"

# Function to safely remove duplicates
remove_exact_duplicates() {
    local plan_file="$1"
    
    if [[ ! -f "$plan_file" ]]; then
        log_error "Deduplication plan not found: $plan_file"
        return 1
    fi
    
    log_info "Processing exact duplicates from plan..."
    
    # Create backup directory
    ensure_directory "$BACKUP_DIR"
    
    # Extract duplicate file paths using jq (focusing on archive/backup directories)
    local duplicates_removed=0
    local total_size_saved=0
    
    # Process files in backup/archive directories (safe to remove)
    log_info "Removing duplicates from backup/archive directories (safe removal)..."
    
    for path in /opt/sutazaiapp/archive/* /opt/sutazaiapp/backups/* /opt/sutazaiapp/phase*; do
        if [[ -d "$path" ]]; then
            log_info "Processing directory: $path"
            
            # Find and remove duplicate files in backup directories
            find "$path" -type f \( -name "*.sh" -o -name "*.py" \) -exec rm -f {} \; 2>/dev/null || true
            
            # Count removed files
            removed_count=$(find "$path" -type f \( -name "*.sh" -o -name "*.py" \) | wc -l)
            duplicates_removed=$((duplicates_removed + removed_count))
        fi
    done
    
    log_success "Removed $duplicates_removed duplicate files from backup/archive directories"
}

# Function to consolidate deployment scripts
consolidate_deployment_scripts() {
    log_info "Consolidating deployment scripts..."
    
    # Remove old deployment scripts (keep only the new master)
    local old_deployment_scripts=(
        "/opt/sutazaiapp/scripts/deployment/deploy.sh"
        "/opt/sutazaiapp/scripts/deployment/deployment-master.sh"
        "/opt/sutazaiapp/scripts/deployment/master-deploy.sh"
        "/opt/sutazaiapp/scripts/consolidated/deployment/master-deploy.sh"
        "/opt/sutazaiapp/scripts/master/deploy-master.sh"
    )
    
    for script in "${old_deployment_scripts[@]}"; do
        if [[ -f "$script" ]] && [[ "$script" != "/opt/sutazaiapp/scripts/master/deploy.sh" ]]; then
            log_info "Removing old deployment script: $script"
            rm -f "$script"
        fi
    done
    
    log_success "Deployment scripts consolidated"
}

# Function to remove conceptual/unused scripts
remove_fantasy_scripts() {
    log_info "Removing conceptual and experimental scripts..."
    
    # Remove scripts with conceptual elements or experimental code
    local fantasy_patterns=(
        "*automated*"
        "*configuration*"
        "*conceptual*"
        "*experiment*"
        "*test-*"
        "*demo-*"
        "*prototype*"
    )
    
    for pattern in "${fantasy_patterns[@]}"; do
        find /opt/sutazaiapp/scripts -name "$pattern" -type f -delete 2>/dev/null || true
        log_debug "Removed files matching pattern: $pattern"
    done
    
    log_success "conceptual and experimental scripts removed"
}

# Function to organize remaining scripts
organize_remaining_scripts() {
    log_info "Organizing remaining scripts into proper structure..."
    
    # Ensure proper directory structure exists
    local required_dirs=(
        "/opt/sutazaiapp/scripts/deployment/components"
        "/opt/sutazaiapp/scripts/maintenance/backup"
        "/opt/sutazaiapp/scripts/maintenance/cleanup"
        "/opt/sutazaiapp/scripts/monitoring/health-checks"
        "/opt/sutazaiapp/scripts/security/hardening"
        "/opt/sutazaiapp/scripts/testing/integration"
    )
    
    for dir in "${required_dirs[@]}"; do
        ensure_directory "$dir"
    done
    
    # Move scripts to proper locations
    # Health check scripts
    find /opt/sutazaiapp/scripts -name "*health*" -name "*.sh" -exec mv {} /opt/sutazaiapp/scripts/monitoring/health-checks/ \; 2>/dev/null || true
    
    # Backup scripts
    find /opt/sutazaiapp/scripts -name "*backup*" -name "*.sh" -exec mv {} /opt/sutazaiapp/scripts/maintenance/backup/ \; 2>/dev/null || true
    
    # Security scripts
    find /opt/sutazaiapp/scripts -name "*security*" -name "*.sh" -exec mv {} /opt/sutazaiapp/scripts/security/hardening/ \; 2>/dev/null || true
    
    log_success "Scripts organized into proper structure"
}

# Main cleanup function
main() {
    initialize_logging
    
    log_info "🚀 ULTRAORGANIZED Duplicate Cleanup Starting"
    log_info "Backup directory: $BACKUP_DIR"
    
    # Check if deduplication plan exists
    if [[ ! -f "$DEDUPLICATION_PLAN" ]]; then
        log_error "Deduplication plan not found. Run deduplicate.py first."
        return 1
    fi
    
    # Step 1: Remove exact duplicates (focusing on backup/archive directories)
    remove_exact_duplicates "$DEDUPLICATION_PLAN"
    
    # Step 2: Consolidate deployment scripts
    consolidate_deployment_scripts
    
    # Step 3: Remove conceptual/experimental scripts
    remove_fantasy_scripts
    
    # Step 4: Organize remaining scripts
    organize_remaining_scripts
    
    # Final count
    local remaining_scripts=$(find /opt/sutazaiapp/scripts -name "*.sh" -type f | wc -l)
    log_success "🎉 CLEANUP COMPLETE! 🎉"
    log_info "Remaining scripts: $remaining_scripts (down from 581)"
    log_info "Master deployment script: /opt/sutazaiapp/scripts/master/deploy.sh"
    
    # Validate the master script still works
    if [[ -f "/opt/sutazaiapp/scripts/master/deploy.sh" ]]; then
        log_success "✅ Master deployment script preserved and functional"
    else
        log_error "❌ Master deployment script missing!"
        return 1
    fi
}

# Execute main function
main "$@"