#!/bin/bash
# SutazAI Safe File Movement Script
# Moves files in phases with system health monitoring

set -euo pipefail

# Configuration
ANALYSIS_DIR="/opt/sutazaiapp/scripts/reorganization/analysis"
LOG_FILE="/opt/sutazaiapp/logs/reorganization.log"
MOVEMENT_LOG="/opt/sutazaiapp/logs/file_movements.log"

# Load archive path
if [ -f /tmp/sutazai_archive_path.env ]; then
    source /tmp/sutazai_archive_path.env
else
    error "Archive path not found. Run create_archive_structure.sh first."
    exit 1
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE" >&2
}

movement_log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S')|$1|$2|$3|$4" >> "$MOVEMENT_LOG"
    log "MOVED: $1 -> $2 ($3)"
}

# Test system health
test_system_health() {
    log "Testing system health..."
    
    # Test Docker
    if ! docker ps >/dev/null 2>&1; then
        error "Docker is not running"
        return 1
    fi
    
    # Test essential services
    local essential_containers=(
        "sutazai-backend-minimal"
        "sutazai-ollama-minimal" 
        "sutazai-postgres-minimal"
        "sutazai-redis-minimal"
    )
    
    for container in "${essential_containers[@]}"; do
        if ! docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
            error "Essential container not running: $container"
            return 1
        fi
    done
    
    # Test backend health endpoint
    if ! curl -sf "http://localhost:8000/health" >/dev/null 2>&1; then
        error "Backend health check failed"
        return 1
    fi
    
    log "System health check passed ✅"
    return 0
}

# Move single file safely
move_file_safely() {
    local source_file="$1"
    local archive_path="$2"
    local reason="$3"
    
    # Validate source file exists
    if [ ! -f "$source_file" ]; then
        log "Warning: Source file not found: $source_file"
        return 1
    fi
    
    # Create destination directory
    local full_archive_path="$SUTAZAI_ARCHIVE_ROOT/$archive_path"
    mkdir -p "$(dirname "$full_archive_path")"
    
    # Move file with timestamp preservation
    if mv "$source_file" "$full_archive_path"; then
        movement_log "$source_file" "$full_archive_path" "$reason" "SUCCESS"
        return 0
    else
        error "Failed to move: $source_file"
        movement_log "$source_file" "$full_archive_path" "$reason" "FAILED"
        return 1
    fi
}

# Phase 1: Move duplicate monitoring scripts
move_phase_1_monitoring() {
    log "Phase 1: Moving duplicate monitoring scripts..."
    
    local files_to_move=(
        "scripts/static_monitor.py|duplicates/monitoring/system_monitors/static_monitor.py|Duplicate monitoring functionality"
        "scripts/compact_monitor.py|duplicates/monitoring/system_monitors/compact_monitor.py|Duplicate monitoring functionality"
        "scripts/emergency_resource_monitor.py|duplicates/monitoring/system_monitors/emergency_resource_monitor.py|Duplicate monitoring functionality"
        "scripts/memory_monitor_dashboard.py|duplicates/monitoring/dashboard_scripts/memory_monitor_dashboard.py|Duplicate dashboard functionality"
        "scripts/monitoring_system_monitor.py|duplicates/monitoring/system_monitors/monitoring_system_monitor.py|Duplicate monitoring functionality"
        "scripts/health-monitor.py|duplicates/monitoring/health_checks/health-monitor.py|Duplicate health check functionality"
        "scripts/live_logs.sh.backup|duplicates/monitoring/log_processors/live_logs.sh.backup|Backup version of active script"
        "scripts/live_logs_clean.sh|duplicates/monitoring/log_processors/live_logs_clean.sh|Alternative version of active script"
    )
    
    local moved=0
    local failed=0
    
    for file_spec in "${files_to_move[@]}"; do
        IFS='|' read -r source_file archive_path reason <<< "$file_spec"
        
        if move_file_safely "/opt/sutazaiapp/$source_file" "$archive_path" "$reason"; then
            ((moved++))
        else
            ((failed++))
        fi
    done
    
    log "Phase 1 completed: $moved moved, $failed failed"
    
    # Test system health after phase 1
    if ! test_system_health; then
        error "System health check failed after Phase 1"
        return 1
    fi
    
    return 0
}

# Phase 2: Move duplicate deployment scripts
move_phase_2_deployment() {
    log "Phase 2: Moving duplicate deployment scripts..."
    
    local files_to_move=(
        "scripts/deploy_complete_system.sh.backup_before_cleanup|duplicates/deployment/backup_deployments/deploy_complete_system.sh.backup_before_cleanup|Backup deployment script"
        "scripts/deploy_complete_system.sh.backup_before_major_cleanup|duplicates/deployment/backup_deployments/deploy_complete_system.sh.backup_before_major_cleanup|Backup deployment script"
        "scripts/deploy_all.sh|duplicates/deployment/old_deploy_scripts/deploy_all.sh|Duplicate deployment functionality"
        "scripts/deploy_all_agents.sh|duplicates/deployment/old_deploy_scripts/deploy_all_agents.sh|Specific deployment case"
        "scripts/deploy_agi_system.sh|duplicates/deployment/old_deploy_scripts/deploy_agi_system.sh|Duplicate deployment functionality"
        "scripts/fixed_deploy.sh|duplicates/deployment/old_deploy_scripts/fixed_deploy.sh|Fixed version, now redundant"
        "scripts/deploy_task_automation.sh|duplicates/deployment/old_deploy_scripts/deploy_task_automation.sh|Specific deployment case"
        "scripts/deploy_taskmaster_integrated_system.sh|duplicates/deployment/old_deploy_scripts/deploy_taskmaster_integrated_system.sh|Specific deployment case"
    )
    
    local moved=0
    local failed=0
    
    for file_spec in "${files_to_move[@]}"; do
        IFS='|' read -r source_file archive_path reason <<< "$file_spec"
        
        if move_file_safely "/opt/sutazaiapp/$source_file" "$archive_path" "$reason"; then
            ((moved++))
        else
            ((failed++))
        fi
    done
    
    log "Phase 2 completed: $moved moved, $failed failed"
    
    # Test system health after phase 2
    if ! test_system_health; then
        error "System health check failed after Phase 2"
        return 1
    fi
    
    return 0
}

# Phase 3: Move duplicate testing scripts
move_phase_3_testing() {
    log "Phase 3: Moving duplicate testing scripts..."
    
    local files_to_move=(
        "scripts/test_brain_deployment.sh|duplicates/testing/old_test_scripts/test_brain_deployment.sh|Duplicate brain testing"
        "scripts/test_brain_deployment_explicit.sh|duplicates/testing/old_test_scripts/test_brain_deployment_explicit.sh|Duplicate brain testing"
        "scripts/test_brain_direct.sh|duplicates/testing/old_test_scripts/test_brain_direct.sh|Duplicate brain testing"
        "scripts/test_brain_final.sh|duplicates/testing/old_test_scripts/test_brain_final.sh|Duplicate brain testing"
        "scripts/test_brain_functions.sh|duplicates/testing/old_test_scripts/test_brain_functions.sh|Duplicate brain testing"
        "scripts/test_brain_simple.sh|duplicates/testing/old_test_scripts/test_brain_simple.sh|Duplicate brain testing"
        "scripts/test_docker_brain.sh|duplicates/testing/old_test_scripts/test_docker_brain.sh|Duplicate brain testing"
        "scripts/test_enhanced_brain.sh|duplicates/testing/old_test_scripts/test_enhanced_brain.sh|Duplicate brain testing"
        "scripts/test_deployment_system.sh|duplicates/testing/old_test_scripts/test_deployment_system.sh|Duplicate deployment testing"
        "scripts/test_deployment_with_brain.sh|duplicates/testing/old_test_scripts/test_deployment_with_brain.sh|Duplicate deployment testing"
        "scripts/test_final_deployment.sh|duplicates/testing/old_test_scripts/test_final_deployment.sh|Duplicate deployment testing"
        "scripts/test_fixed_deployment.sh|duplicates/testing/old_test_scripts/test_fixed_deployment.sh|Duplicate deployment testing"
    )
    
    local moved=0
    local failed=0
    
    for file_spec in "${files_to_move[@]}"; do
        IFS='|' read -r source_file archive_path reason <<< "$file_spec"
        
        if move_file_safely "/opt/sutazaiapp/$source_file" "$archive_path" "$reason"; then
            ((moved++))
        else
            ((failed++))
        fi
    done
    
    log "Phase 3 completed: $moved moved, $failed failed"
    
    # Test system health after phase 3
    if ! test_system_health; then
        error "System health check failed after Phase 3"
        return 1
    fi
    
    return 0
}

# Phase 4: Move obsolete configurations  
move_phase_4_configs() {
    log "Phase 4: Moving obsolete configurations..."
    
    local files_to_move=(
        "docker-compose-agents-complete.yml|obsolete/old_versions/docker-compose-agents-complete.yml|Old Docker Compose version"
        "docker-compose.yml|obsolete/old_versions/docker-compose.yml|Large unused Docker Compose file"
    )
    
    # Only move config/docker/docker-compose.tinyllama.yml if it's not a symlink
    if [ -f "/opt/sutazaiapp/config/docker/docker-compose.tinyllama.yml" ] && [ ! -L "/opt/sutazaiapp/config/docker/docker-compose.tinyllama.yml" ]; then
        files_to_move+=("config/docker/docker-compose.tinyllama.yml|obsolete/deprecated/docker-compose.tinyllama.yml|Duplicate configuration")
    fi
    
    local moved=0
    local failed=0
    
    for file_spec in "${files_to_move[@]}"; do
        IFS='|' read -r source_file archive_path reason <<< "$file_spec"
        
        if move_file_safely "/opt/sutazaiapp/$source_file" "$archive_path" "$reason"; then
            ((moved++))
        else
            ((failed++))
        fi
    done
    
    log "Phase 4 completed: $moved moved, $failed failed"
    
    # Test system health after phase 4
    if ! test_system_health; then
        error "System health check failed after Phase 4"
        return 1
    fi
    
    return 0
}

# Phase 5: Move redundant utilities
move_phase_5_utilities() {
    log "Phase 5: Moving redundant utilities..."
    
    local files_to_move=(
        "scripts/cleanup_redundancies.sh|redundant/utilities/cleanup_scripts/cleanup_redundancies.sh|Similar to cleanup_cache.sh"
        "scripts/cleanup_fantasy_elements.sh|redundant/utilities/cleanup_scripts/cleanup_fantasy_elements.sh|Specific cleanup script"
        "scripts/deep_cleanup.sh|redundant/utilities/cleanup_scripts/deep_cleanup.sh|Similar to cleanup_cache.sh"
        "scripts/final_cleanup_summary.sh|redundant/utilities/cleanup_scripts/final_cleanup_summary.sh|Summary script"
        "scripts/fix_container_issues.sh|redundant/utilities/helper_scripts/fix_container_issues.sh|Specific fix script"
        "scripts/fix_dependencies.sh|redundant/utilities/helper_scripts/fix_dependencies.sh|Specific fix script"
        "scripts/fix_monitoring.sh|redundant/utilities/helper_scripts/fix_monitoring.sh|Specific fix script"
        "scripts/fix_permissions.sh|redundant/utilities/helper_scripts/fix_permissions.sh|Specific fix script"
        "scripts/optimize_system.sh|redundant/utilities/maintenance_tools/optimize_system.sh|General optimization script"
        "scripts/memory_cleanup_service.py|redundant/utilities/maintenance_tools/memory_cleanup_service.py|Similar to memory_optimizer.sh"
    )
    
    local moved=0
    local failed=0
    
    for file_spec in "${files_to_move[@]}"; do
        IFS='|' read -r source_file archive_path reason <<< "$file_spec"
        
        if move_file_safely "/opt/sutazaiapp/$source_file" "$archive_path" "$reason"; then
            ((moved++))
        else
            ((failed++))
        fi
    done
    
    log "Phase 5 completed: $moved moved, $failed failed"
    
    # Test system health after phase 5
    if ! test_system_health; then
        error "System health check failed after Phase 5"
        return 1
    fi
    
    return 0
}

# Create movement summary
create_movement_summary() {
    log "Creating movement summary..."
    
    local total_moved=$(grep -c "|SUCCESS$" "$MOVEMENT_LOG" 2>/dev/null || echo 0)
    local total_failed=$(grep -c "|FAILED$" "$MOVEMENT_LOG" 2>/dev/null || echo 0)
    
    cat > "$SUTAZAI_ARCHIVE_ROOT/movement_summary.md" << EOF
# SutazAI File Movement Summary

**Date:** $(date)
**Total Files Moved:** $total_moved
**Total Failures:** $total_failed

## Movement by Phase

### Phase 1: Monitoring Scripts
$(grep "Phase 1" "$LOG_FILE" | tail -1 || echo "Not executed")

### Phase 2: Deployment Scripts  
$(grep "Phase 2" "$LOG_FILE" | tail -1 || echo "Not executed")

### Phase 3: Testing Scripts
$(grep "Phase 3" "$LOG_FILE" | tail -1 || echo "Not executed")

### Phase 4: Configuration Files
$(grep "Phase 4" "$LOG_FILE" | tail -1 || echo "Not executed")

### Phase 5: Utility Scripts
$(grep "Phase 5" "$LOG_FILE" | tail -1 || echo "Not executed")

## Files Successfully Moved
\`\`\`
$(grep "|SUCCESS$" "$MOVEMENT_LOG" 2>/dev/null | cut -d'|' -f1-2 | sed 's/|/ -> /' || echo "None")
\`\`\`

## Files That Failed to Move
\`\`\`
$(grep "|FAILED$" "$MOVEMENT_LOG" 2>/dev/null | cut -d'|' -f1-2 | sed 's/|/ -> /' || echo "None")
\`\`\`

## System Health
Final system health check: $(tail -1 "$LOG_FILE" | grep "health check" || echo "Unknown")

## Archive Location
Files moved to: $SUTAZAI_ARCHIVE_ROOT

EOF

    log "Movement summary created"
}

# Main function
main() {
    log "Starting safe file movement process..."
    
    # Initialize movement log
    echo "# SutazAI File Movement Log" > "$MOVEMENT_LOG"
    echo "# Format: TIMESTAMP|SOURCE|DESTINATION|REASON|STATUS" >> "$MOVEMENT_LOG"
    
    # Initial system health check
    if ! test_system_health; then
        error "Initial system health check failed. Aborting movement."
        exit 1
    fi
    
    # Execute movement phases
    local phases_completed=0
    
    if move_phase_1_monitoring; then
        ((phases_completed++))
        sleep 5  # Brief pause between phases
    else
        error "Phase 1 failed. Stopping movement process."
        exit 1
    fi
    
    if move_phase_2_deployment; then
        ((phases_completed++))
        sleep 5
    else
        error "Phase 2 failed. Stopping movement process."
        exit 1
    fi
    
    if move_phase_3_testing; then
        ((phases_completed++))
        sleep 5
    else
        error "Phase 3 failed. Stopping movement process."
        exit 1
    fi
    
    if move_phase_4_configs; then
        ((phases_completed++))
        sleep 5
    else
        error "Phase 4 failed. Stopping movement process."
        exit 1
    fi
    
    if move_phase_5_utilities; then
        ((phases_completed++))
    else
        error "Phase 5 failed. Stopping movement process."
        exit 1
    fi
    
    # Final system health check
    if ! test_system_health; then
        error "Final system health check failed!"
        exit 1
    fi
    
    # Create summary
    create_movement_summary
    
    log "File movement completed successfully. $phases_completed phases completed."
    
    echo "✅ File movement completed successfully"
    echo "📊 Summary: $SUTAZAI_ARCHIVE_ROOT/movement_summary.md"
    echo "📋 Movement log: $MOVEMENT_LOG"
    echo "🏥 System health: ✅ All checks passed"
}

# Run main function
main "$@"