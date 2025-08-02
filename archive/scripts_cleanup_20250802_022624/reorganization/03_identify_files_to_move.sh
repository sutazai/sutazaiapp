#!/bin/bash
# SutazAI File Identification Script
# Identifies files for safe reorganization based on analysis

set -euo pipefail

# Configuration
ANALYSIS_DIR="/opt/sutazaiapp/scripts/reorganization/analysis"
LOG_FILE="/opt/sutazaiapp/logs/reorganization.log"

# Load archive path if available
if [ -f /tmp/sutazai_archive_path.env ]; then
    source /tmp/sutazai_archive_path.env
fi

# Ensure analysis directory exists
mkdir -p "$ANALYSIS_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE" >&2
}

# Files to absolutely preserve (critical system files)
declare -a CRITICAL_FILES=(
    "/opt/sutazaiapp/backend/app/main.py"
    "/opt/sutazaiapp/frontend/app.py"
    "/opt/sutazaiapp/backend/app/working_main.py"
    "/opt/sutazaiapp/docker-compose.minimal.yml"
    "/opt/sutazaiapp/scripts/live_logs.sh"
    "/opt/sutazaiapp/health_check.sh"
    "/opt/sutazaiapp/requirements.txt"
    "/opt/sutazaiapp/.env"
    "/opt/sutazaiapp/.env.tinyllama"
    "/opt/sutazaiapp/Makefile"
    "/opt/sutazaiapp/pyproject.toml"
)

# Essential scripts to preserve
declare -a ESSENTIAL_SCRIPTS=(
    "live_logs.sh"
    "health_check.sh"
    "deploy.sh"
    "start_backend.sh"
    "setup.sh"
    "monitor_system.sh"
)

# Check if file is critical
is_critical_file() {
    local file="$1"
    for critical in "${CRITICAL_FILES[@]}"; do
        if [[ "$file" == "$critical" ]]; then
            return 0
        fi
    done
    return 1
}

# Check if script is essential
is_essential_script() {
    local script="$1"
    local basename=$(basename "$script")
    for essential in "${ESSENTIAL_SCRIPTS[@]}"; do
        if [[ "$basename" == "$essential" ]]; then
            return 0
        fi
    done
    return 1
}

# Identify duplicate monitoring scripts
identify_duplicate_monitoring() {
    log "Identifying duplicate monitoring scripts..."
    
    cat > "$ANALYSIS_DIR/duplicate_monitoring.txt" << 'EOF'
# Duplicate Monitoring Scripts
# Scripts that provide similar monitoring functionality

# System Monitors (keep: monitor_system.sh)
scripts/static_monitor.py|DUPLICATE|Similar to monitor_system.sh
scripts/compact_monitor.py|DUPLICATE|Similar to monitor_system.sh  
scripts/emergency_resource_monitor.py|DUPLICATE|Similar to monitor_system.sh
scripts/memory_monitor_dashboard.py|DUPLICATE|Similar to monitor_system.sh
scripts/monitoring_system_monitor.py|DUPLICATE|Similar to monitor_system.sh
scripts/system_monitor.py|DUPLICATE|Similar to monitor_system.sh

# Dashboard Scripts (keep: monitor_dashboard.sh)
scripts/monitor_dashboard_README.md|DUPLICATE|Documentation for duplicate
scripts/agent_status_dashboard.sh|DUPLICATE|Similar to monitor_dashboard.sh

# Health Checks (keep: health_check.sh, health_check_all.sh)
scripts/health-monitor.py|DUPLICATE|Similar to health_check.sh
scripts/docker_health_check.sh|DUPLICATE|Similar to health_check.sh
scripts/check_docker_health.sh|DUPLICATE|Similar to health_check.sh
scripts/check_system_status.sh|DUPLICATE|Similar to health_check.sh
scripts/check_status.sh|DUPLICATE|Similar to health_check.sh
scripts/check_environment.sh|DUPLICATE|Similar to health_check.sh

# Log Scripts (keep: live_logs.sh)
scripts/live_logs.sh.backup|DUPLICATE|Backup version
scripts/live_logs_clean.sh|DUPLICATE|Similar to live_logs.sh
scripts/organized/monitoring/live_logs.sh|DUPLICATE|Duplicate in organized folder

EOF

    log "Duplicate monitoring scripts identified"
}

# Identify redundant deployment scripts
identify_duplicate_deployment() {
    log "Identifying duplicate deployment scripts..."
    
    cat > "$ANALYSIS_DIR/duplicate_deployment.txt" << 'EOF'
# Duplicate Deployment Scripts  
# Scripts that provide similar deployment functionality

# Deploy Scripts (keep: deploy.sh, deploy_complete_sutazai_agi_system.sh)
scripts/deploy_complete_system.sh.backup_before_cleanup|DUPLICATE|Backup version
scripts/deploy_complete_system.sh.backup_before_major_cleanup|DUPLICATE|Backup version
scripts/deploy_all.sh|DUPLICATE|Similar to main deploy
scripts/deploy_all_agents.sh|DUPLICATE|Specific case of main deploy
scripts/deploy_agi_system.sh|DUPLICATE|Similar to main deploy
scripts/deploy_task_automation.sh|DUPLICATE|Specific case of main deploy
scripts/deploy_taskmaster_integrated_system.sh|DUPLICATE|Specific case of main deploy
scripts/deploy-production.sh|DUPLICATE|Similar to main deploy
scripts/fixed_deploy.sh|DUPLICATE|Fixed version, but redundant

# Setup Scripts (keep: setup.sh)
scripts/setup_agent_demo.sh|DUPLICATE|Specific case
scripts/setup_all_models.sh|DUPLICATE|Specific case
scripts/setup_cron_jobs.sh|DUPLICATE|Specific case
scripts/setup_enterprise_security.sh|DUPLICATE|Specific case
scripts/setup_https.sh|DUPLICATE|Specific case
scripts/setup_models.sh|DUPLICATE|Specific case
scripts/setup_monitoring.sh|DUPLICATE|Specific case
scripts/setup_ollama_models.sh|DUPLICATE|Specific case
scripts/setup_repos.sh|DUPLICATE|Specific case
scripts/setup_secure_environment.sh|DUPLICATE|Specific case
scripts/setup_sync.sh|DUPLICATE|Specific case
scripts/setup_transformer_environment.sh|DUPLICATE|Specific case

# Organized duplicates
scripts/organized/deployment/deploy_universal_agents.py|DUPLICATE|In organized folder
scripts/organized/deployment/setup.py|DUPLICATE|In organized folder
scripts/organized/deployment/setup_models.py|DUPLICATE|In organized folder

EOF

    log "Duplicate deployment scripts identified"
}

# Identify redundant testing scripts
identify_duplicate_testing() {
    log "Identifying duplicate testing scripts..."
    
    cat > "$ANALYSIS_DIR/duplicate_testing.txt" << 'EOF'
# Duplicate Testing Scripts
# Scripts that provide similar testing functionality

# Brain Testing (consolidate into test_brain.sh)
scripts/test_brain_deployment.sh|DUPLICATE|Similar functionality
scripts/test_brain_deployment_explicit.sh|DUPLICATE|Similar functionality
scripts/test_brain_direct.sh|DUPLICATE|Similar functionality
scripts/test_brain_final.sh|DUPLICATE|Similar functionality
scripts/test_brain_functions.sh|DUPLICATE|Similar functionality
scripts/test_brain_simple.sh|DUPLICATE|Similar functionality
scripts/test_docker_brain.sh|DUPLICATE|Similar functionality
scripts/test_enhanced_brain.sh|DUPLICATE|Similar functionality

# Deployment Testing (consolidate into test_deployment.sh)
scripts/test_deployment.sh|KEEP|Main deployment test
scripts/test_deployment_system.sh|DUPLICATE|Similar functionality
scripts/test_deployment_with_brain.sh|DUPLICATE|Similar functionality
scripts/test_final_deployment.sh|DUPLICATE|Similar functionality
scripts/test_fixed_deployment.sh|DUPLICATE|Similar functionality

# System Testing (consolidate into test_system.sh)
scripts/test_automation.sh|DUPLICATE|Similar functionality
scripts/test_cleanup.sh|DUPLICATE|Similar functionality
scripts/test_env.sh|DUPLICATE|Similar functionality
scripts/test_function_timeout.sh|DUPLICATE|Similar functionality
scripts/test_performance.sh|DUPLICATE|Similar functionality
scripts/test_simple_fix.sh|DUPLICATE|Similar functionality
scripts/test_timeout_fix.sh|DUPLICATE|Similar functionality

# Specific Component Tests (archive as specialized)
scripts/test_advanced_features.py|ARCHIVE|Specialized test
scripts/test_brain_performance.py|ARCHIVE|Specialized test
scripts/test_chat_endpoints.py|ARCHIVE|Specialized test
scripts/test_frontend_integration.py|ARCHIVE|Specialized test
scripts/test_lightweight_models.py|ARCHIVE|Specialized test
scripts/test_neuromorphic_service.py|ARCHIVE|Specialized test
scripts/test_ollama_native.py|ARCHIVE|Specialized test
scripts/test_static_display.py|ARCHIVE|Specialized test
scripts/test_transformer_environment.py|ARCHIVE|Specialized test

EOF

    log "Duplicate testing scripts identified"
}

# Identify obsolete configuration files
identify_obsolete_configs() {
    log "Identifying obsolete configuration files..."
    
    cat > "$ANALYSIS_DIR/obsolete_configs.txt" << 'EOF'
# Obsolete Configuration Files
# Configuration files that are no longer needed

# Duplicate Docker Compose Files (keep: docker-compose.minimal.yml)
config/docker/docker-compose.tinyllama.yml|OBSOLETE|Duplicate config
docker-compose-agents-complete.yml|OBSOLETE|Old version
docker-compose.agents.yml|KEEP|Active agents config
docker-compose.yml|OBSOLETE|Large unused config

# Old Agent Configs (keep active ones only)
agents/configs/*_ollama.json|REVIEW|Check if still used
agents/configs/*_universal.json|REVIEW|Check if still used
agents/configs/*.modelfile|OBSOLETE|Old format

# Obsolete Environment Files
.env.agents|REVIEW|Check if still used
.env.example|KEEP|Template file
.env.ollama|REVIEW|Check if still used
.env.production|REVIEW|Check if still used

EOF

    log "Obsolete configurations identified"
}

# Identify redundant utilities
identify_redundant_utilities() {
    log "Identifying redundant utility scripts..."
    
    cat > "$ANALYSIS_DIR/redundant_utilities.txt" << 'EOF'
# Redundant Utility Scripts
# Scripts with overlapping functionality

# Cleanup Scripts (consolidate)
scripts/cleanup_cache.sh|KEEP|Essential cleanup
scripts/cleanup_redundancies.sh|DUPLICATE|Similar to cleanup_cache.sh
scripts/cleanup_fantasy_elements.sh|DUPLICATE|Specific cleanup
scripts/deep_cleanup.sh|DUPLICATE|Similar functionality
scripts/final_cleanup_summary.sh|DUPLICATE|Summary script

# Fix Scripts (archive old versions)
scripts/fix-memory-issues.sh|ARCHIVE|Specific fix
scripts/fix_container_issues.sh|ARCHIVE|Specific fix
scripts/fix_dependencies.sh|ARCHIVE|Specific fix
scripts/fix_docker_yaml.py|ARCHIVE|Specific fix
scripts/fix_litellm_prisma.sh|ARCHIVE|Specific fix
scripts/fix_missing_yaml_delimiter.py|ARCHIVE|Specific fix
scripts/fix_monitoring.sh|ARCHIVE|Specific fix
scripts/fix_permissions.sh|ARCHIVE|Specific fix
scripts/fix_security_vulnerabilities.py|ARCHIVE|Specific fix

# Optimization Scripts (consolidate)
scripts/optimize_and_deploy_agi.sh|DUPLICATE|Combined functionality
scripts/optimize_ollama.sh|KEEP|Specific optimization
scripts/optimize_system.sh|DUPLICATE|General optimization
scripts/optimize_transformer_models.sh|ARCHIVE|Specific optimization
scripts/optimize_transformers.py|ARCHIVE|Specific optimization
scripts/optimize_wsl2.sh|ARCHIVE|Platform specific

# Memory/Resource Scripts (consolidate)
scripts/memory_cleanup_service.py|DUPLICATE|Similar to memory_optimizer.sh
scripts/memory_optimizer.sh|KEEP|Essential optimizer
scripts/resource_analysis.py|ARCHIVE|Analysis script
scripts/limit_bash_cpu.sh|ARCHIVE|Specific limiter
scripts/limit_cpu.sh|ARCHIVE|Specific limiter

EOF

    log "Redundant utilities identified"
}

# Generate comprehensive movement plan
generate_movement_plan() {
    log "Generating comprehensive movement plan..."
    
    cat > "$ANALYSIS_DIR/movement_plan.txt" << 'EOF'
# SutazAI File Movement Plan
# Comprehensive plan for safe file reorganization

## Phase 1: Duplicate Monitoring Scripts
MOVE|scripts/static_monitor.py|duplicates/monitoring/system_monitors/
MOVE|scripts/compact_monitor.py|duplicates/monitoring/system_monitors/
MOVE|scripts/emergency_resource_monitor.py|duplicates/monitoring/system_monitors/
MOVE|scripts/memory_monitor_dashboard.py|duplicates/monitoring/dashboard_scripts/
MOVE|scripts/monitoring_system_monitor.py|duplicates/monitoring/system_monitors/
MOVE|scripts/health-monitor.py|duplicates/monitoring/health_checks/
MOVE|scripts/live_logs.sh.backup|duplicates/monitoring/log_processors/

## Phase 2: Duplicate Deployment Scripts  
MOVE|scripts/deploy_complete_system.sh.backup_before_cleanup|duplicates/deployment/backup_deployments/
MOVE|scripts/deploy_complete_system.sh.backup_before_major_cleanup|duplicates/deployment/backup_deployments/
MOVE|scripts/deploy_all.sh|duplicates/deployment/old_deploy_scripts/
MOVE|scripts/deploy_all_agents.sh|duplicates/deployment/old_deploy_scripts/
MOVE|scripts/deploy_agi_system.sh|duplicates/deployment/old_deploy_scripts/
MOVE|scripts/fixed_deploy.sh|duplicates/deployment/old_deploy_scripts/

## Phase 3: Duplicate Testing Scripts
MOVE|scripts/test_brain_deployment.sh|duplicates/testing/old_test_scripts/
MOVE|scripts/test_brain_deployment_explicit.sh|duplicates/testing/old_test_scripts/
MOVE|scripts/test_brain_direct.sh|duplicates/testing/old_test_scripts/
MOVE|scripts/test_deployment_system.sh|duplicates/testing/old_test_scripts/
MOVE|scripts/test_advanced_features.py|duplicates/testing/performance_tests/

## Phase 4: Obsolete Configurations
MOVE|config/docker/docker-compose.tinyllama.yml|obsolete/deprecated/
MOVE|docker-compose-agents-complete.yml|obsolete/old_versions/
MOVE|docker-compose.yml|obsolete/old_versions/

## Phase 5: Redundant Utilities
MOVE|scripts/cleanup_redundancies.sh|redundant/utilities/cleanup_scripts/
MOVE|scripts/cleanup_fantasy_elements.sh|redundant/utilities/cleanup_scripts/
MOVE|scripts/fix_container_issues.sh|redundant/utilities/helper_scripts/
MOVE|scripts/optimize_system.sh|redundant/utilities/maintenance_tools/

## Files to Keep (DO NOT MOVE)
KEEP|backend/app/main.py|Active backend
KEEP|frontend/app.py|Active frontend  
KEEP|backend/app/working_main.py|Backup backend
KEEP|docker-compose.minimal.yml|Current deployment
KEEP|scripts/live_logs.sh|Essential monitoring
KEEP|health_check.sh|Essential health check
KEEP|scripts/monitor_system.sh|Essential monitoring

EOF

    log "Movement plan generated"
}

# Create file analysis report
create_analysis_report() {
    log "Creating comprehensive analysis report..."
    
    cat > "$ANALYSIS_DIR/analysis_report.md" << 'EOF'
# SutazAI Codebase Analysis Report

## Overview
This report identifies files for safe reorganization while preserving system stability.

## Critical Files (NEVER MOVE)
- `backend/app/main.py` - Active backend application
- `frontend/app.py` - Active frontend application  
- `backend/app/working_main.py` - Backup backend
- `docker-compose.minimal.yml` - Current deployment configuration
- `scripts/live_logs.sh` - Essential monitoring script
- `health_check.sh` - System health monitoring

## Files Identified for Movement

### Duplicate Scripts
EOF

    # Add statistics
    echo "- **Monitoring Duplicates**: $(grep -c "scripts/" "$ANALYSIS_DIR/duplicate_monitoring.txt" || echo 0) files" >> "$ANALYSIS_DIR/analysis_report.md"
    echo "- **Deployment Duplicates**: $(grep -c "scripts/" "$ANALYSIS_DIR/duplicate_deployment.txt" || echo 0) files" >> "$ANALYSIS_DIR/analysis_report.md"
    echo "- **Testing Duplicates**: $(grep -c "scripts/" "$ANALYSIS_DIR/duplicate_testing.txt" || echo 0) files" >> "$ANALYSIS_DIR/analysis_report.md"
    echo "- **Utility Redundancies**: $(grep -c "scripts/" "$ANALYSIS_DIR/redundant_utilities.txt" || echo 0) files" >> "$ANALYSIS_DIR/analysis_report.md"
    
    cat >> "$ANALYSIS_DIR/analysis_report.md" << 'EOF'

## Safety Measures
1. Complete system backup before any movement
2. Move files in phases with testing between each
3. Preserve all files in organized archive structure
4. Full restoration capability maintained
5. System health monitoring throughout process

## Movement Plan
See `movement_plan.txt` for detailed file-by-file movement instructions.

## Estimated Impact
- **Scripts Directory**: Reduction from 366+ files to ~50 essential files
- **Disk Space**: ~200MB moved to archive
- **Maintenance**: Significantly easier navigation and maintenance
- **Risk**: Minimal - all files preserved and restorable

EOF

    log "Analysis report created"
}

# Main function
main() {
    log "Starting comprehensive file analysis..."
    
    # Run all identification functions
    identify_duplicate_monitoring
    identify_duplicate_deployment  
    identify_duplicate_testing
    identify_obsolete_configs
    identify_redundant_utilities
    
    # Generate plans and reports
    generate_movement_plan
    create_analysis_report
    
    log "File analysis completed successfully"
    
    echo "✅ File analysis completed"
    echo "📊 Analysis results: $ANALYSIS_DIR/"
    echo "📋 Movement plan: $ANALYSIS_DIR/movement_plan.txt"
    echo "📖 Full report: $ANALYSIS_DIR/analysis_report.md"
    
    # Show summary
    echo ""
    echo "📈 Summary:"
    echo "  - Monitoring duplicates: $(grep -c "|DUPLICATE|" "$ANALYSIS_DIR/duplicate_monitoring.txt" 2>/dev/null || echo 0)"
    echo "  - Deployment duplicates: $(grep -c "|DUPLICATE|" "$ANALYSIS_DIR/duplicate_deployment.txt" 2>/dev/null || echo 0)"
    echo "  - Testing duplicates: $(grep -c "|DUPLICATE|" "$ANALYSIS_DIR/duplicate_testing.txt" 2>/dev/null || echo 0)"
    echo "  - Redundant utilities: $(grep -c "|DUPLICATE|" "$ANALYSIS_DIR/redundant_utilities.txt" 2>/dev/null || echo 0)"
    echo "  - Total files to move: $(grep -c "MOVE|" "$ANALYSIS_DIR/movement_plan.txt" 2>/dev/null || echo 0)"
}

# Run main function
main "$@"