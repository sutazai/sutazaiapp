#!/bin/bash
# Purpose: Automated hygiene maintenance for ongoing rule compliance
# Usage: ./automated-hygiene-maintenance.sh [--mode=daily|weekly|monthly]
# Requirements: Python 3.8+, specialized agents, cron access

set -euo pipefail

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="$PROJECT_ROOT/logs"
ARCHIVE_DIR="$PROJECT_ROOT/archive"
MAINTENANCE_MODE="${1:-daily}"

# Ensure log directory exists
mkdir -p "$LOG_DIR"
mkdir -p "$ARCHIVE_DIR"

# Logging function
log_action() {
    local message="$1"
    local level="${2:-INFO}"
    local timestamp=$(date -Iseconds)
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_DIR/automated-maintenance.log"
}

# Check if hygiene enforcement tools are available
check_prerequisites() {
    log_action "Checking prerequisites for automated maintenance"
    
    local required_scripts=(
        "$PROJECT_ROOT/scripts/hygiene-enforcement-coordinator.py"
        "$PROJECT_ROOT/scripts/agents/hygiene-agent-orchestrator.py"
        "$PROJECT_ROOT/scripts/hygiene-monitor.py"
    )
    
    for script in "${required_scripts[@]}"; do
        if [[ ! -f "$script" ]]; then
            log_action "Missing required script: $script" "ERROR"
            return 1
        fi
    done
    
    # Check Python availability
    if ! command -v python3 &> /dev/null; then
        log_action "Python 3 not available" "ERROR" 
        return 1
    fi
    
    log_action "Prerequisites check passed"
    return 0
}

# Daily maintenance tasks
run_daily_maintenance() {
    log_action "=== STARTING DAILY HYGIENE MAINTENANCE ==="
    
    # Rule 13: Clean up any new junk files
    log_action "Scanning for new junk files (Rule 13)"
    python3 "$PROJECT_ROOT/scripts/hygiene-enforcement-coordinator.py" \
        --phase=1 --dry-run 2>&1 | tee -a "$LOG_DIR/daily-maintenance.log"
    
    # Check for backup files that shouldn't exist
    backup_count=$(find "$PROJECT_ROOT" -name "*.backup*" -type f | wc -l)
    if [[ $backup_count -gt 0 ]]; then
        log_action "Found $backup_count backup files - scheduling cleanup" "WARN"
        
        # Archive backup files automatically
        archive_date=$(date +%Y-%m-%d)
        backup_archive="$ARCHIVE_DIR/$archive_date-auto-backup-cleanup"
        mkdir -p "$backup_archive"
        
        find "$PROJECT_ROOT" -name "*.backup*" -type f -exec cp {} "$backup_archive/" \;
        find "$PROJECT_ROOT" -name "*.backup*" -type f -delete
        
        log_action "Cleaned up $backup_count backup files to $backup_archive"
    fi
    
    # Monitor for rule violations in real-time
    log_action "Starting real-time hygiene monitoring"
    timeout 300 python3 "$PROJECT_ROOT/scripts/hygiene-monitor.py" --duration=300 || true
    
    log_action "Daily maintenance completed"
}

# Weekly maintenance tasks  
run_weekly_maintenance() {
    log_action "=== STARTING WEEKLY HYGIENE MAINTENANCE ==="
    
    # Comprehensive rule enforcement for critical rules
    log_action "Running Phase 1 enforcement (critical rules)"
    python3 "$PROJECT_ROOT/scripts/agents/hygiene-agent-orchestrator.py" \
        --phase=1 2>&1 | tee -a "$LOG_DIR/weekly-maintenance.log"
    
    # Script organization audit (Rule 7)
    log_action "Auditing script organization"
    script_count=$(find "$PROJECT_ROOT" -name "*.sh" -o -name "*.py" | wc -l)
    scripts_in_scripts_dir=$(find "$PROJECT_ROOT/scripts" -name "*.sh" -o -name "*.py" | wc -l)
    
    if [[ $scripts_in_scripts_dir -lt $(($script_count * 80 / 100)) ]]; then
        log_action "Script organization needs attention: $scripts_in_scripts_dir/$script_count in /scripts/" "WARN"
    fi
    
    # Documentation consistency check (Rules 6, 15)
    log_action "Checking documentation consistency"
    readme_count=$(find "$PROJECT_ROOT" -name "README*" -type f | wc -l)
    if [[ $readme_count -gt 5 ]]; then
        log_action "Too many README files found: $readme_count (consider consolidation)" "WARN"
    fi
    
    log_action "Weekly maintenance completed"
}

# Monthly maintenance tasks
run_monthly_maintenance() {
    log_action "=== STARTING MONTHLY HYGIENE MAINTENANCE ==="
    
    # Full system compliance audit
    log_action "Running comprehensive compliance audit"
    python3 "$PROJECT_ROOT/scripts/agents/hygiene-agent-orchestrator.py" \
        --dry-run 2>&1 | tee -a "$LOG_DIR/monthly-maintenance.log"
    
    # Generate compliance report
    log_action "Generating monthly compliance report"
    report_file="$LOG_DIR/monthly-compliance-$(date +%Y-%m).json"
    
    # Create comprehensive report
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "period": "$(date +%Y-%m)",
    "project_root": "$PROJECT_ROOT",
    "statistics": {
        "total_files": $(find "$PROJECT_ROOT" -type f | wc -l),
        "python_files": $(find "$PROJECT_ROOT" -name "*.py" | wc -l),
        "shell_scripts": $(find "$PROJECT_ROOT" -name "*.sh" | wc -l),
        "docker_files": $(find "$PROJECT_ROOT" -name "Dockerfile*" -o -name "docker-compose*" | wc -l),
        "backup_files": $(find "$PROJECT_ROOT" -name "*.backup*" | wc -l),
        "temp_files": $(find "$PROJECT_ROOT" -name "*.tmp" -o -name "*~" | wc -l)
    },
    "compliance_status": {
        "rule_13_junk_files": $(find "$PROJECT_ROOT" -name "*.backup*" -o -name "*.tmp" -o -name "*~" | wc -l),
        "rule_12_deploy_scripts": $(find "$PROJECT_ROOT" -name "*deploy*.sh" -o -name "*deploy*.py" | wc -l),
        "rule_11_docker_files": $(find "$PROJECT_ROOT" -name "Dockerfile*" | wc -l)
    }
}
EOF
    
    log_action "Monthly compliance report saved to $report_file"
    
    # Archive old logs (keep last 3 months)
    log_action "Archiving old logs"
    find "$LOG_DIR" -name "*.log" -mtime +90 -exec mv {} "$ARCHIVE_DIR/" \;
    
    # Clean up old archives (keep last 6 months) 
    find "$ARCHIVE_DIR" -type d -mtime +180 -exec rm -rf {} \; || true
    
    log_action "Monthly maintenance completed"
}

# Emergency cleanup for critical violations
run_emergency_cleanup() {
    log_action "=== RUNNING EMERGENCY HYGIENE CLEANUP ==="
    
    # Immediate junk file removal
    junk_files=$(find "$PROJECT_ROOT" -name "*.backup*" -o -name "*.tmp" -o -name "*~" | wc -l)
    
    if [[ $junk_files -gt 50 ]]; then
        log_action "CRITICAL: $junk_files junk files found - running emergency cleanup" "ERROR"
        
        emergency_archive="$ARCHIVE_DIR/emergency-cleanup-$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$emergency_archive"
        
        # Archive and remove junk files
        find "$PROJECT_ROOT" \( -name "*.backup*" -o -name "*.tmp" -o -name "*~" \) \
            -exec cp {} "$emergency_archive/" \; -delete
        
        log_action "Emergency cleanup completed: $junk_files files archived to $emergency_archive"
    fi
    
    # Check for critical structural issues
    agent_dirs=$(find "$PROJECT_ROOT" -type d -name "*agent*" | wc -l)
    if [[ $agent_dirs -gt 10 ]]; then
        log_action "CRITICAL: $agent_dirs agent directories found - requires manual intervention" "ERROR"
    fi
    
    deploy_scripts=$(find "$PROJECT_ROOT" -name "*deploy*.sh" -o -name "*deploy*.py" | wc -l)
    if [[ $deploy_scripts -gt 3 ]]; then
        log_action "CRITICAL: $deploy_scripts deployment scripts found - requires consolidation" "ERROR"
    fi
}

# Main execution logic
main() {
    log_action "Starting automated hygiene maintenance in $MAINTENANCE_MODE mode"
    
    if ! check_prerequisites; then
        log_action "Prerequisites check failed - aborting maintenance" "ERROR"
        exit 1
    fi
    
    case "$MAINTENANCE_MODE" in
        "daily")
            run_daily_maintenance
            ;;
        "weekly") 
            run_weekly_maintenance
            ;;
        "monthly")
            run_monthly_maintenance
            ;;
        "emergency")
            run_emergency_cleanup
            ;;
        *)
            log_action "Invalid maintenance mode: $MAINTENANCE_MODE" "ERROR"
            echo "Usage: $0 [daily|weekly|monthly|emergency]"
            exit 1
            ;;
    esac
    
    log_action "Automated hygiene maintenance completed successfully"
}

# Trap for cleanup on exit
cleanup() {
    log_action "Maintenance script interrupted or completed"
}
trap cleanup EXIT

# Run main function
main "$@"