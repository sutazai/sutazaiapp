#!/bin/bash
# Purpose: Setup automated hygiene enforcement cron jobs and monitoring
# Usage: ./setup-hygiene-automation.sh [--install|--uninstall|--status]
# Requirements: cron, root access for system-wide installation

set -euo pipefail


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

PROJECT_ROOT="/opt/sutazaiapp"
CRON_USER="${SUDO_USER:-$(whoami)}"
ACTION="${1:-install}"

log_action() {
    echo "[$(date -Iseconds)] $1"
}

install_cron_jobs() {
    log_action "Installing hygiene automation cron jobs for user $CRON_USER"
    
    # Create temporary cron file
    TEMP_CRON=$(mktemp)
    
    # Get existing cron jobs (if any)
    (crontab -u "$CRON_USER" -l 2>/dev/null || true) > "$TEMP_CRON"
    
    # Remove any existing hygiene automation jobs
    grep -v "sutazaiapp.*hygiene" "$TEMP_CRON" > "${TEMP_CRON}.clean" || true
    mv "${TEMP_CRON}.clean" "$TEMP_CRON"
    
    # Add new hygiene automation jobs
    cat >> "$TEMP_CRON" << EOF

# SutazAI Hygiene Automation (installed $(date))
# Daily junk file cleanup at 2 AM
0 2 * * * $PROJECT_ROOT/scripts/utils/automated-hygiene-maintenance.sh daily >> $PROJECT_ROOT/logs/cron-daily.log 2>&1

# Weekly comprehensive check on Sundays at 3 AM  
0 3 * * 0 $PROJECT_ROOT/scripts/utils/automated-hygiene-maintenance.sh weekly >> $PROJECT_ROOT/logs/cron-weekly.log 2>&1

# Monthly full audit on 1st of month at 4 AM
0 4 1 * * $PROJECT_ROOT/scripts/utils/automated-hygiene-maintenance.sh monthly >> $PROJECT_ROOT/logs/cron-monthly.log 2>&1

# Real-time monitoring every 6 hours
0 */6 * * * $PROJECT_ROOT/scripts/hygiene-monitor.py --duration=3600 >> $PROJECT_ROOT/logs/cron-monitor.log 2>&1

# Emergency cleanup if junk files exceed threshold (every 2 hours)
0 */2 * * * if [ \$(find $PROJECT_ROOT -name "*.backup*" -o -name "*.tmp" | wc -l) -gt 20 ]; then $PROJECT_ROOT/scripts/utils/automated-hygiene-maintenance.sh emergency >> $PROJECT_ROOT/logs/cron-emergency.log 2>&1; fi

EOF
    
    # Install the updated cron jobs
    crontab -u "$CRON_USER" "$TEMP_CRON"
    
    # Clean up
    rm "$TEMP_CRON"
    
    log_action "Hygiene automation cron jobs installed successfully"
    
    # Create log rotation for hygiene logs
    create_logrotate_config
}

create_logrotate_config() {
    log_action "Setting up log rotation for hygiene logs"
    
    # Create logrotate configuration
    LOGROTATE_CONFIG="/etc/logrotate.d/sutazai-hygiene"
    
    if [[ -w /etc/logrotate.d/ ]]; then
        cat > "$LOGROTATE_CONFIG" << EOF
$PROJECT_ROOT/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    create 644 $CRON_USER $CRON_USER
}
EOF
        log_action "Log rotation configured at $LOGROTATE_CONFIG"
    else
        log_action "Cannot write to /etc/logrotate.d/ - skipping log rotation setup" "WARN"
    fi
}

uninstall_cron_jobs() {
    log_action "Uninstalling hygiene automation cron jobs for user $CRON_USER"
    
    # Create temporary cron file
    TEMP_CRON=$(mktemp)
    
    # Get existing cron jobs and remove hygiene automation
    (crontab -u "$CRON_USER" -l 2>/dev/null || true) > "$TEMP_CRON"
    
    # Remove hygiene automation jobs and comments
    grep -v -E "(sutazaiapp.*hygiene|SutazAI Hygiene Automation)" "$TEMP_CRON" > "${TEMP_CRON}.clean" || true
    
    # Install cleaned cron jobs
    crontab -u "$CRON_USER" "${TEMP_CRON}.clean"
    
    # Clean up
    rm "$TEMP_CRON" "${TEMP_CRON}.clean"
    
    # Remove logrotate config
    if [[ -f /etc/logrotate.d/sutazai-hygiene ]]; then
        rm -f /etc/logrotate.d/sutazai-hygiene
        log_action "Removed logrotate configuration"
    fi
    
    log_action "Hygiene automation cron jobs uninstalled successfully"
}

show_status() {
    log_action "Checking hygiene automation status for user $CRON_USER"
    
    echo
    echo "=== Current Cron Jobs ==="
    crontab -u "$CRON_USER" -l 2>/dev/null | grep -E "(sutazaiapp|SutazAI)" || echo "No hygiene automation cron jobs found"
    
    echo
    echo "=== Recent Log Activity ==="
    if [[ -d "$PROJECT_ROOT/logs" ]]; then
        echo "Log files:"
        ls -la "$PROJECT_ROOT/logs/"*.log 2>/dev/null | head -10 || echo "No log files found"
        
        echo
        echo "Recent hygiene activity (last 10 lines):"
        tail -10 "$PROJECT_ROOT/logs/automated-maintenance.log" 2>/dev/null || echo "No maintenance log found"
    else
        echo "Log directory not found: $PROJECT_ROOT/logs"
    fi
    
    echo
    echo "=== System Status ==="
    echo "Project root: $PROJECT_ROOT"
    echo "Current junk files: $(find "$PROJECT_ROOT" -name "*.backup*" -o -name "*.tmp" -o -name "*~" 2>/dev/null | wc -l)"
    echo "Current backup files: $(find "$PROJECT_ROOT" -name "*.backup*" 2>/dev/null | wc -l)"
    echo "Deployment scripts: $(find "$PROJECT_ROOT" -name "*deploy*.sh" -o -name "*deploy*.py" 2>/dev/null | wc -l)"
    echo "Agent directories: $(find "$PROJECT_ROOT" -type d -name "*agent*" 2>/dev/null | wc -l)"
}

run_test() {
    log_action "Running hygiene automation test"
    
    # Test daily maintenance in dry-run mode
    log_action "Testing daily maintenance..."
    "$PROJECT_ROOT/scripts/utils/automated-hygiene-maintenance.sh" daily
    
    # Test enforcement coordinator
    log_action "Testing enforcement coordinator..."
    python3 "$PROJECT_ROOT/scripts/hygiene-enforcement-coordinator.py" --phase=1 --dry-run
    
    log_action "Hygiene automation test completed"
}

main() {
    log_action "Setting up SutazAI hygiene automation"
    
    # Ensure we're in the right directory
    if [[ ! -d "$PROJECT_ROOT" ]]; then
        log_action "Project root not found: $PROJECT_ROOT" "ERROR"
        exit 1
    fi
    
    # Ensure required scripts exist
    required_scripts=(
        "$PROJECT_ROOT/scripts/utils/automated-hygiene-maintenance.sh"
        "$PROJECT_ROOT/scripts/hygiene-enforcement-coordinator.py"
        "$PROJECT_ROOT/scripts/hygiene-monitor.py"
    )
    
    for script in "${required_scripts[@]}"; do
        if [[ ! -x "$script" ]]; then
            log_action "Required script not found or not executable: $script" "ERROR"
            exit 1
        fi
    done
    
    # Create logs directory
    mkdir -p "$PROJECT_ROOT/logs"
    
    case "$ACTION" in
        "install")
            install_cron_jobs
            show_status
            ;;
        "uninstall")
            uninstall_cron_jobs
            ;;
        "status")
            show_status
            ;;
        "test")
            run_test
            ;;
        *)
            echo "Usage: $0 [install|uninstall|status|test]"
            echo
            echo "  install   - Install hygiene automation cron jobs"
            echo "  uninstall - Remove hygiene automation cron jobs"
            echo "  status    - Show current automation status"
            echo "  test      - Run automation test in safe mode"
            exit 1
            ;;
    esac
}

main "$@"