#!/bin/bash
# Automated Hygiene Audit Script
# Runs daily to ensure codebase hygiene compliance

set -euo pipefail

# Configuration

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs/hygiene"
REPORT_FILE="$LOG_DIR/hygiene-audit-$(date +%Y%m%d-%H%M%S).log"

# Create log directory
mkdir -p "$LOG_DIR"

# Functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$REPORT_FILE"
}

check_result() {
    if [ $? -eq 0 ]; then
        log "✅ $1: PASSED"
        return 0
    else
        log "❌ $1: FAILED"
        return 1
    fi
}

# Start audit
log "Starting Codebase Hygiene Audit"
log "Project: $PROJECT_ROOT"
log "========================================="

# Track overall status
AUDIT_PASSED=true

# 1. Check for backup files
log ""
log "Checking for backup files..."
if find "$PROJECT_ROOT" -name "*.backup*" -o -name "*.conceptual*" -o -name "*.agi_backup" 2>/dev/null | grep -v "/logs/" | grep -q .; then
    log "Found backup files:"
    find "$PROJECT_ROOT" -name "*.backup*" -o -name "*.conceptual*" -o -name "*.agi_backup" 2>/dev/null | grep -v "/logs/" | tee -a "$REPORT_FILE"
    AUDIT_PASSED=false
else
    log "✅ No backup files found"
fi

# 2. Check for archive directories
log ""
log "Checking for archive directories..."
if find "$PROJECT_ROOT" -type d -name "*archive*" 2>/dev/null | grep -v "/logs/" | grep -q .; then
    log "Found archive directories:"
    find "$PROJECT_ROOT" -type d -name "*archive*" 2>/dev/null | grep -v "/logs/" | tee -a "$REPORT_FILE"
    AUDIT_PASSED=false
else
    log "✅ No archive directories found"
fi

# 3. Run Python hygiene checks
log ""
log "Running Python hygiene checks..."

cd "$PROJECT_ROOT"

# Check secrets
log "Checking for hardcoded secrets..."
python scripts/check_secrets.py >> "$REPORT_FILE" 2>&1
check_result "Secret check" || AUDIT_PASSED=false

# Check naming conventions
log "Checking naming conventions..."
python scripts/check_naming.py >> "$REPORT_FILE" 2>&1
check_result "Naming conventions" || AUDIT_PASSED=false

# Check duplicates
log "Checking for duplicates..."
python scripts/check_duplicates.py >> "$REPORT_FILE" 2>&1
check_result "Duplicate check" || AUDIT_PASSED=false

# Validate agents
log "Validating agent files..."
python scripts/validate_agents.py >> "$REPORT_FILE" 2>&1
check_result "Agent validation" || AUDIT_PASSED=false

# Check requirements
log "Checking requirements files..."
python scripts/check_requirements.py >> "$REPORT_FILE" 2>&1
check_result "Requirements check" || AUDIT_PASSED=false

# 4. Check for large files
log ""
log "Checking for large files..."
large_files=$(find "$PROJECT_ROOT" -type f -size +5M 2>/dev/null | grep -v -E "(\.git|node_modules|venv|\.png|\.jpg|\.jpeg|\.gif|\.pdf|/logs/)" || true)
if [ -n "$large_files" ]; then
    log "WARNING: Large files found (>5MB):"
    echo "$large_files" | tee -a "$REPORT_FILE"
fi

# 5. Check for empty directories
log ""
log "Checking for empty directories..."
empty_dirs=$(find "$PROJECT_ROOT" -type d -empty 2>/dev/null | grep -v -E "(\.git|__pycache__|/logs/)" || true)
if [ -n "$empty_dirs" ]; then
    log "Empty directories found:"
    echo "$empty_dirs" | tee -a "$REPORT_FILE"
fi

# 6. Generate summary
log ""
log "========================================="
log "AUDIT SUMMARY"
log "========================================="

if [ "$AUDIT_PASSED" = true ]; then
    log "✅ ALL HYGIENE CHECKS PASSED"
    exit_code=0
else
    log "❌ HYGIENE VIOLATIONS DETECTED"
    log "Please review the report and fix violations"
    exit_code=1
fi

# 7. Cleanup old logs (keep last 30 days)
log ""
log "Cleaning up old audit logs..."
find "$LOG_DIR" -name "hygiene-audit-*.log" -mtime +30 -delete 2>/dev/null || true

# 8. Send notification if violations found
if [ "$AUDIT_PASSED" = false ]; then
    # Create a summary file for other systems to read
    echo "FAILED" > "$LOG_DIR/latest-status.txt"
    echo "Violations found on $(date)" >> "$LOG_DIR/latest-status.txt"
    
    # If you have a notification system, trigger it here
    # For example: curl -X POST http://localhost:8000/api/notifications/hygiene-violation
else
    echo "PASSED" > "$LOG_DIR/latest-status.txt"
    echo "All checks passed on $(date)" >> "$LOG_DIR/latest-status.txt"
fi

log ""
log "Audit complete. Report saved to: $REPORT_FILE"
log "Latest status: $LOG_DIR/latest-status.txt"

exit $exit_code