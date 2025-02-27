#!/bin/bash
set -euo pipefail

# Comprehensive Audit Script for Sutazaiapp

# Configuration
SUTAZAIAPP_HOME="/opt/sutazaiapp"
AUDIT_REPORT="/opt/sutazaiapp/docs/audit/Final_Audit.md"
AUDIT_LOG="/var/log/sutazaiapp/audit.log"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$AUDIT_LOG"
}

# Prepare audit report
prepare_report_header() {
    cat > "$AUDIT_REPORT" << EOL
# Sutazaiapp Comprehensive Audit Report
## Generated on $(date)

### Audit Sections
- [Code Quality](#code-quality)
- [Security Scan](#security-scan)
- [Type Checking](#type-checking)
- [Summary](#summary)

## Code Quality
EOL
}

# Run Pylint
run_pylint() {
    log "Running Pylint..."
    pylint $(find "$SUTAZAIAPP_HOME" -name "*.py" -not -path "*/venv/*") \
        | tee -a "$AUDIT_REPORT"
}

# Run Semgrep
run_semgrep() {
    log "Running Semgrep..."
    semgrep scan "$SUTAZAIAPP_HOME" \
        --config=r/all \
        --markdown >> "$AUDIT_REPORT"
}

# Run Mypy Type Checking
run_mypy() {
    log "Running Mypy Type Checking..."
    echo "### Type Checking Results" >> "$AUDIT_REPORT"
    mypy "$SUTAZAIAPP_HOME" \
        --ignore-missing-imports \
        | tee -a "$AUDIT_REPORT"
}

# Run Bandit Security Scan
run_bandit() {
    log "Running Bandit Security Scan..."
    echo "### Security Scan Results" >> "$AUDIT_REPORT"
    bandit -r "$SUTAZAIAPP_HOME" \
        -f markdown >> "$AUDIT_REPORT"
}

# Generate Summary
generate_summary() {
    echo "## Summary" >> "$AUDIT_REPORT"
    echo "- **Audit Timestamp**: $(date)" >> "$AUDIT_REPORT"
    echo "- **Total Python Files Scanned**: $(find "$SUTAZAIAPP_HOME" -name "*.py" -not -path "*/venv/*" | wc -l)" >> "$AUDIT_REPORT"
}

# Main Audit Function
main() {
    log "Starting Comprehensive Audit"
    
    prepare_report_header
    run_pylint
    run_semgrep
    run_mypy
    run_bandit
    generate_summary
    
    log "Audit Complete. Report generated at $AUDIT_REPORT"
}

# Execute main function
main 