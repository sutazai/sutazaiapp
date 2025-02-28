#!/bin/bash

# Comprehensive Code Audit Script for Sutazaiapp
# Performs multi-layered code quality and security checks

set -euo pipefail

# Configuration
SUTAZAIAPP_HOME="/opt/sutazaiapp"
AUDIT_LOG_DIR="$SUTAZAIAPP_HOME/logs/code_audit"
REPORT_FILE="$AUDIT_LOG_DIR/code_audit_report_$(date +%Y%m%d_%H%M%S).md"

# Create audit log directory
mkdir -p "$AUDIT_LOG_DIR"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$REPORT_FILE"
}

# Activate virtual environment
source "$SUTAZAIAPP_HOME/venv/bin/activate"

# Start audit report
echo "# Sutazaiapp Code Audit Report" > "$REPORT_FILE"
echo "## Audit Timestamp: $(date)" >> "$REPORT_FILE"

# 1. Semgrep Security Scan
log "🔍 Running Semgrep Security Scan..."
{
    echo "### Semgrep Security Scan" >> "$REPORT_FILE"
    semgrep scan --config=auto "$SUTAZAIAPP_HOME" | tee -a "$REPORT_FILE"
} || log "Semgrep scan encountered issues"

# 2. Pylint Code Quality Check
log "🧐 Running Pylint Code Quality Check..."
{
    echo "### Pylint Code Quality Check" >> "$REPORT_FILE"
    find "$SUTAZAIAPP_HOME" -name "*.py" | xargs pylint \
        --output-format=text \
        --reports=y \
        --evaluation=10.0 \
        | tee -a "$REPORT_FILE"
} || log "Pylint encountered issues"

# 3. Mypy Type Checking
log "🕵️ Running Mypy Type Checking..."
{
    echo "### Mypy Type Checking" >> "$REPORT_FILE"
    mypy "$SUTAZAIAPP_HOME" \
        --ignore-missing-imports \
        --follow-imports=silent \
        --show-column-numbers \
        | tee -a "$REPORT_FILE"
} || log "Mypy type checking encountered issues"

# 4. Bandit Security Scanning
log "🛡️ Running Bandit Security Scan..."
{
    echo "### Bandit Security Scan" >> "$REPORT_FILE"
    bandit -r "$SUTAZAIAPP_HOME" \
        -f custom \
        -x "**/tests/**" \
        | tee -a "$REPORT_FILE"
} || log "Bandit security scan encountered issues"

# 5. Dependency Vulnerability Check
log "🔬 Checking Dependencies for Vulnerabilities..."
{
    echo "### Dependency Vulnerability Check" >> "$REPORT_FILE"
    safety check -r "$SUTAZAIAPP_HOME/requirements.txt" | tee -a "$REPORT_FILE"
} || log "Safety dependency check encountered issues"

# 6. Code Complexity Analysis
log "📊 Analyzing Code Complexity..."
{
    echo "### Code Complexity Analysis" >> "$REPORT_FILE"
    radon cc "$SUTAZAIAPP_HOME" -s -a -nb | tee -a "$REPORT_FILE"
} || log "Radon complexity analysis encountered issues"

# Final summary
log "✅ Code Audit Complete. Full report available at $REPORT_FILE"
echo "## Audit Completed: $(date)" >> "$REPORT_FILE"

# Optional: Send notification or trigger actions based on audit results
# You can add custom logic here to handle different audit scenarios 