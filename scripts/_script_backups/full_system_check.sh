#!/bin/bash
set -euo pipefail

# Comprehensive System Check and Self-Healing Script for SutazAI

LOG_DIR="logs"
AUDIT_LOG="$LOG_DIR/audit.log"

echo "Starting full system check for SutazAI..."

# Run initial audit
if python3 scripts/audit_system.py; then
    echo "Initial system audit passed successfully."
else
    echo "Initial system audit failed. Running auto-fix procedures..."
    python3 scripts/auto_fix.py
    echo "Re-running system audit after auto-fix..."
    python3 scripts/audit_system.py
fi

# Organize project structure
echo "Running file organization..."
python3 scripts/organize_project.py

echo "Full system check and self-healing completed. Please review the logs in the '$LOG_DIR' directory for details."