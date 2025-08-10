#!/bin/bash

# Strict error handling
set -euo pipefail

# Autonomous code improvement cron script

# Load environment

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

export PATH=/usr/local/bin:/usr/bin:/bin
cd /opt/sutazaiapp

# Log file
LOG_FILE="/app/logs/improvement_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /app/logs

echo "[$(date)] Starting autonomous code improvement..." >> "$LOG_FILE"

# Trigger improvement via API
curl -X POST http://localhost:8080/improve \
  -H "Content-Type: application/json" \
  -d '{
    "target_path": "/opt/sutazaiapp",
    "file_patterns": ["*.py"],
    "improvement_types": ["security", "optimize", "refactor"],
    "require_approval": true,
    "agents": ["semgrep", "pylint", "black"]
  }' >> "$LOG_FILE" 2>&1

echo "[$(date)] Improvement cycle completed" >> "$LOG_FILE"