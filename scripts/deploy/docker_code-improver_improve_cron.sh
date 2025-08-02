#!/bin/bash
# Autonomous code improvement cron script

# Load environment
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