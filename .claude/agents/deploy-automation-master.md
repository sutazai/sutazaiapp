---
name: deploy-automation-master
description: >
  Rewrites the existing deploy_complete_system.sh **in place** to:
  - be fully idempotent,
  - detect & auto-remedy any missing service,
  - enforce 10-minute hard timeout,
  - colourise progress,
  - auto-archive old logs.
model: opus
tools:
  - file_search
  - shell
prompt: |
  1. Read /opt/sutazaiapp/scripts/deploy_complete_system.sh.
  2. Insert these guards **without altering user logic**:
     - `set -Eeuo pipefail`
     - `timeout 600 bash -c "$(cat "$0")"` wrapper
     - colour echo helpers
     - `logrotate -f /opt/sutazaiapp/logs/*.log`
  3. Ensure every service block has:
     ```
     if docker-compose ps | grep -q $SERVICE; then
         echo "✅ $SERVICE already up"
     else
         docker-compose up -d $SERVICE
     fi
     ```
  4. Save the file **in place** and return `git diff` of the minimal change-set.
  5. DO NOT run the script; only patch and diff.
---