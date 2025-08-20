# Live Logs Sanity (Dry)
Generated: 20250819_214540_UTC

- live_logs.sh: PRESENT

## --help / usage (first lines)
  #!/bin/bash
  # Purpose: SutazAI Live Logs Management System - Comprehensive log monitoring and management
  # Usage: ./live_logs.sh [live|follow|cleanup|reset|debug|config|status] [OPTIONS]
  # Requires: docker, docker-compose, system utilities
  
  set -euo pipefail
  
  # Optional: test-friendly dry-run mode and non-interactive env guards
  # - LIVE_LOGS_DRY_RUN=true      -> intercepts docker/systemctl/journalctl/sudo/pkill/dockerd
  # - LIVE_LOGS_SKIP_NUMLOCK=true -> skips numlock manipulation for CI/containers
  
  # DRY-RUN command shims (functions override binaries within this script scope)
  if [[ "${LIVE_LOGS_DRY_RUN:-}" == "true" ]]; then
      docker() { echo "[DRY_RUN] docker $*"; return 0; }
      sudo() { echo "[DRY_RUN] sudo $*"; return 0; }
      systemctl() { echo "[DRY_RUN] systemctl $*"; return 0; }
      journalctl() { echo "[DRY_RUN] journalctl $*"; return 0; }
      pkill() { echo "[DRY_RUN] pkill $*"; return 0; }
      dockerd() { echo "[DRY_RUN] dockerd $*"; return 0; }
  fi

## Dry-run status output (status)
  [0;34m╔══════════════════════════════════════════════════════════════╗[0m
  [0;34m║[0m               [0;36mSutazAI Live Logs Management System[0m              [0;34m║[0m
  [0;34m╚══════════════════════════════════════════════════════════════╝[0m
  
  [1;33mConfiguration:[0m
    Debug Mode: false
    Log Level: ERROR
    Max Log Size: 100M
    Max Log Files: 10
    Cleanup Days: 7
  
  [0;36mLog Status and Disk Usage:[0m
  
  [1;33mLog Directory Size:[0m 52K
  [1;33mLog Directory Path:[0m /opt/sutazaiapp/logs
  [1;33mTotal Log Files:[0m 4
  
  [0;36mContainer Log Status:[0m
  
  [0;36mSystem Disk Usage:[0m
    Root: 55G used, 901G available (6% full)
  [0;36mDocker Disk Usage:[0m

