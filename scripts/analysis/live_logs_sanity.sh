#!/usr/bin/env bash
# Purpose: Validate presence and dry-run status of live_logs.sh without altering system
# Outputs: /reports/cleanup/live_logs_sanity_<timestamp>.md

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"
LIVE_LOGS="${PROJECT_ROOT}/scripts/monitoring/live_logs.sh"
REPORTS_DIR="${PROJECT_ROOT}/reports/cleanup"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S_UTC)"
OUT_MD="${REPORTS_DIR}/live_logs_sanity_${TIMESTAMP}.md"

mkdir -p "${REPORTS_DIR}"

{
  echo "# Live Logs Sanity (Dry)"
  echo "Generated: ${TIMESTAMP}"
  echo
  if [[ -f "${LIVE_LOGS}" ]]; then
    echo "- live_logs.sh: PRESENT"
  else
    echo "- live_logs.sh: MISSING (${LIVE_LOGS})"
    exit 0
  fi
  echo
  echo "## --help / usage (first lines)"
  head -n 20 "${LIVE_LOGS}" | sed 's/^/  /'
  echo
  echo "## Dry-run status output (status)"
  LIVE_LOGS_DRY_RUN=true LIVE_LOGS_NONINTERACTIVE=true bash -c "\"${LIVE_LOGS}\" status" 2>&1 | sed 's/^/  /' || true
  echo
} > "${OUT_MD}"

echo "Report written to: ${OUT_MD}"}  
