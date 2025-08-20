#!/usr/bin/env bash
# Purpose: Static inventory of docker-related files (read-only)
# Outputs: /reports/cleanup/docker_inventory_<timestamp>.md

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"
REPORTS_DIR="${PROJECT_ROOT}/reports/cleanup"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S_UTC)"
OUT_MD="${REPORTS_DIR}/docker_inventory_${TIMESTAMP}.md"

mkdir -p "${REPORTS_DIR}"

{
  echo "# Docker Inventory"
  echo "Generated: ${TIMESTAMP}"
  echo
  echo "## Docker-related files"
  find "${PROJECT_ROOT}" -type f | \
    grep -Ei "(dockerfile$|docker-compose.*\\.ya?ml$|compose.*\\.ya?ml$|\\.dockerfile$)" | sort | sed 's/^/  - /' || true
  echo
  echo "## Notes"
  echo "- Inventory only; not validated for active use."
} > "${OUT_MD}"

echo "Report written to: ${OUT_MD}"