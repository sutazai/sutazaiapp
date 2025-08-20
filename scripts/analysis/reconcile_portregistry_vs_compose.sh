#!/usr/bin/env bash
# Purpose: Read-only reconciliation of IMPORTANT/diagrams/PortRegistry.md vs docker compose/Dockerfiles
# Outputs: /reports/cleanup/port_registry_reconciliation_<timestamp>.md
# Safety: Read-only; does not start/stop services; no MCP changes.

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"
PORT_REGISTRY_MD="${PROJECT_ROOT}/IMPORTANT/diagrams/PortRegistry.md"
REPORTS_DIR="${PROJECT_ROOT}/reports/cleanup"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S_UTC)"
OUT_MD="${REPORTS_DIR}/port_registry_reconciliation_${TIMESTAMP}.md"

mkdir -p "${REPORTS_DIR}"

{
  echo "# Port Registry vs Compose Reconciliation"
  echo "Generated: ${TIMESTAMP}"
  echo

  if [[ ! -f "${PORT_REGISTRY_MD}" ]]; then
    echo "PortRegistry not found at ${PORT_REGISTRY_MD}. Aborting."
    exit 1
  fi

  echo "## Source of Truth"
  echo "- ${PORT_REGISTRY_MD}"
  echo

  echo "## Declared Ports in PortRegistry.md"
  grep -E "^- +[0-9]{4,5}:" -n "${PORT_REGISTRY_MD}" | sed 's/^/  /' || true
  echo

  echo "## Docker-related files (static inventory)"
  find "${PROJECT_ROOT}" -type f | \
    grep -Ei "(dockerfile$|docker-compose.*\\.ya?ml$|compose.*\\.ya?ml$|\\.dockerfile$)" | sort | sed 's/^/  - /' || true
  echo

  echo "## Compose Exposed Ports (heuristic parse)"
  COMPOSE_LIST=$(mktemp)
  find "${PROJECT_ROOT}" -type f | \
    grep -Ei "(docker-compose.*\\.ya?ml$|compose.*\\.ya?ml$)" | sort > "${COMPOSE_LIST}" || true

  if [[ -s "${COMPOSE_LIST}" ]]; then
    while IFS= read -r compose_file; do
      echo "### ${compose_file}"
      # Show services block context
      awk '/^services:/{p=1;print;next} p && /^[^[:space:]]/{p=0} p{print}' "${compose_file}" | sed 's/^/    /' || true
      # Extract ports lines
      grep -E "^[[:space:]]*- [\"']?[0-9]{2,5}:[0-9]{2,5}[\"']?" -n "${compose_file}" | sed 's/^/    /' || true
      echo
    done < "${COMPOSE_LIST}"
  else
    echo "No compose files found."
  fi
  echo

  echo "## Preliminary Findings"
  echo "- Static, read-only reconciliation. Does not verify runtime state."
  echo "- Next: correlate declared ports with compose exposure and produce mismatch table."
  echo
} > "${OUT_MD}"

echo "Report written to: ${OUT_MD}"