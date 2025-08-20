#!/usr/bin/env bash
# Purpose: Rule 20-safe MCP audit (read-only). No modifications to .mcp.json or wrappers.
# Outputs: /reports/cleanup/mcp_readonly_selfcheck_<timestamp>.md

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"
MCP_JSON="${PROJECT_ROOT}/.mcp.json"
WRAPPERS_DIR="${PROJECT_ROOT}/scripts/mcp/wrappers"
REPORTS_DIR="${PROJECT_ROOT}/reports/cleanup"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S_UTC)"
OUT_MD="${REPORTS_DIR}/mcp_readonly_selfcheck_${TIMESTAMP}.md"

mkdir -p "${REPORTS_DIR}"

{
  echo "# MCP Read-only Self-check"
  echo "Generated: ${TIMESTAMP}"
  echo
  echo "## Rule 20 Notice"
  echo "This script is read-only and does not modify any MCP configuration or runtime."
  echo

  echo "## Config Presence"
  if [[ -f "${MCP_JSON}" ]]; then
    echo "- .mcp.json: PRESENT"
  else
    echo "- .mcp.json: MISSING"
  fi

  if [[ -d "${WRAPPERS_DIR}" ]]; then
    echo "- wrappers dir: PRESENT (${WRAPPERS_DIR})"
  else
    echo "- wrappers dir: MISSING (${WRAPPERS_DIR})"
  fi
  echo

  echo "## Declared STDIO Servers in .mcp.json (grep fallback)"
  if [[ -f "${MCP_JSON}" ]]; then
    grep -n '"type"[[:space:]]*:[[:space:]]*"stdio"' -n "${MCP_JSON}" -n | sed 's/^/  /' || true
    echo
    echo "### Commands declared"
    grep -E '"command"[[:space:]]*:' -n "${MCP_JSON}" | sed 's/^/  /' || true
  fi
  echo

  echo "## Wrapper Scripts Present"
  if [[ -d "${WRAPPERS_DIR}" ]]; then
    find "${WRAPPERS_DIR}" -maxdepth 1 -type f -name "*.sh" | sort | sed 's/^/  - /' || true
  fi
  echo

  echo "## Basic Dependency Checks (npx, node, python, jq)"
  for bin in npx node python3 jq; do
    if command -v "$bin" >/dev/null 2>&1; then
      echo "- $bin: OK ($(command -v $bin))"
    else
      echo "- $bin: MISSING"
    fi
  done
  echo

  echo "## Next Steps"
  echo "- Use existing scripts in scripts/mcp/* for deeper read-only validation (dry-run if supported)."
  echo "- Do not start/stop any MCP servers without explicit authorization."
} > "${OUT_MD}"

echo "Report written to: ${OUT_MD}"