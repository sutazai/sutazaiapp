#!/usr/bin/env bash
# Purpose: Discover Playwright config and test roots (read-only)
# Outputs: /reports/cleanup/playwright_discovery_<timestamp>.md

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"
REPORTS_DIR="${PROJECT_ROOT}/reports/cleanup"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S_UTC)"
OUT_MD="${REPORTS_DIR}/playwright_discovery_${TIMESTAMP}.md"

mkdir -p "${REPORTS_DIR}"

CONFIG_PATHS=$(find "${PROJECT_ROOT}" -maxdepth 3 -type f -name "playwright.config.ts" -o -name "playwright.config.js" | sort)

{
  echo "# Playwright Discovery"
  echo "Generated: ${TIMESTAMP}"
  echo
  if [[ -z "${CONFIG_PATHS}" ]]; then
    echo "No Playwright config files found."
    exit 0
  fi

  echo "## Config Files"
  echo "${CONFIG_PATHS}" | sed 's/^/  - /'
  echo

  echo "## Extracted testDir (heuristic)"
  while IFS= read -r cfg; do
    echo "### ${cfg}"
    # heuristic: find testDir assignments
    grep -E "testDir|projects|testMatch" -n "${cfg}" | sed 's/^/    /' || true
    echo
  done <<< "${CONFIG_PATHS}"

  echo "## Candidate test files (ts/js)"
  find "${PROJECT_ROOT}" -type f \( -iname "*.spec.ts" -o -iname "*.test.ts" -o -iname "*.spec.js" -o -iname "*.test.js" \) | sort | sed 's/^/  - /' || true
  echo
} > "${OUT_MD}"

echo "Report written to: ${OUT_MD}"