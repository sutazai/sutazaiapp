#!/usr/bin/env bash
# Purpose: Scan for CHANGELOG.md presence in top-level project directories (read-only compliance check)
# Outputs: /reports/cleanup/changelog_compliance_<timestamp>.md

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"
REPORTS_DIR="${PROJECT_ROOT}/reports/cleanup"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S_UTC)"
OUT_MD="${REPORTS_DIR}/changelog_compliance_${TIMESTAMP}.md"

mkdir -p "${REPORTS_DIR}"

{
  echo "# CHANGELOG.md Compliance Scan"
  echo "Generated: ${TIMESTAMP}"
  echo
  echo "## Top-level directories under ${PROJECT_ROOT}"
  find "${PROJECT_ROOT}" -maxdepth 1 -mindepth 1 -type d | sort | sed 's/^/  - /'
  echo
  echo "## CHANGELOG.md presence by directory (depth 2)"
  while IFS= read -r dir; do
    has_cl="no"
    if find "$dir" -maxdepth 1 -type f -name "CHANGELOG.md" | grep -q .; then
      has_cl="yes"
    fi
    echo "- ${dir} : CHANGELOG.md=${has_cl}"
  done < <(find "${PROJECT_ROOT}" -maxdepth 2 -mindepth 1 -type d | sort)
  echo
  echo "## Notes"
  echo "- This is informational and non-binding; follow project rules for which directories require CHANGELOG.md."
} > "${OUT_MD}"

echo "Report written to: ${OUT_MD}"