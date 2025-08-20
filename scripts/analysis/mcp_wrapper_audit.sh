#!/usr/bin/env bash
# Purpose: Read-only audit of MCP wrappers declared in .mcp.json
# Output: /reports/cleanup/mcp_wrapper_audit_<timestamp>.md
# Safety: Rule 20 compliant (no modifications)

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"
MCP_JSON="${PROJECT_ROOT}/.mcp.json"
WRAPPERS_DIR="${PROJECT_ROOT}/scripts/mcp/wrappers"
REPORTS_DIR="${PROJECT_ROOT}/reports/cleanup"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S_UTC)"
OUT_MD="${REPORTS_DIR}/mcp_wrapper_audit_${TIMESTAMP}.md"

mkdir -p "${REPORTS_DIR}"

# Extract command paths from .mcp.json (simple grep/sed parsing)
mapfile -t COMMANDS < <(grep -E '"command"[[:space:]]*:' -n "$MCP_JSON" | sed -E 's/.*"command"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/' || true)

{
  echo "# MCP Wrapper Audit (Read-only)"
  echo "Generated: ${TIMESTAMP}"
  echo
  echo "## Rule 20 Notice"
  echo "Do not modify MCP configuration or wrappers based on this report without explicit authorization."
  echo
  echo "## .mcp.json Command Entries"
  if [[ ${#COMMANDS[@]:-0} -gt 0 ]]; then
    for c in "${COMMANDS[@]}"; do echo "  - ${c}"; done
  else
    echo "  - (none found)"
  fi
  echo
  echo "## Wrapper Files Presence & Permissions"
  if [[ -d "$WRAPPERS_DIR" ]]; then
    for c in "${COMMANDS[@]:-}"; do
      if [[ "$c" == /*.sh && -f "$c" ]]; then
        bn=$(basename "$c")
        perm=$(stat -c %A "$c" 2>/dev/null || echo "?")
        execbit="no"
        [[ -x "$c" ]] && execbit="yes"
        shebang=$(head -n 1 "$c" 2>/dev/null || echo "")
        echo "- ${bn}: present, perms=${perm}, executable=${execbit}, shebang='${shebang}'"
      elif [[ "$c" == /*.sh ]]; then
        echo "- $(basename "$c"): MISSING (${c})"
      else
        # Non-wrapper commands (e.g., npx) — report presence
        if command -v "$c" >/dev/null 2>&1; then
          echo "- ${c}: FOUND in PATH ($(command -v "$c"))"
        else
          echo "- ${c}: NOT FOUND in PATH"
        fi
      fi
    done
  else
    echo "- Wrappers directory missing: ${WRAPPERS_DIR}"
  fi
  echo
  echo "## Suggested Follow-ups (no changes performed)"
  echo "- Ensure all wrapper scripts are executable and have proper shebang."
  echo "- Verify each wrapper can start its MCP server in an isolated dry-run environment."
  echo "- Attach this audit to change proposals per Rule 20."
} > "$OUT_MD"

echo "Report written to: ${OUT_MD}"