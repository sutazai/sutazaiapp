#!/usr/bin/env bash
#
# Purpose: Persist Claude MCP shell helpers into root's shell profile idempotently.
# Usage:   ./scripts/shell/install_claude_aliases.sh
#
set -euo pipefail


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

TARGET="/root/.bashrc"
SOURCE_LINE="source /opt/sutazaiapp/scripts/shell/claude_mcp_aliases.sh"

if [[ ! -f "$TARGET" ]]; then
  echo "[install_claude_aliases] Creating $TARGET"
  touch "$TARGET"
fi

if grep -Fq "$SOURCE_LINE" "$TARGET"; then
  echo "[install_claude_aliases] Already installed in $TARGET"
else
  echo "[install_claude_aliases] Installing into $TARGET"
  printf "\n# SutazAI Claude MCP helpers\n%s\n" "$SOURCE_LINE" >> "$TARGET"
  echo "[install_claude_aliases] Installed. Open a new shell to load."
fi

