#!/usr/bin/env bash
#
# Purpose: Persist Claude MCP shell helpers into root's shell profile idempotently.
# Usage:   ./scripts/shell/install_claude_aliases.sh
#
set -euo pipefail

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

