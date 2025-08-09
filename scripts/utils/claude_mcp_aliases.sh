#!/usr/bin/env bash
#
# Purpose: Provide convenient shell helpers to work with Claude MCP contexts.
# Author:  DevOps Automation (Claude Code)
# Date:    2025-08-07
#
# This file is sourced from your shell profile to:
# - Quickly switch active MCP context profiles for this project.
# - Provide friendly wrappers to list/get MCP registrations.
#
# Usage:
#   source /opt/sutazaiapp/scripts/shell/claude_mcp_aliases.sh
#   mcp:ls            # list configured servers
#   mcp:get name      # show details
#   mcp:use context7  # ensure context7 is present (project-scoped)
#   mcp:use seq       # ensure sequentialthinking is present (project-scoped)
#
set -euo pipefail

_mcp_log() { echo "[mcp-shell] $*"; }

# List configured MCP servers
mcp:ls() {
  if claude mcp list --json >/dev/null 2>&1; then
    claude mcp list --json
  else
    claude mcp list
  fi
}

# Get details about one MCP server
mcp:get() {
  local name="${1:-}"; if [[ -z "$name" ]]; then echo "Usage: mcp:get <name>"; return 2; fi
  claude mcp get "$name"
}

# Ensure project-scoped registration for context7
mcp:use() {
  local which="${1:-}"; if [[ -z "$which" ]]; then echo "Usage: mcp:use <context7|seq|sequentialthinking>"; return 2; fi
  case "$which" in
    context7)
      /opt/sutazaiapp/scripts/mcp/register_mcp_contexts.sh --scope local || return $?
      _mcp_log "context7 ensured."
      ;;
    seq|sequentialthinking)
      /opt/sutazaiapp/scripts/mcp/register_mcp_contexts.sh --scope local || return $?
      _mcp_log "sequentialthinking ensured. If missing image, build it: docker build -t mcp/sequentialthinking -f servers/src/sequentialthinking/Dockerfile ."
      ;;
    *)
      echo "Unknown context alias: $which"; return 2
      ;;
  esac
}

_mcp_log "Loaded Claude MCP shell helpers. Use mcp:ls, mcp:get, mcp:use."

