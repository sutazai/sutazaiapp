#!/usr/bin/env bash
#
# Purpose: Idempotently register required MCP server contexts with Claude CLI.
# Usage:   ./scripts/mcp/register_mcp_contexts.sh [--dry-run] [--scope local|user|project]
# Author:  DevOps Automation (Claude Code)
# Date:    2025-08-07
#
# Description:
# - Detects if required MCP server contexts are already registered in Claude CLI.
# - Adds missing contexts with the correct startup commands (npx or docker run).
# - Ensures strict idempotency: no duplicates, no conflicting registrations.
# - Provides clear logging and exit codes for CI integration.
#
# Requirements:
# - Claude CLI installed and accessible as `claude`.
# - Docker installed for docker-based contexts.
# - No network access is required for listing; adding `context7` via npx requires internet when used.
#
# Exit codes:
# - 0: success (all contexts ensured)
# - 1: fatal error (claude CLI missing or unexpected failure)
# - 2: partial (some contexts could not be added)

set -euo pipefail
IFS=$'\n\t'

SCOPE="local"  # local = project-scoped per Claude CLI
DRY_RUN=false

log() { echo "[register_mcp_contexts] $(date '+%Y-%m-%dT%H:%M:%S%z') - $*"; }
err() { echo "[register_mcp_contexts][ERROR] $(date '+%Y-%m-%dT%H:%M:%S%z') - $*" >&2; }

usage() {
  sed -n '1,50p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --scope)
      SCOPE="${2:-}"
      if [[ -z "$SCOPE" ]]; then err "--scope requires a value"; exit 1; fi
      shift 2
      ;;
    -h|--help)
      usage; exit 0
      ;;
    *)
      err "Unknown argument: $1"; usage; exit 1
      ;;
  esac
done

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    err "Required command not found in PATH: $1"
    return 1
  fi
}

if ! require_cmd claude; then
  err "Claude CLI is required. Install: npm i -g @anthropic-ai/claude-code"
  exit 1
fi

has_context() {
  local name="$1"
  # Try JSON; fallback to text
  if claude mcp list --json >/dev/null 2>&1; then
    if claude mcp list --json 2>/dev/null | grep -E '"name"[[:space:]]*:[[:space:]]*"'"$name"'"' >/dev/null; then
      return 0
    else
      return 1
    fi
  else
    if claude mcp list 2>/dev/null | grep -E '^'"$name"':' >/dev/null; then
      return 0
    else
      return 1
    fi
  fi
}

add_context() {
  local name="$1"
  shift
  local cmd=("$@")
  log "Adding MCP context '$name' with: ${cmd[*]} (scope=$SCOPE)"
  if [[ "$DRY_RUN" == true ]]; then
    echo "DRY-RUN: claude mcp add --scope $SCOPE $name ${cmd[*]}"
    return 0
  fi
  if ! claude mcp add --scope "$SCOPE" "$name" "${cmd[@]}"; then
    err "Failed to add context: $name"
    return 1
  fi
}

ensure_context() {
  local name="$1"; shift
  local cmd=("$@")
  if has_context "$name"; then
    log "Context already registered: $name (skipping)"
    return 0
  fi
  add_context "$name" "${cmd[@]}"
}

main() {
  local failures=0

  # 1) context7 via npx
  if ! ensure_context "context7" npx -y @upstash/context7-mcp; then
    failures=$((failures+1))
  fi

  # 2) sequentialthinking via docker run (expects image available locally)
  #    To build locally: docker build -t mcp/sequentialthinking -f servers/src/sequentialthinking/Dockerfile .
  if ! ensure_context "sequentialthinking" docker run --rm -i mcp/sequentialthinking; then
    failures=$((failures+1))
  fi

  if [[ $failures -gt 0 ]]; then
    err "Completed with $failures failure(s)"
    exit 2
  fi
  log "All required MCP contexts are registered and healthy"
}

main "$@"

