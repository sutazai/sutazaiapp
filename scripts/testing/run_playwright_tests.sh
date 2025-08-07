#!/usr/bin/env bash
#
# Purpose: Orchestrate MCP server containers and run Playwright E2E tests.
# Usage:   ./scripts/testing/run_playwright_tests.sh [--keep-up] [--compose docker/docker-compose.mcp.yml] [--contexts "context7,sequentialthinking"] [--timeout 120]
# Author:  DevOps Automation (Claude Code)
# Date:    2025-08-07
#
# Description:
# - Starts required MCP server containers (if not already running) with health/timeouts.
# - Executes Playwright test suite in a container with real-time logs.
# - Writes reports to mounted volumes for easy access.
# - Optionally keeps containers running after tests.
# - Returns clear exit codes for CI integration.
#
# Exit codes:
# - 0: success
# - 1: fatal error
# - 2: tests failed

set -euo pipefail
IFS=$'\n\t'

KEEP_UP=false
COMPOSE_FILE="docker/docker-compose.mcp.yml"
CONTEXTS="context7,sequentialthinking"
WAIT_TIMEOUT=120

log() { echo "[playwright_orchestrator] $(date '+%Y-%m-%dT%H:%M:%S%z') - $*"; }
err() { echo "[playwright_orchestrator][ERROR] $(date '+%Y-%m-%dT%H:%M:%S%z') - $*" >&2; }

usage() {
  sed -n '1,40p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-up)
      KEEP_UP=true; shift;;
    --compose)
      COMPOSE_FILE="${2:-}"; if [[ -z "$COMPOSE_FILE" ]]; then err "--compose needs value"; exit 1; fi; shift 2;;
    --contexts)
      CONTEXTS="${2:-}"; if [[ -z "$CONTEXTS" ]]; then err "--contexts needs value"; exit 1; fi; shift 2;;
    --timeout)
      WAIT_TIMEOUT="${2:-}"; if [[ -z "$WAIT_TIMEOUT" ]]; then err "--timeout needs value"; exit 1; fi; shift 2;;
    -h|--help)
      usage; exit 0;;
    *) err "Unknown arg: $1"; usage; exit 1;;
  esac
done

require_cmd() { command -v "$1" >/dev/null 2>&1 || { err "Missing command: $1"; return 1; }; }

if ! require_cmd docker || ! require_cmd docker compose; then
  err "Docker and docker compose are required."
  exit 1
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
  err "Compose file not found: $COMPOSE_FILE"
  exit 1
fi

export PLAYWRIGHT_MCP_CONTEXTS="$CONTEXTS"

log "Bringing up MCP services (compose=$COMPOSE_FILE)"
docker compose -f "$COMPOSE_FILE" up -d mcp-sequentialthinking

# Wait for container to be running
svc="sutazai-mcp-mcp-sequentialthinking-1"
deadline=$(( $(date +%s) + WAIT_TIMEOUT ))
until docker ps --format '{{.Names}} {{.Status}}' | grep -E "^${svc} .*Up" >/dev/null 2>&1; do
  if [[ $(date +%s) -ge $deadline ]]; then
    err "Timed out waiting for mcp-sequentialthinking to be Up"
    docker compose -f "$COMPOSE_FILE" logs --no-color mcp-sequentialthinking || true
    [[ "$KEEP_UP" == true ]] || docker compose -f "$COMPOSE_FILE" down -v || true
    exit 1
  fi
  sleep 2
done
log "mcp-sequentialthinking is Up"

log "Running Playwright tests"
set +e
docker compose -f "$COMPOSE_FILE" run --rm -e PLAYWRIGHT_MCP_CONTEXTS="$CONTEXTS" playwright-mcp-tests
test_exit=$?
set -e

if [[ "$KEEP_UP" == true ]]; then
  log "Keeping containers up as requested (--keep-up)"
else
  log "Tearing down containers"
  docker compose -f "$COMPOSE_FILE" down -v
fi

if [[ $test_exit -ne 0 ]]; then
  err "Playwright tests failed (exit=$test_exit)"
  exit 2
fi

log "Playwright tests completed successfully"
exit 0

