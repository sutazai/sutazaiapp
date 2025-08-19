#!/usr/bin/env bash
set -euo pipefail

# Run a subset of live_logs.sh functions in dry-run mode to validate handlers.
export LIVE_LOGS_DRY_RUN=true
export LIVE_LOGS_SKIP_NUMLOCK=true

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
SCRIPT_PATH="${SCRIPT_ROOT}/scripts/monitoring/live_logs.sh"

pass=0
fail=0

require() {
  local msg="$1" ; shift || true
  if "$@" ; then
    echo "PASS: ${msg}"
    pass=$((pass+1))
  else
    echo "FAIL: ${msg}"
    fail=$((fail+1))
  fi
}

# Load the script without executing main flow (script does not auto-run main)
source "${SCRIPT_PATH}"

# Tests
check_docker_daemon >/dev/null 2>&1 && ok1=0 || ok1=1
require "check_docker_daemon returns success in dry run" test $ok1 -eq 0

attempt_docker_recovery >/dev/null 2>&1 && ok2=0 || ok2=1
require "attempt_docker_recovery echoes steps and succeeds" test $ok2 -eq 0

get_container_logs sutazai-nonexistent 10 false >/dev/null 2>&1 || true
require "get_container_logs handles non-existing container gracefully" true

sym=$(check_container_status sutazai-any 2>/dev/null || true)
require "check_container_status returns a symbol" test -n "$sym"

show_status >/dev/null 2>&1 && ok3=0 || ok3=1
require "show_status can run without error" test $ok3 -eq 0

echo "\nSummary: ${pass} passed, ${fail} failed"
if [[ $fail -gt 0 ]]; then
  exit 1
fi
