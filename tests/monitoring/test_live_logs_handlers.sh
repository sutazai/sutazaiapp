#!/usr/bin/env bash
set -euo pipefail

export LIVE_LOGS_DRY_RUN=true
export LIVE_LOGS_SKIP_NUMLOCK=true
export LIVE_LOGS_NO_AUTORUN=true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
source "${ROOT}/scripts/monitoring/live_logs.sh"

pass=0
fail=0

require() {
  local msg="$1" ; shift || true
  local cmd="$*"
  if eval "$cmd" ; then
    echo "PASS: ${msg}"
    pass=$((pass+1))
  else
    echo "FAIL: ${msg}"
    fail=$((fail+1))
  fi
}

# show_live_logs should not block in dry-run (no real follow)
require "show_live_logs returns in dry-run" 'show_live_logs >/dev/null 2>&1'

# show_unified_live_logs should not block in dry-run
require "show_unified_live_logs returns in dry-run" 'show_unified_live_logs >/dev/null 2>&1'

# cleanup_logs dry-run path
mkdir -p "$ROOT/logs"
echo test > "$ROOT/logs/sample.test.log"
before=$(ls -1 "$ROOT/logs" | wc -l)
cleanup_logs 0d true >/dev/null 2>&1 || true
after=$(ls -1 "$ROOT/logs" | wc -l)
require "cleanup_logs dry-run does not delete files" 'test "$before" -eq "$after"'

# reset_logs confirm=true path
require "reset_logs confirm=true returns" 'reset_logs true >/dev/null 2>&1'

# toggle_debug on/off and set_log_level
require "toggle_debug on" 'toggle_debug on >/dev/null 2>&1'
require "toggle_debug off" 'toggle_debug off >/dev/null 2>&1'
require "set_log_level DEBUG" 'set_log_level DEBUG >/dev/null 2>&1'
require "set_log_level ERROR" 'set_log_level ERROR >/dev/null 2>&1'
require "set_log_level invalid fails" '! set_log_level INVALID >/dev/null 2>&1'

echo "\nSummary: ${pass} passed, ${fail} failed"
if [[ $fail -gt 0 ]]; then
  exit 1
fi
