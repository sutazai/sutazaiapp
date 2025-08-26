#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../_common.sh"

# Wrapper for mcp-language-server to control memory and process count.
# - Caps Node/TS server memory via NODE_OPTIONS
# - Prunes excess/old mcp-language-server instances for this workspace

WORKSPACE="/opt/sutazaiapp"
MCP_BIN="/root/go/bin/mcp-language-server"
TS_LSP="$(which typescript-language-server 2>/dev/null || echo /usr/local/bin/typescript-language-server)"

MAX_INSTANCES="${MCP_LANGSERVER_MAX_INSTANCES:-6}"
MEM_MB="${MCP_LANGSERVER_NODE_MAX_MB:-512}"

if [ "${1:-}" = "--selfcheck" ]; then
  section "language-server selfcheck $(ts)"
  has_cmd "$MCP_BIN" && ok_line "mcp-language-server present" || { err_line "mcp-language-server missing at $MCP_BIN"; exit 127; }
  has_cmd "$TS_LSP" && ok_line "typescript-language-server present" || { err_line "typescript-language-server missing"; exit 127; }
  ok_line "will cap Node memory to ${MEM_MB}MB and limit to ${MAX_INSTANCES} instances"
  exit 0
fi

prune_old_instances(){
  # Keep at most MAX_INSTANCES language-servers for this workspace; kill oldest beyond the cap
  # Filter strictly by command line matching our workspace to avoid collateral damage
  mapfile -t pids < <(ps -eo pid=,lstart=,command= \
    | awk -v ws="$WORKSPACE" '$0 ~ /mcp-language-server/ && $0 ~ ws {pid=$1; $1=""; print pid" "$0}' \
    | awk '{print $1}' \
    | xargs -r -I{} bash -c 'printf "%s %s\n" "{}" "$(stat -c %X /proc/{}/ 2>/dev/null || echo 0)"' \
    | sort -k2,2n | awk '{print $1}')

  local count=${#pids[@]}
  if (( count > MAX_INSTANCES )); then
    local to_kill=$((count - MAX_INSTANCES))
    warn "Pruning $to_kill stale mcp-language-server instance(s) (of $count)"
    for ((i=0; i<to_kill; i++)); do
      local pid="${pids[$i]}"
      if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
        warn_line "Killing stale mcp-language-server pid=$pid"
        kill "$pid" 2>/dev/null || true
        sleep 0.2
        kill -9 "$pid" 2>/dev/null || true
      fi
    done
  fi

  # Also prune orphaned typescript-language-server tied to this workspace exceeding a soft cap
  mapfile -t tspids < <(ps -eo pid=,command= | awk -v ws="$WORKSPACE" '$0 ~ /typescript-language-server/ && $0 ~ ws {print $1}')
  local ts_count=${#tspids[@]}
  local ts_cap=$(( MAX_INSTANCES * 2 ))
  if (( ts_count > ts_cap )); then
    local ts_to_kill=$((ts_count - ts_cap))
    warn "Pruning $ts_to_kill stale typescript-language-server instance(s) (of $ts_count)"
    for pid in "${tspids[@]:0:ts_to_kill}"; do
      kill "$pid" 2>/dev/null || true
      sleep 0.1
      kill -9 "$pid" 2>/dev/null || true
    done
  fi
}

# Housekeeping before starting a new instance
prune_old_instances || true

# Cap Node memory for LSP child processes
export NODE_OPTIONS="--max-old-space-size=${MEM_MB} ${NODE_OPTIONS:-}"

# Exec the real server with original args (ensure required defaults if none passed)
exec "$MCP_BIN" --workspace "$WORKSPACE" --lsp "$TS_LSP" -- --stdio

