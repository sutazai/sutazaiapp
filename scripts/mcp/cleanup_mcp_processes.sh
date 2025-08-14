#!/usr/bin/env bash
set -Eeuo pipefail

CAP_MCP=${MCP_LANGSERVER_MAX_INSTANCES:-6}
WORKSPACE="/opt/sutazaiapp"

echo "Cleaning up MCP processes for workspace: $WORKSPACE"

# Prune mcp-language-server (keep newest CAP_MCP)
mapfile -t pids < <(ps -eo pid=,lstart=,command= \
  | awk -v ws="$WORKSPACE" '$0 ~ /mcp-language-server/ && $0 ~ ws {pid=$1; $1=""; print pid" "$0}' \
  | awk '{print $1}' \
  | xargs -r -I{} bash -c 'printf "%s %s\n" "{}" "$(stat -c %X /proc/{}/ 2>/dev/null || echo 0)"' \
  | sort -k2,2n | awk '{print $1}')

count=${#pids[@]:-0}
if (( count > CAP_MCP )); then
  to_kill=$((count - CAP_MCP))
  echo "Pruning $to_kill stale mcp-language-server instance(s) (of $count)"
  for ((i=0; i<to_kill; i++)); do
    pid="${pids[$i]}"
    if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
      echo "Killing mcp-language-server pid=$pid"
      kill "$pid" 2>/dev/null || true
      sleep 0.2
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
else
  echo "mcp-language-server instances within cap ($count <= $CAP_MCP)"
fi

# Prune typescript-language-server (keep newest CAP_MCP*2)
mapfile -t tspids < <(ps -eo pid=,lstart=,command= \
  | awk -v ws="$WORKSPACE" '$0 ~ /typescript-language-server/ && $0 ~ ws {pid=$1; $1=""; print pid" "$0}' \
  | awk '{print $1}' \
  | xargs -r -I{} bash -c 'printf "%s %s\n" "{}" "$(stat -c %X /proc/{}/ 2>/dev/null || echo 0)"' \
  | sort -k2,2n | awk '{print $1}')

tscap=$((CAP_MCP * 2))
tscount=${#tspids[@]:-0}
if (( tscount > tscap )); then
  ts_to_kill=$((tscount - tscap))
  echo "Pruning $ts_to_kill stale typescript-language-server instance(s) (of $tscount)"
  for ((i=0; i<ts_to_kill; i++)); do
    pid="${tspids[$i]}"
    if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
      echo "Killing typescript-language-server pid=$pid"
      kill "$pid" 2>/dev/null || true
      sleep 0.2
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
else
  echo "typescript-language-server instances within cap ($tscount <= $tscap)"
fi

echo "Cleanup complete."

