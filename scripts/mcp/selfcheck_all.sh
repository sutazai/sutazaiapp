#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/opt/sutazaiapp"
WRAP="$ROOT/scripts/mcp/wrappers"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
OUT="$LOG_DIR/mcp_selfcheck_$(date +%Y%m%d_%H%M%S).log"

servers=(
  "files"
  "context7"
  "http_fetch"
  "ddg"
  "sequentialthinking"
  "nx-mcp"
  "extended-memory"
  "mcp_ssh"
  "ultimatecoder"
  "postgres"
  "playwright-mcp"
  "memory-bank-mcp"
  "puppeteer-mcp (no longer in use)"
  "knowledge-graph-mcp"
  "compass-mcp"
  "claude-flow"
  "ruv-swarm"
  "claude-task-runner"
)

declare -A map
map=(
  [files]="$WRAP/files.sh"
  [context7]="$WRAP/context7.sh"
  [http_fetch]="$WRAP/http_fetch.sh"
  [ddg]="$WRAP/ddg.sh"
  [sequentialthinking]="$WRAP/sequentialthinking.sh"
  [nx-mcp]="$WRAP/nx-mcp.sh"
  [extended-memory]="$WRAP/extended-memory.sh"
  [mcp_ssh]="$WRAP/mcp_ssh.sh"
  [ultimatecoder]="$WRAP/ultimatecoder.sh"
  [postgres]="$WRAP/postgres.sh"
  [playwright-mcp]="$WRAP/playwright-mcp.sh"
  [memory-bank-mcp]="$WRAP/memory-bank-mcp.sh"
  [puppeteer-mcp (no longer in use)]="$WRAP/puppeteer-mcp (no longer in use).sh"
  [knowledge-graph-mcp]="$WRAP/knowledge-graph-mcp.sh"
  [compass-mcp]="$WRAP/compass-mcp.sh"
  [claude-flow]="$WRAP/claude-flow.sh"
  [ruv-swarm]="$WRAP/ruv-swarm.sh"
  [claude-task-runner]="$WRAP/claude-task-runner.sh"
)

echo "MCP selfcheck started at $(date)" | tee "$OUT"
FAIL=0
for s in "${servers[@]}"; do
  bin="${map[$s]}"
  if [ ! -x "$bin" ]; then
    echo "[ERR] $s: wrapper not found at $bin" | tee -a "$OUT"
    FAIL=1
    continue
  fi
  echo "--- $s ---" | tee -a "$OUT"
  if "$bin" --selfcheck >>"$OUT" 2>&1; then
    echo "[OK] $s selfcheck passed" | tee -a "$OUT"
  else
    echo "[ERR] $s selfcheck failed (see log)" | tee -a "$OUT"
    FAIL=1
  fi
done

echo "MCP selfcheck finished at $(date)" | tee -a "$OUT"
echo "Report: $OUT"
exit $FAIL

