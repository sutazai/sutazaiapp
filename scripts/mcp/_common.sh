#!/usr/bin/env bash
set -Eeuo pipefail

log(){ printf "\033[1;36m== %s ==\033[0m\n" "$*"; }
ok(){  printf "\033[1;32m✓ %s\033[0m\n" "$*"; }
warn(){ printf "\033[1;33m! %s\033[0m\n" "$*"; }
err(){ printf "\033[1;31m✗ %s\033[0m\n" "$*"; }

has_cmd(){ command -v "$1" >/dev/null 2>&1; }

require_cmd(){ if ! has_cmd "$1"; then err "Missing command: $1"; exit 127; fi }

ts(){ date +"%Y-%m-%dT%H:%M:%S%z"; }

section(){ echo; log "$1"; }

ok_line(){ echo "[OK] $1"; }
warn_line(){ echo "[WARN] $1"; }
err_line(){ echo "[ERR] $1"; }

# Default memory cap for Node-based MCP servers (can be overridden by env)
: "${MCP_NODE_MAX_MB:=384}"
export NODE_OPTIONS="--max-old-space-size=${MCP_NODE_MAX_MB} ${NODE_OPTIONS:-}"
