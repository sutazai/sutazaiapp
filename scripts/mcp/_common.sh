#!/usr/bin/env bash
set -Eeuo pipefail

# MCP Container Management
# Provides session-aware container lifecycle management to prevent accumulation

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

# MCP Session Management
# Generate unique session ID for container management
generate_session_id() {
  local session_base
  # Use combination of PID, PPID, terminal, and timestamp for uniqueness
  session_base="${PPID:-0}-${$}-$(date +%s)"
  # Include terminal if available for better session tracking
  if [ -n "${SSH_TTY:-}" ]; then
    session_base="${session_base}-$(basename "${SSH_TTY}")"
  elif [ -n "${TTY:-}" ]; then
    session_base="${session_base}-$(basename "${TTY}")"
  fi
  echo "mcp-session-${session_base}" | tr '/' '_'
}

# Get MCP container name for a session
get_mcp_container_name() {
  local service="$1"
  local session_id="${2:-$(generate_session_id)}"
  echo "${service}-${session_id}"
}

# Check if MCP container exists and is running
container_exists() {
  local container_name="$1"
  docker ps --format '{{.Names}}' | grep -qx "$container_name"
}

# Check if MCP container exists (running or stopped)
container_exists_any() {
  local container_name="$1"
  docker ps -a --format '{{.Names}}' | grep -qx "$container_name"
}

# Setup cleanup trap for MCP containers
setup_mcp_cleanup() {
  local container_name="$1"
  # Setup cleanup on script termination
  trap 'cleanup_mcp_container "'"$container_name"'"' EXIT INT TERM
}

# Cleanup MCP container
cleanup_mcp_container() {
  local container_name="$1"
  if container_exists "$container_name"; then
    log "Cleaning up MCP container: $container_name"
    docker stop "$container_name" >/dev/null 2>&1 || true
    docker rm "$container_name" >/dev/null 2>&1 || true
  fi
}

# Log container lifecycle events
log_container_event() {
  local event="$1"
  local container_name="$2"
  local message="$3"
  local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")
  local log_dir="/opt/sutazaiapp/logs/mcp"
  
  # Create log directory if it doesn't exist
  mkdir -p "$log_dir" 2>/dev/null || true
  
  # Log to file with structured format
  local log_file="$log_dir/container-lifecycle.log"
  echo "[$timestamp] [$event] [$container_name] $message" >> "$log_file" 2>/dev/null || true
  
  # Also log to syslog if available
  if has_cmd logger; then
    logger -t "mcp-container" "[$event] [$container_name] $message" 2>/dev/null || true
  fi
}
