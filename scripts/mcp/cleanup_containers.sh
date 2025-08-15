#!/usr/bin/env bash
set -Eeuo pipefail

# MCP Container Cleanup Utility
# Provides background cleanup of orphaned and aged MCP containers
# Usage: cleanup_containers.sh [--daemon] [--once] [--force]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_common.sh"

# Configuration
CLEANUP_INTERVAL="${MCP_CLEANUP_INTERVAL:-300}"  # 5 minutes default
MAX_CONTAINER_AGE="${MCP_MAX_AGE:-3600}"         # 1 hour default  
DAEMON_MODE=false
FORCE_CLEANUP=false
RUN_ONCE=false
PID_FILE="/tmp/mcp-cleanup.pid"
LOG_FILE="/opt/sutazaiapp/logs/mcp/cleanup.log"

# Parse command line arguments
parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --daemon)
        DAEMON_MODE=true
        ;;
      --once)
        RUN_ONCE=true
        ;;
      --force)
        FORCE_CLEANUP=true
        ;;
      --help|-h)
        show_help
        exit 0
        ;;
      *)
        err "Unknown option: $1"
        show_help
        exit 1
        ;;
    esac
    shift
  done
}

show_help() {
  cat << EOF
MCP Container Cleanup Utility

DESCRIPTION:
    Manages cleanup of orphaned and aged MCP containers to prevent accumulation.
    
USAGE:
    $(basename "$0") [OPTIONS]

OPTIONS:
    --daemon        Run in daemon mode (continuous cleanup)
    --once          Run cleanup once and exit
    --force         Force cleanup of all MCP containers regardless of age
    --help, -h      Show this help message

EXAMPLES:
    $(basename "$0") --once              # Run cleanup once
    $(basename "$0") --daemon            # Run as daemon
    $(basename "$0") --force --once      # Force cleanup all containers

ENVIRONMENT VARIABLES:
    MCP_CLEANUP_INTERVAL    Cleanup interval in seconds (default: 300)
    MCP_MAX_AGE            Max container age in seconds (default: 3600)

EXIT CODES:
    0   Success
    1   General error
    2   Invalid arguments
    3   Already running (daemon mode)
    4   Permission denied
EOF
}

# Check if daemon is already running
check_daemon_running() {
  if [ -f "$PID_FILE" ]; then
    local pid
    pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      return 0  # Running
    else
      # Stale PID file
      rm -f "$PID_FILE" 2>/dev/null || true
      return 1  # Not running
    fi
  fi
  return 1  # Not running
}

# Start daemon mode
start_daemon() {
  if check_daemon_running; then
    err "MCP cleanup daemon is already running (PID: $(cat "$PID_FILE" 2>/dev/null))"
    exit 3
  fi
  
  # Setup logging
  mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
  
  # Daemonize
  if [ "$DAEMON_MODE" = true ]; then
    log "Starting MCP cleanup daemon with interval ${CLEANUP_INTERVAL}s"
    echo $$ > "$PID_FILE"
    
    # Setup signal handlers for graceful shutdown
    trap 'cleanup_daemon; exit 0' INT TERM
    trap 'log "Received SIGHUP, continuing"' HUP
    
    # Main daemon loop
    while true; do
      log_to_file "Daemon cleanup cycle started"
      perform_cleanup
      log_to_file "Daemon cleanup cycle completed, sleeping ${CLEANUP_INTERVAL}s"
      sleep "$CLEANUP_INTERVAL"
    done
  fi
}

# Cleanup daemon resources
cleanup_daemon() {
  log "Stopping MCP cleanup daemon"
  rm -f "$PID_FILE" 2>/dev/null || true
  log_to_file "Daemon stopped"
}

# Log to file with timestamp
log_to_file() {
  local message="$1"
  local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")
  echo "[$timestamp] [DAEMON] $message" >> "$LOG_FILE" 2>/dev/null || true
}

# Perform container cleanup
perform_cleanup() {
  local current_time=$(date +%s)
  local cleanup_count=0
  local total_containers=0
  
  log_container_event "CLEANUP_START" "system" "Starting cleanup cycle"
  
  # Cleanup aged containers
  cleanup_count=$(cleanup_aged_containers "$current_time")
  
  # Cleanup orphaned containers  
  cleanup_count=$((cleanup_count + $(cleanup_orphaned_containers)))
  
  # Cleanup unnamed/unlabeled postgres-mcp containers (legacy)  
  legacy_cleanup=$(cleanup_legacy_containers)
  cleanup_count=$((cleanup_count + legacy_cleanup))
  
  # Count total postgres-mcp containers after cleanup
  total_containers=$(docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" | wc -l)
  
  local message="Cleaned up $cleanup_count containers, $total_containers remain"
  log "$message"
  log_container_event "CLEANUP_END" "system" "$message"
}

# Cleanup containers based on age
cleanup_aged_containers() {
  local current_time="$1"
  local count=0
  
  # Get all postgres-mcp containers with labels
  docker ps -a --filter="label=mcp-service=postgres" --format="{{.Names}} {{.Label \"mcp-started\"}}" | while IFS=' ' read -r container_name started_time; do
    if [ -n "$started_time" ] && [ "$started_time" != "<no value>" ]; then
      local age=$((current_time - started_time))
      if [ $age -gt $MAX_CONTAINER_AGE ] || [ "$FORCE_CLEANUP" = true ]; then
        log_container_event "CLEANUP" "$container_name" "Removing aged container (${age}s old)"
        if docker stop "$container_name" >/dev/null 2>&1; then
          docker rm "$container_name" >/dev/null 2>&1 || true
          count=$((count + 1))
        fi
      fi
    fi
  done 2>/dev/null
  
  echo $count
}

# Cleanup orphaned containers (parent process dead)
cleanup_orphaned_containers() {
  local count=0
  
  # Get all postgres-mcp containers with PID labels
  docker ps --filter="label=mcp-service=postgres" --format="{{.Names}} {{.Label \"mcp-pid\"}}" | while IFS=' ' read -r container_name session_pid; do
    if [ -n "$session_pid" ] && [ "$session_pid" != "<no value>" ]; then
      # Check if the process is still alive
      if ! kill -0 "$session_pid" 2>/dev/null || [ "$FORCE_CLEANUP" = true ]; then
        log_container_event "CLEANUP" "$container_name" "Removing orphaned container (PID $session_pid dead)"
        if docker stop "$container_name" >/dev/null 2>&1; then
          docker rm "$container_name" >/dev/null 2>&1 || true
          count=$((count + 1))
        fi
      fi
    fi
  done 2>/dev/null
  
  echo $count
}

# Cleanup legacy containers without proper labeling
cleanup_legacy_containers() {
  local count=0
  local temp_file="/tmp/mcp-cleanup-$$"
  
  # Get all postgres-mcp containers and process them
  docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" > "$temp_file" 2>/dev/null || true
  
  while read -r container_name; do
    if [ -n "$container_name" ]; then
      # Check if it has our service label
      local has_service_label=$(docker inspect "$container_name" --format "{{index .Config.Labels \"mcp-service\"}}" 2>/dev/null || echo "")
      
      if [ "$has_service_label" != "postgres" ]; then
        # This is a legacy container without proper labeling
        if [ "$FORCE_CLEANUP" = true ]; then
          log_container_event "CLEANUP" "$container_name" "Removing legacy unlabeled container (force mode)"
          if docker stop "$container_name" >/dev/null 2>&1; then
            if docker rm "$container_name" >/dev/null 2>&1; then
              count=$((count + 1))
            fi
          fi
        else
          # For non-force cleanup, check if container is old enough
          local created_time=$(docker inspect "$container_name" --format "{{.Created}}" 2>/dev/null || echo "")
          if [ -n "$created_time" ]; then
            local created_epoch=$(date -d "$created_time" +%s 2>/dev/null || echo "0")
            local current_time=$(date +%s)
            local age=$((current_time - created_epoch))
            
            if [ $age -gt $MAX_CONTAINER_AGE ]; then
              log_container_event "CLEANUP" "$container_name" "Removing aged legacy container (${age}s old)"
              if docker stop "$container_name" >/dev/null 2>&1; then
                if docker rm "$container_name" >/dev/null 2>&1; then
                  count=$((count + 1))
                fi
              fi
            fi
          fi
        fi
      fi
    fi
  done < "$temp_file"
  
  # Cleanup temp file
  rm -f "$temp_file" 2>/dev/null || true
  
  echo $count
}

# Show current MCP container status
show_status() {
  section "MCP Container Status $(ts)"
  
  local total_containers=$(docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" | wc -l)
  local running_containers=$(docker ps --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" | wc -l)
  
  ok_line "Total postgres-mcp containers: $total_containers"
  ok_line "Running postgres-mcp containers: $running_containers"
  
  if [ $total_containers -gt 0 ]; then
    echo ""
    echo "Container Details:"
    docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}\t{{.Labels}}" | head -20
  fi
  
  if check_daemon_running; then
    ok_line "Cleanup daemon is running (PID: $(cat "$PID_FILE" 2>/dev/null))"
  else
    warn_line "Cleanup daemon is not running"
  fi
}

# Main execution
main() {
  parse_args "$@"
  
  # Check docker availability
  if ! has_cmd docker; then
    err "Docker is required for MCP container cleanup"
    exit 1
  fi
  
  # Handle different modes
  if [ "$RUN_ONCE" = true ]; then
    log "Running one-time cleanup"
    perform_cleanup
  elif [ "$DAEMON_MODE" = true ]; then
    start_daemon
  else
    # Default: show status and run cleanup once
    show_status
    echo ""
    log "Running one-time cleanup"
    perform_cleanup
  fi
}

# Execute main function
main "$@"