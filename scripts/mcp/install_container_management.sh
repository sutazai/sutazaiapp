#!/usr/bin/env bash
set -Eeuo pipefail

# MCP Container Management Installation Script
# Installs and configures session-aware container lifecycle management

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_common.sh"

SYSTEMD_SERVICE_FILE="mcp-cleanup.service"
SYSTEMD_TARGET_DIR="/etc/systemd/system"
LOG_DIR="/opt/sutazaiapp/logs/mcp"

show_help() {
  cat << EOF
MCP Container Management Installation Script

DESCRIPTION:
    Installs and configures session-aware MCP container lifecycle management
    to prevent container accumulation and improve resource efficiency.

USAGE:
    $(basename "$0") [COMMAND] [OPTIONS]

COMMANDS:
    install         Install container management system
    uninstall       Remove container management system  
    status          Show current system status
    start           Start cleanup daemon
    stop            Stop cleanup daemon
    restart         Restart cleanup daemon
    force-cleanup   Force cleanup all MCP containers
    
OPTIONS:
    --help, -h      Show this help message
    --daemon        Enable daemon mode for cleanup
    --no-daemon     Disable daemon mode

EXAMPLES:
    $(basename "$0") install                    # Install with daemon
    $(basename "$0") install --no-daemon        # Install without daemon
    $(basename "$0") status                     # Check system status
    $(basename "$0") force-cleanup              # Emergency cleanup

EXIT CODES:
    0   Success
    1   General error
    2   Invalid arguments
    3   Permission denied
    4   Service error
EOF
}

# Check if we're running as root or with sudo
check_permissions() {
  if [ "$EUID" -ne 0 ] && ! groups | grep -q docker; then
    err "This script requires root privileges or docker group membership"
    err "Run with sudo or add user to docker group: sudo usermod -aG docker \$USER"
    exit 3
  fi
}

# Install container management system
install_system() {
  local enable_daemon=${1:-true}
  
  section "Installing MCP Container Management System $(ts)"
  
  # Create required directories
  ok "Creating directories..."
  mkdir -p "$LOG_DIR" || true
  mkdir -p "/tmp/mcp-locks" || true
  
  # Set permissions
  chmod 755 "$SCRIPT_DIR/cleanup_containers.sh"
  chmod 644 "$SCRIPT_DIR/$SYSTEMD_SERVICE_FILE" 2>/dev/null || true
  
  if [ "$enable_daemon" = true ]; then
    # Install systemd service
    if [ -d "$SYSTEMD_TARGET_DIR" ] && command -v systemctl >/dev/null 2>&1; then
      ok "Installing systemd service..."
      cp "$SCRIPT_DIR/$SYSTEMD_SERVICE_FILE" "$SYSTEMD_TARGET_DIR/"
      systemctl daemon-reload
      systemctl enable mcp-cleanup.service
      systemctl start mcp-cleanup.service
      ok "MCP cleanup daemon enabled and started"
    else
      warn "Systemd not available, daemon mode disabled"
      enable_daemon=false
    fi
  fi
  
  # Test the installation
  ok "Testing installation..."
  if "$SCRIPT_DIR/cleanup_containers.sh" --once; then
    ok "Container management system installed successfully"
  else
    err "Installation test failed"
    return 1
  fi
  
  # Show status
  show_system_status
  
  ok "Installation completed successfully"
  if [ "$enable_daemon" = true ]; then
    ok "Cleanup daemon is running and will maintain container hygiene automatically"
  else
    warn "Daemon mode disabled - run cleanup manually with: $SCRIPT_DIR/cleanup_containers.sh --once"
  fi
}

# Uninstall container management system
uninstall_system() {
  section "Uninstalling MCP Container Management System $(ts)"
  
  # Stop and disable systemd service
  if systemctl is-active mcp-cleanup.service >/dev/null 2>&1; then
    ok "Stopping MCP cleanup daemon..."
    systemctl stop mcp-cleanup.service
  fi
  
  if systemctl is-enabled mcp-cleanup.service >/dev/null 2>&1; then
    ok "Disabling MCP cleanup daemon..."
    systemctl disable mcp-cleanup.service
  fi
  
  # Remove systemd service file
  if [ -f "$SYSTEMD_TARGET_DIR/$SYSTEMD_SERVICE_FILE" ]; then
    ok "Removing systemd service file..."
    rm -f "$SYSTEMD_TARGET_DIR/$SYSTEMD_SERVICE_FILE"
    systemctl daemon-reload
  fi
  
  # Clean up any remaining MCP containers (with confirmation)
  local mcp_containers=$(docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" | wc -l)
  if [ $mcp_containers -gt 0 ]; then
    warn "Found $mcp_containers MCP containers"
    echo "Remove all MCP containers? [y/N]: "
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
      "$SCRIPT_DIR/cleanup_containers.sh" --force --once
    fi
  fi
  
  ok "Uninstallation completed"
}

# Show system status
show_system_status() {
  section "MCP Container Management Status $(ts)"
  
  # Check if daemon is running
  if systemctl is-active mcp-cleanup.service >/dev/null 2>&1; then
    ok_line "Cleanup daemon: Running"
  elif [ -f "/tmp/mcp-cleanup.pid" ]; then
    local pid=$(cat "/tmp/mcp-cleanup.pid" 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      ok_line "Cleanup daemon: Running (manual mode, PID: $pid)"
    else
      warn_line "Cleanup daemon: Stopped (stale PID file)"
      rm -f "/tmp/mcp-cleanup.pid" 2>/dev/null || true
    fi
  else
    warn_line "Cleanup daemon: Stopped"
  fi
  
  # Show container statistics
  local total_containers=$(docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" | wc -l)
  local running_containers=$(docker ps --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" | wc -l)
  
  if [ $total_containers -eq 0 ]; then
    ok_line "MCP containers: None"
  elif [ $total_containers -le 3 ]; then
    ok_line "MCP containers: $total_containers total, $running_containers running (healthy)"
  else
    warn_line "MCP containers: $total_containers total, $running_containers running (consider cleanup)"
  fi
  
  # Show recent log entries if available
  if [ -f "$LOG_DIR/container-lifecycle.log" ]; then
    local recent_logs=$(tail -3 "$LOG_DIR/container-lifecycle.log" 2>/dev/null | wc -l)
    ok_line "Log entries: $recent_logs recent events"
  fi
  
  # Check enhanced postgres.sh features
  if grep -q "session-aware container management" "$SCRIPT_DIR/wrappers/postgres.sh" 2>/dev/null; then
    ok_line "Enhanced postgres.sh: Installed"
  else
    err_line "Enhanced postgres.sh: Not installed"
  fi
}

# Start cleanup daemon
start_daemon() {
  if systemctl is-active mcp-cleanup.service >/dev/null 2>&1; then
    ok "MCP cleanup daemon is already running"
  else
    ok "Starting MCP cleanup daemon..."
    systemctl start mcp-cleanup.service
    sleep 2
    if systemctl is-active mcp-cleanup.service >/dev/null 2>&1; then
      ok "MCP cleanup daemon started successfully"
    else
      err "Failed to start MCP cleanup daemon"
      systemctl status mcp-cleanup.service
      exit 4
    fi
  fi
}

# Stop cleanup daemon
stop_daemon() {
  if systemctl is-active mcp-cleanup.service >/dev/null 2>&1; then
    ok "Stopping MCP cleanup daemon..."
    systemctl stop mcp-cleanup.service
    ok "MCP cleanup daemon stopped"
  else
    warn "MCP cleanup daemon is not running"
  fi
}

# Restart cleanup daemon
restart_daemon() {
  ok "Restarting MCP cleanup daemon..."
  systemctl restart mcp-cleanup.service
  sleep 2
  if systemctl is-active mcp-cleanup.service >/dev/null 2>&1; then
    ok "MCP cleanup daemon restarted successfully"
  else
    err "Failed to restart MCP cleanup daemon"
    exit 4
  fi
}

# Force cleanup all containers
force_cleanup() {
  warn "This will remove ALL postgres-mcp containers"
  echo "Continue? [y/N]: "
  read -r response
  if [[ "$response" =~ ^[Yy]$ ]]; then
    "$SCRIPT_DIR/cleanup_containers.sh" --force --once
  else
    ok "Cleanup cancelled"
  fi
}

# Main execution
main() {
  local command="${1:-status}"
  local enable_daemon=true
  
  # Parse arguments
  while [[ $# -gt 0 ]]; do
    case $1 in
      install|uninstall|status|start|stop|restart|force-cleanup)
        command="$1"
        ;;
      --daemon)
        enable_daemon=true
        ;;
      --no-daemon)
        enable_daemon=false
        ;;
      --help|-h)
        show_help
        exit 0
        ;;
      *)
        if [ "$1" != "$command" ]; then
          err "Unknown option: $1"
          show_help
          exit 2
        fi
        ;;
    esac
    shift
  done
  
  # Check dependencies
  if ! has_cmd docker; then
    err "Docker is required for MCP container management"
    exit 1
  fi
  
  # Execute command
  case "$command" in
    install)
      check_permissions
      install_system "$enable_daemon"
      ;;
    uninstall)
      check_permissions
      uninstall_system
      ;;
    status)
      show_system_status
      ;;
    start)
      check_permissions
      start_daemon
      ;;
    stop)
      check_permissions
      stop_daemon
      ;;
    restart)
      check_permissions
      restart_daemon
      ;;
    force-cleanup)
      check_permissions
      force_cleanup
      ;;
    *)
      show_help
      exit 2
      ;;
  esac
}

# Execute main function
main "$@"