#!/bin/bash
#
# SutazAI Cursor Terminal Fix Script
# This script fixes execution problems with Cursor's terminal in the SutazAI environment
#

# Strict error handling
set -euo pipefail

# Define paths
SUTAZAI_BASE="/opt/sutazaiapp"
CURSOR_SERVER_PATH="/root/.cursor-server"
LOG_FILE="/opt/sutazaiapp/logs/cursor-terminal-fix.log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Log function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Starting SutazAI Cursor Terminal Fix"

# 1. Fix terminal device permissions
log "Fixing terminal device permissions"
chmod a+rw /dev/pts/* 2>/dev/null || log "Warning: Could not change /dev/pts permissions"

# 2. Fix sutazaiapp directory permissions
log "Fixing SutazAI app directory permissions"
find "$SUTAZAI_BASE/scripts" -type f -name "*.sh" -exec chmod +x {} \; 2>/dev/null
find "$SUTAZAI_BASE/scripts" -type f -name "*.py" -exec chmod +x {} \; 2>/dev/null

# 3. Fix virtual environment permissions
if [ -d "$SUTAZAI_BASE/venv" ]; then
  log "Fixing virtual environment permissions"
  chmod -R +x "$SUTAZAI_BASE/venv/bin" 2>/dev/null || log "Warning: Could not fix venv permissions"
fi

# 4. Fix Cursor shell integration script permissions
if [ -d "$CURSOR_SERVER_PATH" ]; then
  log "Fixing Cursor shell integration scripts"
  find "$CURSOR_SERVER_PATH" -name "shellIntegration-bash.sh" -exec chmod +x {} \; 2>/dev/null
fi

# 5. Create wrapper script for terminal execution
TERMINAL_WRAPPER="$SUTAZAI_BASE/scripts/terminal-wrapper.sh"
log "Creating terminal wrapper script at $TERMINAL_WRAPPER"

cat > "$TERMINAL_WRAPPER" << 'EOF'
#!/bin/bash

# SutazAI Terminal Wrapper
# This script ensures proper execution environment for the terminal

# Set environment variables
export TERM=xterm-256color
export PYTHONPATH=$PYTHONPATH:/opt/sutazaiapp
export SUTAZAI_HOME=/opt/sutazaiapp
export PATH=/opt/sutazaiapp/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH

# Fix tty permissions if needed
if [ -t 0 ]; then
  TTY=$(tty)
  if [ -n "$TTY" ]; then
    chmod +rw "$TTY" 2>/dev/null || true
  fi
fi

# Reset terminal settings
stty sane

# Activate virtual environment if available
if [ -f "/opt/sutazaiapp/venv/bin/activate" ]; then
  source /opt/sutazaiapp/venv/bin/activate
fi

# Start a new shell
if [ -n "$1" ]; then
  exec "$@"
else
  exec $SHELL -l
fi
EOF

# Make wrapper executable
chmod 755 "$TERMINAL_WRAPPER"

# 6. Fix file descriptor limits
log "Fixing file descriptor limits"
ulimit -n 4096 2>/dev/null || log "Warning: Could not increase file descriptor limit"

# 7. Create systemd user environment file for terminal
SYSTEMD_DIR="/etc/systemd/system"
USER_ENV_DIR="/etc/systemd/user.conf.d"

if [ ! -d "$USER_ENV_DIR" ]; then
  log "Creating systemd user environment directory"
  mkdir -p "$USER_ENV_DIR"
fi

log "Creating systemd user environment file"
cat > "$USER_ENV_DIR/sutazai-terminal.conf" << 'EOF'
[Manager]
DefaultEnvironment="SUTAZAI_HOME=/opt/sutazaiapp" "PYTHONPATH=/opt/sutazaiapp" "TERM=xterm-256color"
EOF

# 8. Apply execution permissions to key directories
log "Setting execution permissions for key directories"
find "$SUTAZAI_BASE" -type d -exec chmod +x {} \; 2>/dev/null || log "Warning: Could not set directory execution permissions"

log "Terminal execution fix completed successfully"
log "You can now use the terminal wrapper with: $TERMINAL_WRAPPER"
log ""
log "If Cursor chat terminal still freezes, try launching it with:"
log "  1. /opt/sutazaiapp/scripts/terminal-wrapper.sh"
log "  2. Or source this fix script: source /opt/sutazaiapp/scripts/cursor-terminal-fix.sh"
log ""

# Finally, apply all the fixes to the current shell
export TERM=xterm-256color
export PYTHONPATH=$PYTHONPATH:/opt/sutazaiapp
export SUTAZAI_HOME=/opt/sutazaiapp
export PATH=/opt/sutazaiapp/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
stty sane

# Success message
echo ""
echo "=========================================================="
echo "  SutazAI Cursor Terminal Fix Applied Successfully"
echo "  Your terminal should now work properly with Cursor"
echo "=========================================================="
echo "" 