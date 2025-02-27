#!/bin/bash
#
# Emergency Fix for Cursor Chat and Terminal Freezing
# This script applies minimal changes to restore functionality

echo "Applying emergency fix for Cursor freezing..."

# 1. Kill any potentially stuck Cursor processes
echo "Terminating stuck processes..."
pkill -f "semgrep|pylint|mypy" 2>/dev/null || true

# 2. Reset all terminal devices
echo "Resetting terminal devices..."
for tty in /dev/pts/*; do
  chmod a+rw "$tty" 2>/dev/null || true
done

# 3. Apply critical environment fixes
cat > ~/.cursor-fix.sh << 'EOF'
# Critical Cursor fixes
export TERM=xterm-256color
export PYTHONUNBUFFERED=1
stty sane

# Fix output buffering issues
export PYTHONSTARTUP=$(cat <<PYFILE
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)
PYFILE
)

# Limit resource usage
ulimit -t 30  # CPU time in seconds
ulimit -v 1000000  # Memory size in KB
EOF

chmod +x ~/.cursor-fix.sh
echo "source ~/.cursor-fix.sh" >> ~/.bashrc

# 4. Apply immediate fixes to current session
source ~/.cursor-fix.sh

# 5. Create quick restart script for cursor chat
cat > ~/restart-cursor-chat.sh << 'EOF'
#!/bin/bash
# Quick restart for frozen cursor chat
# Run this when chat or terminal freezes

# Reset terminal
stty sane
export TERM=xterm-256color

# Kill any stuck processes
pkill -f "semgrep|pylint|mypy" 2>/dev/null || true

# Fix terminal permissions
for tty in /dev/pts/*; do
  chmod a+rw "$tty" 2>/dev/null || true
done

# Apply cursor fix
source ~/.cursor-fix.sh

# Print success
echo "Cursor chat reset complete. Try the terminal again."
EOF

chmod +x ~/restart-cursor-chat.sh

echo "Emergency fix applied!"
echo "If you experience freezing again, run: ~/restart-cursor-chat.sh" 