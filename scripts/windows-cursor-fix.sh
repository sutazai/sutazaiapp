#!/bin/bash
#
# Windows Cursor Terminal Fix
# This script fixes terminal issues specific to Windows users

echo "Applying Windows-specific Cursor terminal fixes..."

# Fix terminal size issues
stty rows 24 columns 80

# Fix output buffering - critical for Windows SSH connections
export PYTHONUNBUFFERED=1
export TERM=xterm-256color

# Set lower limits for Windows SSH connections
ulimit -t 30  # CPU time limit
ulimit -v 800000  # Virtual memory limit

# Fix tool blocking issues
for cmd in semgrep pylint mypy; do
  cat > "/usr/local/bin/$cmd-safe" << EOF
#!/bin/bash
# Safe wrapper for $cmd that prevents terminal freezing
# Run with lower priority and timeout
timeout 60s nice -n 19 $cmd "\$@" 2>&1
EOF
  chmod +x "/usr/local/bin/$cmd-safe"
done

# Create aliases for safe tool usage
cat > ~/.cursor-aliases << EOF
# Safe tool aliases for Windows Cursor users
alias semgrep="semgrep-safe"
alias pylint="pylint-safe"
alias mypy="mypy-safe"

# Quick terminal reset function
cursor-reset() {
  stty sane
  stty rows 24 columns 80
  export TERM=xterm-256color
  export PYTHONUNBUFFERED=1
  echo "Terminal reset complete."
}
EOF

# Add aliases to bashrc if not already there
grep -q "source ~/.cursor-aliases" ~/.bashrc || echo "source ~/.cursor-aliases" >> ~/.bashrc

# Apply settings immediately
source ~/.cursor-aliases

echo "Windows Cursor terminal fix applied!"
echo "When having issues, type 'cursor-reset' to fix the terminal."
echo "Use the -safe versions of tools: semgrep-safe, pylint-safe, mypy-safe" 