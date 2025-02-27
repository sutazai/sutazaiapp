#!/bin/bash
#
# Simple Fix for Cursor Terminal Issues
# Run this script when experiencing terminal freezing

echo "Applying quick fix for Cursor terminal..."

# Kill any hanging processes
pkill -f "semgrep|pylint|mypy" 2>/dev/null || true
echo "✓ Terminated any stuck processes"

# Fix terminal settings
stty sane
stty rows 40 columns 120
echo "✓ Reset terminal settings"

# Set environment variables
export TERM=xterm-256color
export PYTHONUNBUFFERED=1
echo "✓ Set proper environment variables"

# Create wrapper scripts for analysis tools
mkdir -p /tmp/cursor-wrappers

for cmd in semgrep pylint mypy; do
  cat > "/tmp/cursor-wrappers/$cmd" << EOF
#!/bin/bash
# Safe wrapper for $cmd
echo "Running $cmd safely..."
timeout 60s $cmd "\$@" 2>&1
echo "Completed $cmd"
EOF
  chmod +x "/tmp/cursor-wrappers/$cmd"
done

echo "✓ Created safer tool wrappers in /tmp/cursor-wrappers"

# Add wrappers to PATH
export PATH="/tmp/cursor-wrappers:$PATH"
echo "export PATH=\"/tmp/cursor-wrappers:\$PATH\"" >> ~/.bashrc

echo "Terminal fix applied! Run code analysis with the commands in /tmp/cursor-wrappers"
echo "If terminal freezes again, simply run this script again." 