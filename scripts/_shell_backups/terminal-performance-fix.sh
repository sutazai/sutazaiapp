#!/bin/bash
#
# SutazAI Terminal Performance Fix
# Specifically addresses terminal freezing during code analysis tools
#

# Define colors for better visibility
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}SutazAI Terminal Performance Fix${NC}"
echo "This script will optimize terminal performance for code analysis tools"

# 1. Limit resource usage for analysis tools
echo -e "\n${YELLOW}Setting resource limits...${NC}"
# Create a wrapper for code analysis tools
mkdir -p /opt/sutazaiapp/scripts/wrappers

# Create Pylint wrapper
cat > /opt/sutazaiapp/scripts/wrappers/pylint-wrapper.sh << 'EOF'
#!/bin/bash
# Wrapper for pylint to prevent terminal freezing
ulimit -t 60  # CPU time limit (seconds)
ulimit -v 2000000  # Virtual memory limit (KB)
exec pylint "$@" 2>&1
EOF
chmod +x /opt/sutazaiapp/scripts/wrappers/pylint-wrapper.sh

# Create Mypy wrapper
cat > /opt/sutazaiapp/scripts/wrappers/mypy-wrapper.sh << 'EOF'
#!/bin/bash
# Wrapper for mypy to prevent terminal freezing
ulimit -t 60  # CPU time limit (seconds)
ulimit -v 2000000  # Virtual memory limit (KB)
exec mypy "$@" 2>&1
EOF
chmod +x /opt/sutazaiapp/scripts/wrappers/mypy-wrapper.sh

# Create Semgrep wrapper
cat > /opt/sutazaiapp/scripts/wrappers/semgrep-wrapper.sh << 'EOF'
#!/bin/bash
# Wrapper for semgrep to prevent terminal freezing
ulimit -t 120  # CPU time limit (seconds)
ulimit -v 2000000  # Virtual memory limit (KB)
exec semgrep "$@" 2>&1
EOF
chmod +x /opt/sutazaiapp/scripts/wrappers/semgrep-wrapper.sh

# 2. Fix TTY buffering issues
echo -e "\n${YELLOW}Fixing TTY buffering...${NC}"
# Create a .bashrc extension for the project
cat > /opt/sutazaiapp/.bashrc.extension << 'EOF'
# SutazAI Terminal Performance Extensions

# Use stdbuf to prevent output buffering issues
alias pylint='stdbuf -oL /opt/sutazaiapp/scripts/wrappers/pylint-wrapper.sh'
alias mypy='stdbuf -oL /opt/sutazaiapp/scripts/wrappers/mypy-wrapper.sh'
alias semgrep='stdbuf -oL /opt/sutazaiapp/scripts/wrappers/semgrep-wrapper.sh'

# Use unbuffer for other potentially problematic commands
if command -v unbuffer >/dev/null 2>&1; then
  alias npm='unbuffer npm'
  alias python='unbuffer python'
else
  echo "Warning: 'unbuffer' not found. Install 'expect' package for better terminal behavior."
fi

# Fix cursor terminal settings
export TERM=xterm-256color
stty sane

# Avoid terminal freezing with large outputs
export PYTHONUNBUFFERED=1
export NODE_NO_READLINE=1

# Ensure proper path
export PATH="/opt/sutazaiapp/scripts/wrappers:$PATH"

# Message
echo "SutazAI Terminal Performance Extensions loaded"
EOF

# 3. Special fix for Cursor terminal
echo -e "\n${YELLOW}Applying Cursor terminal specific fixes...${NC}"
# Install unbuffer if not present
if ! command -v unbuffer >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -qq -y expect
  echo "Installed 'unbuffer' utility from the 'expect' package"
fi

# 4. Create a specialized launcher for code analysis
cat > /opt/sutazaiapp/scripts/code-analysis.sh << 'EOF'
#!/bin/bash
# Safe code analysis runner that prevents terminal freezing

# Source the performance extensions
source /opt/sutazaiapp/.bashrc.extension

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}SutazAI Code Analysis Runner${NC}"
echo "This tool safely runs code analysis without freezing the terminal"

# Function to run a command with timeout and output management
safe_run() {
  local cmd="$1"
  local timeout_sec="$2"
  local name="$3"
  local output_file="/tmp/sutazai_${name}_output.txt"
  
  echo -e "\n${YELLOW}Running ${name}...${NC}"
  timeout "$timeout_sec" bash -c "$cmd" > "$output_file" 2>&1
  
  local status=$?
  if [ $status -eq 124 ]; then
    echo -e "${RED}${name} timed out after ${timeout_sec} seconds${NC}"
    echo "Partial output:"
    tail -30 "$output_file"
  elif [ $status -ne 0 ]; then
    echo -e "${RED}${name} completed with errors (status code: $status)${NC}"
    cat "$output_file"
  else
    echo -e "${GREEN}${name} completed successfully${NC}"
    cat "$output_file"
  fi
  
  echo ""
  return $status
}

# Parse arguments
if [ $# -eq 0 ]; then
  echo "Usage: $0 [all|pylint|mypy|semgrep] [target_directory]"
  echo "Example: $0 all backend/"
  exit 1
fi

ANALYSIS_TYPE="$1"
TARGET_DIR="${2:-.}"  # Default to current directory if not specified

case "$ANALYSIS_TYPE" in
  all)
    safe_run "cd /opt/sutazaiapp && semgrep --config=auto $TARGET_DIR" 120 "semgrep"
    safe_run "cd /opt/sutazaiapp && pylint $TARGET_DIR" 60 "pylint" 
    safe_run "cd /opt/sutazaiapp && mypy $TARGET_DIR" 60 "mypy"
    ;;
  pylint)
    safe_run "cd /opt/sutazaiapp && pylint $TARGET_DIR" 60 "pylint"
    ;;
  mypy)
    safe_run "cd /opt/sutazaiapp && mypy $TARGET_DIR" 60 "mypy"
    ;;
  semgrep)
    safe_run "cd /opt/sutazaiapp && semgrep --config=auto $TARGET_DIR" 120 "semgrep"
    ;;
  *)
    echo "Unknown analysis type: $ANALYSIS_TYPE"
    echo "Supported types: all, pylint, mypy, semgrep"
    exit 1
    ;;
esac

echo -e "${GREEN}Code analysis completed${NC}"
EOF
chmod +x /opt/sutazaiapp/scripts/code-analysis.sh

# 5. Apply the changes to the current terminal session
echo -e "\n${YELLOW}Applying changes to current session...${NC}"
if [ -f "/opt/sutazaiapp/.bashrc.extension" ]; then
  source /opt/sutazaiapp/.bashrc.extension
fi

# Fix current terminal
stty sane
export TERM=xterm-256color
export PYTHONUNBUFFERED=1

echo -e "\n${GREEN}Terminal performance fixes applied!${NC}"
echo "To use the optimized code analysis tools:"
echo "1. Source the performance extensions: source /opt/sutazaiapp/.bashrc.extension"
echo "2. Run analysis with the safe runner: /opt/sutazaiapp/scripts/code-analysis.sh all backend/"
echo "   (Replace 'all' with 'pylint', 'mypy', or 'semgrep' for individual tools)"
echo ""
echo "These changes will prevent terminal freezing during code analysis." 