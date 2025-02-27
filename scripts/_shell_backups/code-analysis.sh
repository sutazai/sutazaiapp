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
