#!/bin/bash

# format_codebase.sh
# A script to systematically fix PEP 8 violations in a Python codebase
# using existing tools: isort, autoflake, black, and autopep8

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   Python Codebase PEP 8 Formatter      ${NC}"
echo -e "${BLUE}=========================================${NC}"

# Check if backup option is enabled
BACKUP=false
DIRECTORIES=()
EXCLUDE_DIRS=()
FIX_LINE_LENGTH=false
PROBLEM_FILES_LOG="problem_files.log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backup)
      BACKUP=true
      shift
      ;;
    --exclude=*)
      EXCLUDE_DIRS+=("${1#*=}")
      shift
      ;;
    --fix-line-length)
      FIX_LINE_LENGTH=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options] [directories...]"
      echo "Options:"
      echo "  --backup                Create backups before making changes"
      echo "  --exclude=DIR           Exclude directory from processing"
      echo "  --fix-line-length       Attempt to fix line length issues more aggressively"
      echo "  --help                  Show this help message"
      echo ""
      echo "If no directories are specified, the script will process the current directory."
      exit 0
      ;;
    *)
      DIRECTORIES+=("$1")
      shift
      ;;
  esac
done

# If no directories specified, use current directory
if [ ${#DIRECTORIES[@]} -eq 0 ]; then
  DIRECTORIES=(".")
fi

# Exclude patterns
EXCLUDE_PATTERN=""
for dir in "${EXCLUDE_DIRS[@]}"; do
  if [ -z "$EXCLUDE_PATTERN" ]; then
    EXCLUDE_PATTERN="$dir"
  else
    EXCLUDE_PATTERN="$EXCLUDE_PATTERN,$dir"
  fi
done

# Initialize problem files log
echo "Files with syntax errors that couldn't be formatted:" > "$PROBLEM_FILES_LOG"
echo "$(date)" >> "$PROBLEM_FILES_LOG"
echo "----------------------------------------" >> "$PROBLEM_FILES_LOG"

# Create backup if requested
if [ "$BACKUP" = true ]; then
  echo -e "${YELLOW}Creating backups before formatting...${NC}"
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  
  for dir in "${DIRECTORIES[@]}"; do
    BACKUP_DIR="${dir}_backup_${TIMESTAMP}"
    echo -e "  Backing up ${dir} to ${BACKUP_DIR}"
    cp -R "$dir" "$BACKUP_DIR"
  done
  
  echo -e "${GREEN}Backups created successfully.${NC}"
fi

echo -e "${YELLOW}Starting code formatting process...${NC}"

# Define a function to format a directory
format_directory() {
  local dir=$1
  local exclude=$2
  
  echo -e "${BLUE}Processing directory: ${dir}${NC}"
  
  # Step 1: Remove unused imports with autoflake
  echo -e "  ${YELLOW}Removing unused imports with autoflake...${NC}"
  if [ -z "$exclude" ]; then
    python3 -m autoflake --remove-all-unused-imports --recursive --in-place "$dir" 2>/dev/null || echo -e "${RED}Some files could not be processed by autoflake${NC}"
  else
    python3 -m autoflake --remove-all-unused-imports --recursive --in-place --exclude "$exclude" "$dir" 2>/dev/null || echo -e "${RED}Some files could not be processed by autoflake${NC}"
  fi
  
  # Step 2: Sort imports with isort
  echo -e "  ${YELLOW}Sorting imports with isort...${NC}"
  if [ -z "$exclude" ]; then
    python3 -m isort "$dir" 2>/dev/null || echo -e "${RED}Some files could not be processed by isort${NC}"
  else
    python3 -m isort "$dir" --skip "$exclude" 2>/dev/null || echo -e "${RED}Some files could not be processed by isort${NC}"
  fi
  
  # Step 3: Format code with black
  echo -e "  ${YELLOW}Formatting code with black...${NC}"
  if [ -z "$exclude" ]; then
    # Using --skip-string-normalization to prevent Black from converting all strings to double quotes
    python3 -m black --skip-string-normalization "$dir" 2> >(grep "Cannot parse" >> "$PROBLEM_FILES_LOG") || echo -e "${RED}Some files could not be processed by black${NC}"
  else
    python3 -m black --skip-string-normalization "$dir" --exclude "$exclude" 2> >(grep "Cannot parse" >> "$PROBLEM_FILES_LOG") || echo -e "${RED}Some files could not be processed by black${NC}"
  fi
  
  # Step 4: Apply additional PEP 8 fixes with autopep8
  echo -e "  ${YELLOW}Applying additional PEP 8 fixes with autopep8...${NC}"
  
  if [ "$FIX_LINE_LENGTH" = true ]; then
    # More aggressive line wrapping with autopep8
    AUTOPEP8_ARGS="--in-place --recursive --aggressive --aggressive --max-line-length 79"
  else
    AUTOPEP8_ARGS="--in-place --recursive --aggressive --aggressive"
  fi
  
  if [ -z "$exclude" ]; then
    python3 -m autopep8 $AUTOPEP8_ARGS "$dir" 2>/dev/null || echo -e "${RED}Some files could not be processed by autopep8${NC}"
  else
    find "$dir" -name "*.py" -not -path "*/$exclude/*" -exec python3 -m autopep8 $AUTOPEP8_ARGS {} \; 2>/dev/null || echo -e "${RED}Some files could not be processed by autopep8${NC}"
  fi
}

# Process each directory
for dir in "${DIRECTORIES[@]}"; do
  format_directory "$dir" "$EXCLUDE_PATTERN"
done

# Step 5: Run flake8 to check for remaining issues
echo -e "${YELLOW}Checking for remaining issues with flake8...${NC}"
for dir in "${DIRECTORIES[@]}"; do
  echo -e "${BLUE}Checking directory: ${dir}${NC}"
  if [ -z "$EXCLUDE_PATTERN" ]; then
    python3 -m flake8 "$dir" || echo -e "${RED}Some issues remain in ${dir}${NC}"
  else
    python3 -m flake8 "$dir" --exclude "$EXCLUDE_PATTERN" || echo -e "${RED}Some issues remain in ${dir}${NC}"
  fi
done

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}   Formatting process completed!         ${NC}"
echo -e "${GREEN}=========================================${NC}"
echo -e "The following tools were applied:"
echo -e "  1. autoflake - Removed unused imports"
echo -e "  2. isort     - Sorted and organized imports"
echo -e "  3. black     - Formatted code style"
echo -e "  4. autopep8  - Applied additional PEP 8 fixes"
echo -e "  5. flake8    - Checked for remaining issues"
echo -e ""
echo -e "Some manual fixes may still be required. Review the flake8 output above."
echo -e "Files with syntax errors that couldn't be formatted are listed in: ${PROBLEM_FILES_LOG}"
echo -e ""
echo -e "To fix line length issues more aggressively, use: ./format_codebase.sh --fix-line-length [directories]"
echo -e "Use './format_codebase.sh --help' for additional options." 