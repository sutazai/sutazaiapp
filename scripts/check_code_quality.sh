#!/bin/bash
# Script to run comprehensive code quality checks

set -e  # Exit on error

# Print with colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting code quality checks...${NC}"

# Check if we're in the correct directory (repository root)
if [ ! -f "main.py" ]; then
    echo -e "${RED}Error: This script must be run from the repository root directory.${NC}"
    exit 1
fi

# Create a directory for reports
REPORT_DIR="reports/code_quality"
mkdir -p "$REPORT_DIR"

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install dependencies if they don't exist
echo -e "${GREEN}Checking dependencies...${NC}"
DEPS_TO_INSTALL=()

if ! command_exists ruff; then
    DEPS_TO_INSTALL+=("ruff")
fi

if ! command_exists mypy; then
    DEPS_TO_INSTALL+=("mypy")
fi

if ! command_exists bandit; then
    DEPS_TO_INSTALL+=("bandit")
fi

if ! command_exists radon; then
    DEPS_TO_INSTALL+=("radon")
fi

if [ ${#DEPS_TO_INSTALL[@]} -gt 0 ]; then
    echo -e "${YELLOW}Installing missing dependencies: ${DEPS_TO_INSTALL[*]}${NC}"
    pip install "${DEPS_TO_INSTALL[@]}"
fi

# Run Ruff for linting
echo -e "${GREEN}Running Ruff linter...${NC}"
ruff check . --output-file "$REPORT_DIR/ruff_report.txt" || {
    echo -e "${RED}Ruff found issues. See $REPORT_DIR/ruff_report.txt for details.${NC}"
    FAILED=1
}

# Run Mypy for type checking
echo -e "${GREEN}Running Mypy type checker...${NC}"
mypy --ignore-missing-imports --exclude venv --exclude node_modules --python-version 3.11 . > "$REPORT_DIR/mypy_report.txt" || {
    echo -e "${RED}Mypy found issues. See $REPORT_DIR/mypy_report.txt for details.${NC}"
    FAILED=1
}

# Run Bandit for security checks
echo -e "${GREEN}Running Bandit security checker...${NC}"
bandit -r . -x venv,node_modules -f txt -o "$REPORT_DIR/bandit_report.txt" || {
    echo -e "${RED}Bandit found security issues. See $REPORT_DIR/bandit_report.txt for details.${NC}"
    FAILED=1
}

# Run Radon for code complexity
echo -e "${GREEN}Running Radon complexity analyzer...${NC}"
radon cc . -a -s -nb > "$REPORT_DIR/radon_complexity.txt"
radon mi . -s > "$REPORT_DIR/radon_maintainability.txt"

# Print summary of findings
echo -e "${GREEN}Generating summary report...${NC}"
{
    echo "# Code Quality Report $(date)"
    echo
    echo "## File counts"
    echo "- Python files: $(find . -name "*.py" -not -path "*/venv/*" -not -path "*/node_modules/*" | wc -l)"
    echo "- Total lines of Python code: $(find . -name "*.py" -not -path "*/venv/*" -not -path "*/node_modules/*" -exec cat {} \; | wc -l)"
    echo
    
    echo "## Linting Issues"
    RUFF_COUNT=$(grep -c "" "$REPORT_DIR/ruff_report.txt" 2>/dev/null || echo "0")
    echo "- Ruff found $RUFF_COUNT issues"
    
    echo
    echo "## Type Checking Issues"
    MYPY_COUNT=$(grep -c "error:" "$REPORT_DIR/mypy_report.txt" 2>/dev/null || echo "0")
    echo "- Mypy found $MYPY_COUNT type issues"
    
    echo
    echo "## Security Issues"
    BANDIT_COUNT=$(grep -c "Issue:" "$REPORT_DIR/bandit_report.txt" 2>/dev/null || echo "0")
    echo "- Bandit found $BANDIT_COUNT security issues"
    
    echo
    echo "## Code Complexity"
    echo "- Check $REPORT_DIR/radon_complexity.txt for detailed complexity analysis"
    echo "- Check $REPORT_DIR/radon_maintainability.txt for maintainability index"
    
    echo
    echo "## Recommendations"
    echo "1. Fix all linting issues identified by Ruff"
    echo "2. Address type checking errors from Mypy"
    echo "3. Resolve security issues found by Bandit"
    echo "4. Refactor complex code identified by Radon"
} > "$REPORT_DIR/summary.md"

echo -e "${GREEN}Code quality check complete!${NC}"
echo -e "${GREEN}Summary report saved to $REPORT_DIR/summary.md${NC}"

if [ "$FAILED" == "1" ]; then
    echo -e "${RED}Some checks failed. Please fix the issues and run the script again.${NC}"
    exit 1
fi

echo -e "${GREEN}All checks passed successfully!${NC}"
exit 0 