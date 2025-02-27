#!/bin/bash

# Comprehensive Python 3.11 Compatibility and Code Quality Check Script

set -e  # Exit immediately if a command exits with a non-zero status

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="/opt/sutazaiapp"

# Log file
LOG_FILE="$PROJECT_ROOT/logs/compatibility_check.log"
SUMMARY_FILE="$PROJECT_ROOT/logs/compatibility_summary.log"

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Clear previous logs
> "$LOG_FILE"
> "$SUMMARY_FILE"

# Function to log messages
log_message() {
    local level="$1"
    local message="$2"
    local color=""

    case "$level" in
        "INFO") color="$GREEN" ;;
        "WARN") color="$YELLOW" ;;
        "ERROR") color="$RED" ;;
        "SUCCESS") color="$GREEN" ;;
        *) color="$NC" ;;
    esac

    echo -e "${color}[$level] $message${NC}" | tee -a "$LOG_FILE"
    
    # Add errors and warnings to summary file
    if [[ "$level" == "ERROR" || "$level" == "WARN" || "$level" == "FAILED" ]]; then
        echo "[$level] $message" >> "$SUMMARY_FILE"
    fi
}

# Find application Python files (excluding venv, .git, etc.)
find_app_files() {
    local exclude_dirs=(
        "venv"
        ".git"
        "__pycache__"
        "node_modules"
        "tests/fixtures"
    )
    
    local exclude_args=()
    for dir in "${exclude_dirs[@]}"; do
        exclude_args+=("-not" "-path" "*/$dir/*")
    done
    
    # Find all Python files in the project, excluding the specified directories
    find "$PROJECT_ROOT" -name "*.py" "${exclude_args[@]}" -type f > "$PROJECT_ROOT/logs/app_files.txt"
    
    log_message "INFO" "Found $(wc -l < "$PROJECT_ROOT/logs/app_files.txt") Python files to check."
}

# Ensure virtual environment is activated
ensure_venv() {
    if [ -z "$VIRTUAL_ENV" ] || [[ "$VIRTUAL_ENV" != *"venv"* ]]; then
        log_message "INFO" "Activating virtual environment..."
        source "$PROJECT_ROOT/venv/bin/activate"
    fi
}

# Install missing tools
install_tools() {
    log_message "INFO" "Installing/Updating Python development tools..."
    pip install --quiet pylint mypy bandit semgrep black pyright 2>&1 | tee -a "$LOG_FILE"
}

# Check Python version
check_python_version() {
    local required_version="3.11"
    
    # Check current Python version in the environment
    local current_version=$(python --version | cut -d' ' -f2)

    log_message "INFO" "Checking Python version compatibility..."
    
    if [[ "$(printf '%s\n' "$required_version" "$current_version" | sort -V | head -n1)" != "$required_version" ]]; then
        log_message "ERROR" "Python version must be >= $required_version. Current: $current_version"
        return 1
    fi
    
    log_message "SUCCESS" "Python version is compatible: $current_version"
    return 0
}

# Create a summary of errors and warnings
create_summary() {
    local error_count=$(grep -c -i "ERROR" "$LOG_FILE" || echo "0")
    local warning_count=$(grep -c -i "WARN" "$LOG_FILE" || echo "0")
    
    echo "=============== COMPATIBILITY CHECK SUMMARY ===============" >> "$SUMMARY_FILE"
    echo "Errors: $error_count" >> "$SUMMARY_FILE"
    echo "Warnings: $warning_count" >> "$SUMMARY_FILE"
    echo "=========================================================" >> "$SUMMARY_FILE"
    
    if [ "$error_count" -gt 0 ] || [ "$warning_count" -gt 0 ]; then
        echo "" >> "$SUMMARY_FILE"
        echo "Top errors (up to 20):" >> "$SUMMARY_FILE"
        grep -i "ERROR" "$LOG_FILE" | head -n 20 >> "$SUMMARY_FILE"
        
        echo "" >> "$SUMMARY_FILE"
        echo "Top warnings (up to 20):" >> "$SUMMARY_FILE"
        grep -i "WARN" "$LOG_FILE" | head -n 20 >> "$SUMMARY_FILE"
    fi
    
    log_message "INFO" "Summary created at $SUMMARY_FILE"
}

# Run comprehensive checks
run_checks() {
    log_message "INFO" "Starting Comprehensive Python 3.11 Compatibility Check"

    # Check Python version first
    check_python_version || return 1

    # Find application files
    log_message "INFO" "Finding application Python files..."
    find_app_files

    # Install tools
    install_tools

    if [ ! -s "$PROJECT_ROOT/logs/app_files.txt" ]; then
        log_message "ERROR" "No Python files found to check. Please check the directory structure."
        return 1
    fi

    # Define paths to the installed tools in the virtual environment
    local VENV_BIN="$VIRTUAL_ENV/bin"
    
    # Run Black for code formatting with Python 3.11 target
    log_message "INFO" "Running Black with Python 3.11 target..."
    cat "$PROJECT_ROOT/logs/app_files.txt" | xargs "$VENV_BIN/black" --target-version py311 2>&1 | tee -a "$LOG_FILE" || log_message "WARN" "Black formatting had issues"

    # Run Pylint with Python 3.11 configuration
    log_message "INFO" "Running Pylint with Python 3.11 configuration..."
    cat "$PROJECT_ROOT/logs/app_files.txt" | xargs "$VENV_BIN/pylint" --py-version=3.11 2>&1 | tee -a "$LOG_FILE" || log_message "WARN" "Pylint found issues"

    # Run MyPy with Python 3.11 type checking - specifically on backend and ai_agents
    log_message "INFO" "Running MyPy with Python 3.11 type checking..."
    "$VENV_BIN/mypy" --python-version 3.11 --explicit-package-bases "$PROJECT_ROOT/backend" "$PROJECT_ROOT/ai_agents" 2>&1 | tee -a "$LOG_FILE" || log_message "WARN" "MyPy found type issues"

    # Run Pyright for additional type checking
    log_message "INFO" "Running Pyright for Python 3.11 type checking..."
    "$VENV_BIN/pyright" "$PROJECT_ROOT/backend" "$PROJECT_ROOT/ai_agents" 2>&1 | tee -a "$LOG_FILE" || log_message "WARN" "Pyright found type issues"

    # Run Bandit for security issues
    log_message "INFO" "Running Bandit for security issues..."
    "$VENV_BIN/bandit" -r "$PROJECT_ROOT/backend" "$PROJECT_ROOT/ai_agents" -lll -x "*/tests/*" 2>&1 | tee -a "$LOG_FILE" || log_message "WARN" "Bandit found security issues"

    # Check for deprecated features or potential 3.11 incompatibilities
    log_message "INFO" "Scanning for potential Python 3.11 compatibility issues..."
    
    # Specific patterns that may cause issues in Python 3.11
    local patterns=(
        "collections.Callable"  # Use collections.abc instead
        "from collections import Mapping" # Use from collections.abc instead
        "asyncio.coroutine" # Removed in Python 3.11
        "loop.create_future" # Changed in Python 3.11
        "time.clock" # Removed
        "cgi.parse" # Legacy modules deprecated
        "bind_async" # Changed behavior
        "locale.getdefaultlocale" # Changed behavior
        "importlib.abc.PathEntryFinder" # Changed
        "xml.etree.ElementTree.VERSION" # Changed
    )
    
    for pattern in "${patterns[@]}"; do
        log_message "INFO" "Checking for pattern: $pattern"
        while read -r file; do
            grep -n "$pattern" "$file" >> "$LOG_FILE" 2>/dev/null || true
        done < "$PROJECT_ROOT/logs/app_files.txt"
    done

    log_message "INFO" "Python 3.11 Compatibility Check Complete"
}

# Main execution
main() {
    # Ensure virtual environment is activated
    ensure_venv
    
    # Run all checks
    run_checks
    
    # Create summary
    create_summary
    
    # Check for any critical errors
    local error_count=$(grep -c -i "ERROR" "$SUMMARY_FILE" || echo "0")
    
    if [ "$error_count" -gt 0 ]; then
        log_message "ERROR" "Critical Python 3.11 compatibility issues found. Please review: $SUMMARY_FILE"
        echo -e "${RED}Found $error_count errors. See $SUMMARY_FILE for details.${NC}"
        exit 1
    else
        log_message "SUCCESS" "No critical Python 3.11 compatibility issues found!"
        echo -e "${GREEN}No critical compatibility issues found!${NC}"
        exit 0
    fi
}

main 