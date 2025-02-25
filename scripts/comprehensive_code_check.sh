#!/bin/bash

# Comprehensive Python 3.11 Compatibility and Code Quality Check Script

set -e  # Exit immediately if a command exits with a non-zero status

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="/opt/SutazAI"

# Log file
LOG_FILE="$HOME/python311_compatibility_check.log"

# Clear previous log
> "$LOG_FILE"

# Function to log messages
log_message() {
    local level="$1"
    local message="$2"
    local color=""

    case "$level" in
        "INFO") color="$GREEN" ;;
        "WARN") color="$YELLOW" ;;
        "ERROR") color="$RED" ;;
        *) color="$NC" ;;
    esac

    echo -e "${color}[$level] $message${NC}" | tee -a "$LOG_FILE"
}

# Ensure virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    log_message "INFO" "Activating virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Install missing tools
install_tools() {
    log_message "INFO" "Installing/Updating Python development tools..."
    pip install --upgrade pip
    pip install pylint mypy bandit semgrep black pyright 2>&1 | tee -a "$LOG_FILE"
}

# Check Python version
check_python_version() {
    local required_version="3.11"
    local current_version=$(python3 --version | cut -d' ' -f2)

    log_message "INFO" "Checking Python version compatibility..."
    
    if [[ "$(printf '%s\n' "$required_version" "$current_version" | sort -V | head -n1)" != "$required_version" ]]; then
        log_message "ERROR" "Python version must be >= $required_version. Current: $current_version"
        return 1
    fi
    
    log_message "SUCCESS" "Python version is compatible: $current_version"
    return 0
}

# Run comprehensive checks
run_checks() {
    log_message "INFO" "Starting Comprehensive Python 3.11 Compatibility Check"

    # Check Python version first
    check_python_version || return 1

    # Install tools
    install_tools

    # Run Black for code formatting with Python 3.11 target
    log_message "INFO" "Running Black with Python 3.11 target..."
    find "$PROJECT_ROOT" -name "*.py" -print0 | xargs -0 black --target-version py311 2>&1 | tee -a "$LOG_FILE" || true

    # Run Pylint with Python 3.11 configuration
    log_message "INFO" "Running Pylint with Python 3.11 configuration..."
    find "$PROJECT_ROOT" -name "*.py" -print0 | xargs -0 pylint --py-version=3.11 2>&1 | tee -a "$LOG_FILE" || true

    # Run MyPy with Python 3.11 type checking
    log_message "INFO" "Running MyPy with Python 3.11 type checking..."
    mypy --python-version 3.11 "$PROJECT_ROOT" 2>&1 | tee -a "$LOG_FILE" || true

    # Run Pyright for additional type checking
    log_message "INFO" "Running Pyright for Python 3.11 type checking..."
    pyright --pythonVersion 3.11 "$PROJECT_ROOT" 2>&1 | tee -a "$LOG_FILE" || true

    # Run Semgrep with basic configuration
    log_message "INFO" "Running Semgrep with basic configuration..."
    semgrep --config=r/all 2>&1 | tee -a "$LOG_FILE" || true

    bandit -r "$PROJECT_ROOT" -lll -x "**/tests/*" 2>&1 | tee -a "$LOG_FILE" || true

    # Check for deprecated features or potential 3.11 incompatibilities
    log_message "INFO" "Scanning for potential Python 3.11 compatibility issues..."
    find "$PROJECT_ROOT" -name "*.py" -exec grep -Hn "def __future__" {} \; 2>&1 | tee -a "$LOG_FILE" || true

    log_message "INFO" "Python 3.11 Compatibility Check Complete"
}

# Main execution
main() {
    run_checks

    # Check for any critical errors
    if grep -q "ERROR" "$LOG_FILE"; then
        log_message "ERROR" "Critical Python 3.11 compatibility issues found. Please review the log file: $LOG_FILE"
        exit 1
    else
        log_message "SUCCESS" "No critical Python 3.11 compatibility issues found!"
        exit 0
    fi
}

main 