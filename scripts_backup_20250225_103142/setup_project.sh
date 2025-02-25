#!/bin/bash

# SutazAI Project Setup Script

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="/opt/SutazAI"

# Log file
LOG_FILE="$HOME/sutazai_project_setup.log"

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

# Create virtual environment
create_venv() {
    log_message "INFO" "Creating virtual environment..."
    python3.11 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
}

# Install Poetry
install_poetry() {
    log_message "INFO" "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3.11 -
    poetry --version
}

# Install project dependencies
install_dependencies() {
    log_message "INFO" "Installing project dependencies..."
    poetry install --no-interaction
}

# Install pre-commit hooks
setup_pre_commit() {
    log_message "INFO" "Setting up pre-commit hooks..."
    pre-commit install
    pre-commit autoupdate
}

# Run initial checks
run_initial_checks() {
    log_message "INFO" "Running initial code quality checks..."
    poetry run black .
    poetry run pylint **/*.py
    poetry run mypy .
}

# Main setup function
main() {
    cd "$PROJECT_ROOT"

    log_message "INFO" "Starting SutazAI Project Setup"

    # Perform checks and setup steps
    check_python_version
    create_venv
    install_poetry
    install_dependencies
    setup_pre_commit
    run_initial_checks

    log_message "SUCCESS" "SutazAI Project Setup Completed Successfully!"
}

# Execute main function
main 