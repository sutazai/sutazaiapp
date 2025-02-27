#!/usr/bin/env bash

# Ensure Python 3.11 is used
PYTHON_CMD="python3.11"

# Verify Python version
verify_python_version() {
    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        echo "❌ Error: Python 3.11 is not installed."
        exit 1
    fi

    PYTHON_VERSION=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$PYTHON_VERSION" != "3.11" ]]; then
        echo "❌ Error: Python 3.11 is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
}

# Verify Python version before proceeding
verify_python_version

# Comprehensive Python 3.11 Compatibility and Code Quality Check Script

set -e  # Exit immediately if a command exits with a non-zero status

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="/opt/sutazaiapp"

# Logging function with enhanced verbosity
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case "$level" in
        "INFO")
            echo -e "${GREEN}[$timestamp][INFO]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "WARN")
            echo -e "${YELLOW}[$timestamp][WARN]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "ERROR")
            echo -e "${RED}[$timestamp][ERROR]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "DEBUG")
            echo -e "${YELLOW}[$timestamp][DEBUG]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[$timestamp][SUCCESS]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        *)
            echo -e "[$timestamp][UNKNOWN] $message" | tee -a "$LOG_FILE"
            ;;
    esac
}

# Comprehensive linting and error checking function
comprehensive_lint_check() {
    local directory="$1"
    local log_file="$2"

    log "INFO" "Running comprehensive linting on $directory"

    # Pylint with detailed configuration
    log "INFO" "Running Pylint with detailed analysis..."
    "$PYTHON_CMD" -m pylint \
        --rcfile=.pylintrc \
        --output-format=text \
        --reports=yes \
        --score=yes \
        "$directory" 2>&1 | tee -a "$log_file"

    # Ruff for fast linting and potential fixes
    log "INFO" "Running Ruff for fast linting and potential fixes..."
    "$PYTHON_CMD" -m ruff check \
        --fix \
        --show-fixes \
        --output-format=json \
        --ignore-noqa \
        --select=ALL \
        --ignore=D,E501,F401 \
        "$directory" 2>&1 | tee -a "$log_file"

    # MyPy for strict type checking
    log "INFO" "Running MyPy with strict type checking..."
    "$PYTHON_CMD" -m mypy \
        --strict \
        --show-error-codes \
        --pretty \
        --no-warn-unused-configs \
        --ignore-missing-imports \
        --disallow-untyped-defs \
        --disallow-incomplete-defs \
        "$directory" 2>&1 | tee -a "$log_file"

    # Install missing type stubs
    log "INFO" "Installing missing type stubs..."
    "$PYTHON_CMD" -m pip install \
        types-networkx \
        types-psutil \
        types-PyYAML \
        types-requests \
        types-setuptools 2>&1 | tee -a "$log_file"

    "$PYTHON_CMD" -m mypy --install-types --non-interactive 2>&1 | tee -a "$log_file"

    # Bandit for security vulnerability scanning
    log "INFO" "Running Bandit for security vulnerability scanning..."
    "$PYTHON_CMD" -m bandit \
        -r "$directory" \
        -f custom \
        -o "$log_file.bandit" 2>&1 | tee -a "$log_file"

    # Black for code formatting
    log "INFO" "Running Black for code formatting..."
    "$PYTHON_CMD" -m black \
        --check \
        --diff \
        "$directory" 2>&1 | tee -a "$log_file"
}

# Main script execution
main() {
    # Determine log directory
    LOG_DIR=$(determine_log_directory)
    
    # Setup log file paths
    LOG_FILE="$LOG_DIR/comprehensive_lint_check.log"
    
    # Clear previous logs
    > "$LOG_FILE"
    
    log "INFO" "Starting comprehensive code quality check..."

    # Directories to check
    DIRECTORIES_TO_CHECK=(
        "core_system"
        "misc"
        "system_integration"
        "scripts"
    )

    # Run comprehensive checks on each directory
    for dir in "${DIRECTORIES_TO_CHECK[@]}"; do
        comprehensive_lint_check "$PROJECT_ROOT/$dir" "$LOG_FILE"
    done

    # Final summary
    log "SUCCESS" "🎉 Comprehensive code quality check completed!"
    
    # Generate detailed report
    generate_detailed_report "$LOG_FILE"
}

# Generate a detailed report
generate_detailed_report() {
    local log_file="$1"
    local report_file="$LOG_DIR/code_quality_report.md"

    echo "# Code Quality Report" > "$report_file"
    echo "## Timestamp: $(date)" >> "$report_file"
    echo "" >> "$report_file"

    # Extract key metrics and issues
    grep -E "Your code has been rated|error:|warning:" "$log_file" >> "$report_file"

    log "INFO" "Detailed report generated at $report_file"
}

# Determine log directory function (from previous script)
determine_log_directory() {
    local potential_dirs=(
        "/opt/sutazaiapp/logs"
        "/var/log/sutazaiapp"
        "/tmp/sutazaiapp_logs"
    )
    
    for dir in "${potential_dirs[@]}"; do
        mkdir -p "$dir" 2>/dev/null
        
        if [ -d "$dir" ] && [ -w "$dir" ]; then
            echo "$dir"
            return 0
        fi
    done
    
    local temp_log_dir="/tmp/sutazaiapp_logs_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$temp_log_dir"
    echo "$temp_log_dir"
}

# Execute main function
main 