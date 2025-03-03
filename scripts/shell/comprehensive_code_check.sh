#!/usr/bin/env bash

# Ensure Python 3.11 is used
PYTHON_CMD="python3.11"

# Limit CPU usage
limit_cpu() {
    local max_cpu_percent="${1:-50}"
    local pid=$$
    
    # Background process to monitor and limit CPU
    (
        while true; do
            cpu_usage=$(ps -p "$pid" -o %cpu | tail -n 1 | tr -d ' ')
            if (( $(echo "$cpu_usage > $max_cpu_percent" | bc -l) )); then
                # Pause the process if CPU usage is too high
                kill -STOP "$pid"
                sleep 5
                kill -CONT "$pid"
            fi
            sleep 2
        done
    ) &
    CPU_LIMIT_PID=$!
    trap 'kill $CPU_LIMIT_PID' EXIT
}

# Call CPU limit at the start of the script
limit_cpu 50  # Limit to 50% CPU

# Verify Python version
verify_python_version() {
    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        echo "ERROR: Python 3.11 not found. Attempting to install..."
        apt-get update && apt-get install -y python3.11 python3.11-dev
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to install Python 3.11"
            exit 1
        fi
    fi

    PYTHON_VERSION=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$PYTHON_VERSION" != "3.11" ]]; then
        echo "ERROR: Python 3.11 required. Found $PYTHON_VERSION. Updating alternatives..."
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
        update-alternatives --set python3 /usr/bin/python3.11
        # Verify again after update
        PYTHON_VERSION=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ "$PYTHON_VERSION" != "3.11" ]]; then
            echo "ERROR: Failed to set Python 3.11 as default"
            exit 1
        fi
    fi
    echo "SUCCESS: Python version verified: $PYTHON_VERSION"
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

    log "INFO" "Running lightweight linting on $directory"

    # Ruff for fast linting with fewer checks
    log "INFO" "Running Ruff with minimal checks..."
    "$PYTHON_CMD" -m ruff check \
        --select=E,F \
        "$directory" 2>&1 | tee -a "$log_file"

    # Minimal MyPy type checking
    log "INFO" "Running minimal MyPy type checking..."
    "$PYTHON_CMD" -m mypy \
        --ignore-missing-imports \
        "$directory" 2>&1 | tee -a "$log_file"
}

# Main script execution
main() {
    # Verify Python version
    verify_python_version

    # Determine log directory
    LOG_DIR="/opt/sutazaiapp/logs"
    mkdir -p "$LOG_DIR"
    
    # Setup log file paths
    LOG_FILE="$LOG_DIR/comprehensive_lint_check.log"
    
    # Clear previous logs
    > "$LOG_FILE"
    
    log "INFO" "Starting comprehensive code quality check..."

    # Directories to check
    DIRECTORIES_TO_CHECK=(
        "core_system"
        "backend"
    )

    # Run comprehensive checks on each directory
    for dir in "${DIRECTORIES_TO_CHECK[@]}"; do
        comprehensive_lint_check "/opt/sutazaiapp/$dir" "$LOG_FILE"
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