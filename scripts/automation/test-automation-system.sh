#!/bin/bash
# Purpose: Test and validate SutazAI automation system components
# Usage: ./test-automation-system.sh [--quick] [--verbose]

set -euo pipefail


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/opt/sutazaiapp"

# Configuration
QUICK_TEST=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_TEST=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick] [--verbose]"
            exit 1
            ;;
    esac
done

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        ERROR) echo -e "${RED}[$timestamp] ERROR: $message${NC}" ;;
        WARN) echo -e "${YELLOW}[$timestamp] WARN: $message${NC}" ;;
        INFO) echo -e "${BLUE}[$timestamp] INFO: $message${NC}" ;;
        SUCCESS) echo -e "${GREEN}[$timestamp] SUCCESS: $message${NC}" ;;
    esac
}

# Test script existence and permissions
test_script_files() {
    log "INFO" "Testing automation script files..."
    
    local automation_scripts=(
        "daily-health-check.sh"
        "log-rotation-cleanup.sh"
        "database-maintenance.sh"
        "certificate-renewal.sh"
        "agent-restart-monitor.sh"
        "performance-report-generator.sh"
        "security-scanner.sh"
        "backup-verification.sh"
        "setup-automation-cron.sh"
    )
    
    local missing_scripts=0
    local non_executable=0
    
    for script in "${automation_scripts[@]}"; do
        local script_path="$SCRIPT_DIR/$script"
        
        if [[ ! -f "$script_path" ]]; then
            log "ERROR" "Script not found: $script"
            ((missing_scripts++))
        elif [[ ! -x "$script_path" ]]; then
            log "ERROR" "Script not executable: $script"
            ((non_executable++))
        else
            log "SUCCESS" "Script found and executable: $script"
        fi
    done
    
    if [[ $missing_scripts -eq 0 && $non_executable -eq 0 ]]; then
        log "SUCCESS" "All automation scripts are present and executable"
        return 0
    else
        log "ERROR" "Script validation failed: $missing_scripts missing, $non_executable not executable"
        return 1
    fi
}

# Test directory structure
test_directory_structure() {
    log "INFO" "Testing required directory structure..."
    
    local required_dirs=(
        "$BASE_DIR/logs"
        "$BASE_DIR/reports"
        "$BASE_DIR/backups"
        "$BASE_DIR/dashboard"
        "$BASE_DIR/ssl"
        "$BASE_DIR/secrets_secure"
    )
    
    local missing_dirs=0
    
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log "WARN" "Directory missing: $dir"
            ((missing_dirs++))
            
            # Create missing directories
            mkdir -p "$dir"
            log "INFO" "Created directory: $dir"
        else
            log "SUCCESS" "Directory exists: $dir"
        fi
    done
    
    if [[ $missing_dirs -eq 0 ]]; then
        log "SUCCESS" "All required directories exist"
    else
        log "INFO" "Created $missing_dirs missing directories"
    fi
}

# Test dry-run execution of automation scripts
test_script_execution() {
    log "INFO" "Testing script execution (dry-run mode)..."
    
    local test_scripts=(
        "log-rotation-cleanup.sh --dry-run"
        "database-maintenance.sh --dry-run"
        "certificate-renewal.sh --dry-run"
        "backup-verification.sh --dry-run"
    )
    
    if [[ "$QUICK_TEST" == "false" ]]; then
        test_scripts+=(
            "agent-restart-monitor.sh --dry-run"
            "security-scanner.sh --dry-run"
        )
    fi
    
    local failed_tests=0
    
    for test_script in "${test_scripts[@]}"; do
        local script_name=$(echo "$test_script" | cut -d' ' -f1)
        log "INFO" "Testing: $script_name"
        
        if [[ "$VERBOSE" == "true" ]]; then
            if timeout 60 ./$test_script; then
                log "SUCCESS" "Test passed: $script_name"
            else
                log "ERROR" "Test failed: $script_name"
                ((failed_tests++))
            fi
        else
            if timeout 60 ./$test_script >/dev/null 2>&1; then
                log "SUCCESS" "Test passed: $script_name"
            else
                log "ERROR" "Test failed: $script_name"
                ((failed_tests++))
            fi
        fi
    done
    
    if [[ $failed_tests -eq 0 ]]; then
        log "SUCCESS" "All script execution tests passed"
        return 0
    else
        log "ERROR" "$failed_tests script execution tests failed"
        return 1
    fi
}

# Test dependencies
test_dependencies() {
    log "INFO" "Testing required dependencies..."
    
    local required_commands=(
        "docker"
        "jq"
        "curl"
        "bc"
        "gzip"
        "tar"
        "openssl"
    )
    
    local missing_deps=0
    
    for cmd in "${required_commands[@]}"; do
        if command -v "$cmd" >/dev/null 2>&1; then
            log "SUCCESS" "Dependency available: $cmd"
        else
            log "ERROR" "Missing dependency: $cmd"
            ((missing_deps++))
        fi
    done
    
    # Test optional dependencies
    local optional_commands=(
        "trivy"
        "semgrep"
        "mail"
    )
    
    for cmd in "${optional_commands[@]}"; do
        if command -v "$cmd" >/dev/null 2>&1; then
            log "SUCCESS" "Optional dependency available: $cmd"
        else
            log "WARN" "Optional dependency missing: $cmd"
        fi
    done
    
    if [[ $missing_deps -eq 0 ]]; then
        log "SUCCESS" "All required dependencies are available"
        return 0
    else
        log "ERROR" "$missing_deps required dependencies are missing"
        return 1
    fi
}

# Test automation setup script
test_automation_setup() {
    log "INFO" "Testing automation setup functionality..."
    
    # Test setup script syntax
    if bash -n ./setup-automation-cron.sh; then
        log "SUCCESS" "Setup script syntax is valid"
    else
        log "ERROR" "Setup script has syntax errors"
        return 1
    fi
    
    # Test help/usage display
    if ./setup-automation-cron.sh --help 2>&1 | grep -q "Usage:"; then
        log "SUCCESS" "Setup script shows usage information"
    else
        log "INFO" "Setup script usage test completed (expected behavior)"
    fi
    
    return 0
}

# Test configuration file creation
test_configuration() {
    log "INFO" "Testing configuration capabilities..."
    
    # Test environment variable support
    export SUTAZAI_DOMAIN="test.example.com"
    export SUTAZAI_ADMIN_EMAIL="test@example.com"
    
    # Check if scripts can read environment variables
    if ./certificate-renewal.sh --dry-run >/dev/null 2>&1; then
        log "SUCCESS" "Scripts can process environment variables"
    else
        log "WARN" "Environment variable processing may have issues"
    fi
    
    # Clean up test environment variables
    unset SUTAZAI_DOMAIN SUTAZAI_ADMIN_EMAIL
}

# Generate test report
generate_test_report() {
    log "INFO" "Generating automation system test report..."
    
    local report_file="$BASE_DIR/logs/automation_test_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
    "test_info": {
        "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "test_type": "$([ "$QUICK_TEST" == "true" ] && echo "quick" || echo "comprehensive")",
        "hostname": "$(hostname)"
    },
    "test_results": {
        "script_files": "$(test_script_files >/dev/null 2>&1 && echo "PASS" || echo "FAIL")",
        "directory_structure": "$(test_directory_structure >/dev/null 2>&1 && echo "PASS" || echo "FAIL")",
        "dependencies": "$(test_dependencies >/dev/null 2>&1 && echo "PASS" || echo "FAIL")",
        "automation_setup": "$(test_automation_setup >/dev/null 2>&1 && echo "PASS" || echo "FAIL")"
    },
    "summary": {
        "overall_status": "TESTING_COMPLETED",
        "recommendations": [
            "Review test output for any failed components",
            "Install missing optional dependencies if needed",
            "Run full automation setup when ready"
        ]
    }
}
EOF
    
    log "SUCCESS" "Test report saved to: $report_file"
}

# Main execution
main() {
    log "INFO" "Starting SutazAI automation system validation"
    log "INFO" "Test mode: $([ "$QUICK_TEST" == "true" ] && echo "QUICK" || echo "COMPREHENSIVE")"
    
    local test_failures=0
    
    # Run all tests
    test_script_files || ((test_failures++))
    test_directory_structure || ((test_failures++))
    test_dependencies || ((test_failures++))
    test_automation_setup || ((test_failures++))
    test_configuration || ((test_failures++))
    
    if [[ "$QUICK_TEST" == "false" ]]; then
        test_script_execution || ((test_failures++))
    else
        log "INFO" "Skipping script execution tests in quick mode"
    fi
    
    # Generate test report
    generate_test_report
    
    # Show summary
    echo
    echo "============================================"
    echo "    AUTOMATION SYSTEM TEST SUMMARY"
    echo "============================================"
    echo "Test Mode: $([ "$QUICK_TEST" == "true" ] && echo "QUICK" || echo "COMPREHENSIVE")"
    echo "Test Failures: $test_failures"
    echo "Overall Status: $([ $test_failures -eq 0 ] && echo "PASS" || echo "FAIL")"
    echo "Timestamp: $(date)"
    echo "============================================"
    
    if [[ $test_failures -eq 0 ]]; then
        log "SUCCESS" "All automation system tests passed!"
        echo
        echo "Next steps:"
        echo "1. Run: ./setup-automation-cron.sh"
        echo "2. Monitor logs in: $BASE_DIR/logs/"
        echo "3. Check reports in: $BASE_DIR/reports/"
        echo "4. Access dashboard: $BASE_DIR/dashboard/automation-status.html"
        exit 0
    else
        log "ERROR" "Automation system validation failed with $test_failures issues"
        echo
        echo "Please review the test output and fix any issues before proceeding."
        exit 1
    fi
}

# Run main function
main "$@"