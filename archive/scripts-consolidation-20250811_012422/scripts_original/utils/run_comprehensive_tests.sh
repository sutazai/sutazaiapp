#!/bin/bash

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

"""
Comprehensive Test Runner for Hardware Resource Optimizer
========================================================

Quick start script for running comprehensive testing suites.
This script automates the entire testing process and provides
different testing modes for various scenarios.

Author: QA Team Lead
Version: 1.0.0
"""

set -e  # Exit on any error

# Configuration
AGENT_URL="${AGENT_URL:-http://localhost:8116}"
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="test_execution_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "\n${BLUE}================================================================================================${NC}"
    echo -e "${BLUE} $1 ${NC}"
    echo -e "${BLUE}================================================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "CHECKING PREREQUISITES"
    
    # Check Python 3
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    print_success "Python 3 is available"
    
    # Check required Python packages
    local required_packages=("requests" "psutil" "numpy" "aiohttp")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            print_warning "$package is not installed. Installing..."
            pip3 install "$package" --user
        else
            print_success "$package is available"
        fi
    done
    
    # Check if agent is running
    print_info "Checking agent availability at $AGENT_URL..."
    if curl -s "$AGENT_URL/health" > /dev/null 2>&1; then
        print_success "Agent is responding"
    else
        print_error "Agent is not responding at $AGENT_URL"
        print_info "Please ensure the hardware-resource-optimizer agent is running"
        exit 1
    fi
}

# Install missing dependencies
install_dependencies() {
    print_header "INSTALLING DEPENDENCIES"
    
    # Install Python packages
    pip3 install --user requests psutil numpy aiohttp matplotlib schedule
    
    print_success "Dependencies installed"
}

# Show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS] [TEST_TYPE]"
    echo ""
    echo "TEST_TYPE:"
    echo "  quick      - Quick smoke tests (5-10 minutes)"
    echo "  standard   - E2E and performance tests (20-30 minutes)"
    echo "  comprehensive - All automated tests (45-60 minutes)"
    echo "  manual     - Manual testing procedures (requires user interaction)"
    echo "  continuous - Continuous testing setup"
    echo "  performance - Performance and stress tests only"
    echo "  custom     - Custom test configuration"
    echo ""
    echo "OPTIONS:"
    echo "  --url URL          Agent URL (default: http://localhost:8116)"
    echo "  --config FILE      Custom configuration file"
    echo "  --output FILE      Output report file"
    echo "  --no-cleanup       Skip cleanup after tests"
    echo "  --install-deps     Install dependencies"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 quick                           # Run quick tests"
    echo "  $0 standard --output report.json  # Run standard tests with custom output"
    echo "  $0 comprehensive --url http://localhost:8116"
    echo "  $0 manual                          # Run manual tests"
    echo ""
}

# Run quick tests
run_quick_tests() {
    print_header "RUNNING QUICK TESTS"
    
    python3 "$TEST_DIR/automated_continuous_tests.py" \
        --mode once \
        --test-type smoke \
        --config "$TEST_DIR/test_config.json" \
        2>&1 | tee -a "$LOG_FILE"
}

# Run standard tests
run_standard_tests() {
    print_header "RUNNING STANDARD TESTS"
    
    local output_file="standard_test_report_${TIMESTAMP}.json"
    
    python3 "$TEST_DIR/test_execution_orchestrator.py" \
        --url "$AGENT_URL" \
        --config "$TEST_DIR/test_config.json" \
        --tests e2e performance \
        --output "$output_file" \
        2>&1 | tee -a "$LOG_FILE"
    
    print_success "Standard tests completed. Report: $output_file"
}

# Run comprehensive tests
run_comprehensive_tests() {
    print_header "RUNNING COMPREHENSIVE TESTS"
    
    local output_file="comprehensive_test_report_${TIMESTAMP}.json"
    
    python3 "$TEST_DIR/test_execution_orchestrator.py" \
        --url "$AGENT_URL" \
        --config "$TEST_DIR/test_config.json" \
        --tests e2e performance continuous \
        --output "$output_file" \
        2>&1 | tee -a "$LOG_FILE"
    
    print_success "Comprehensive tests completed. Report: $output_file"
}

# Run manual tests
run_manual_tests() {
    print_header "RUNNING MANUAL TESTS"
    
    print_info "Starting interactive manual testing procedures..."
    print_warning "This requires user interaction throughout the process"
    
    python3 "$TEST_DIR/manual_test_procedures.py" \
        --url "$AGENT_URL" \
        2>&1 | tee -a "$LOG_FILE"
}

# Run performance tests only
run_performance_tests() {
    print_header "RUNNING PERFORMANCE TESTS"
    
    local output_file="performance_test_report_${TIMESTAMP}.json"
    
    python3 "$TEST_DIR/performance_stress_tests.py" \
        --url "$AGENT_URL" \
        --test all \
        --output "$output_file" \
        2>&1 | tee -a "$LOG_FILE"
    
    print_success "Performance tests completed. Report: $output_file"
}

# Setup continuous testing
setup_continuous_testing() {
    print_header "SETTING UP CONTINUOUS TESTING"
    
    print_info "Starting continuous testing orchestrator..."
    
    python3 "$TEST_DIR/automated_continuous_tests.py" \
        --mode continuous \
        --config "$TEST_DIR/test_config.json" \
        2>&1 | tee -a "$LOG_FILE"
}

# Run custom tests
run_custom_tests() {
    print_header "RUNNING CUSTOM TESTS"
    
    if [[ -n "$CUSTOM_CONFIG" ]]; then
        local config_file="$CUSTOM_CONFIG"
    else
        local config_file="$TEST_DIR/test_config.json"
    fi
    
    local output_file="custom_test_report_${TIMESTAMP}.json"
    
    python3 "$TEST_DIR/test_execution_orchestrator.py" \
        --url "$AGENT_URL" \
        --config "$config_file" \
        --output "$output_file" \
        "${CUSTOM_ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"
    
    print_success "Custom tests completed. Report: $output_file"
}

# Generate test summary
generate_summary() {
    print_header "TEST EXECUTION SUMMARY"
    
    if [[ -f "$LOG_FILE" ]]; then
        echo "Log file: $LOG_FILE"
        
        # Count success/failure patterns
        local total_tests=$(grep -c "✅\|❌" "$LOG_FILE" 2>/dev/null || echo "0")
        local passed_tests=$(grep -c "✅" "$LOG_FILE" 2>/dev/null || echo "0")
        local failed_tests=$(grep -c "❌" "$LOG_FILE" 2>/dev/null || echo "0")
        
        echo "Test Results Summary:"
        echo "  Total Tests: $total_tests"
        echo "  Passed: $passed_tests"
        echo "  Failed: $failed_tests"
        
        if [[ $failed_tests -eq 0 ]] && [[ $total_tests -gt 0 ]]; then
            print_success "All tests passed!"
        elif [[ $failed_tests -gt 0 ]]; then
            print_warning "$failed_tests tests failed"
        fi
    fi
    
    # List generated reports
    echo -e "\nGenerated Reports:"
    find . -name "*_report_${TIMESTAMP}.json" -type f 2>/dev/null | while read -r report; do
        echo "  - $(basename "$report")"
    done
    
    echo -e "\nTest artifacts saved in: $TEST_DIR"
}

# Main execution
main() {
    local test_type="standard"
    local custom_config=""
    local custom_output=""
    local no_cleanup=""
    local install_deps=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --url)
                AGENT_URL="$2"
                shift 2
                ;;
            --config)
                custom_config="$2"
                shift 2
                ;;
            --output)
                custom_output="$2"
                shift 2
                ;;
            --no-cleanup)
                no_cleanup="--no-cleanup"
                shift
                ;;
            --install-deps)
                install_deps="true"
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            quick|standard|comprehensive|manual|continuous|performance|custom)
                test_type="$1"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Install dependencies if requested
    if [[ "$install_deps" == "true" ]]; then
        install_dependencies
    fi
    
    # Check prerequisites
    check_prerequisites
    
    # Create logs directory
    mkdir -p logs reports
    
    # Set custom configuration if provided
    if [[ -n "$custom_config" ]]; then
        CUSTOM_CONFIG="$custom_config"
    fi
    
    # Set custom arguments
    CUSTOM_ARGS=()
    if [[ -n "$custom_output" ]]; then
        CUSTOM_ARGS+=(--output "$custom_output")
    fi
    if [[ -n "$no_cleanup" ]]; then
        CUSTOM_ARGS+=("$no_cleanup")
    fi
    
    print_header "HARDWARE RESOURCE OPTIMIZER - COMPREHENSIVE TESTING"
    echo "Test Type: $test_type"
    echo "Agent URL: $AGENT_URL"
    echo "Timestamp: $TIMESTAMP"
    echo "Log File: $LOG_FILE"
    echo ""
    
    # Execute based on test type
    case $test_type in
        quick)
            run_quick_tests
            ;;
        standard)
            run_standard_tests
            ;;
        comprehensive)
            run_comprehensive_tests
            ;;
        manual)
            run_manual_tests
            ;;
        continuous)
            setup_continuous_testing
            ;;
        performance)
            run_performance_tests
            ;;
        custom)
            run_custom_tests
            ;;
        *)
            print_error "Unknown test type: $test_type"
            show_usage
            exit 1
            ;;
    esac
    
    # Generate summary
    generate_summary
    
    print_success "Test execution completed!"
}

# Run main function with all arguments
main "$@"