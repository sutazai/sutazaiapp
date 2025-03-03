#!/bin/bash

# Test Runner Script for Supreme AI Orchestrator
# This script runs the test suite for the orchestrator system.

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/venv"
PYTHON="$VENV_PATH/bin/python3.11"
TEST_DIR="$PROJECT_ROOT/tests"
COVERAGE_DIR="$PROJECT_ROOT/coverage"
REPORT_DIR="$PROJECT_ROOT/test_reports"

# Function to check virtual environment
check_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
        exit 1
    fi
}

# Function to install test dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing test dependencies...${NC}"
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install pytest pytest-asyncio pytest-cov pytest-xdist pytest-html
}

# Function to prepare test environment
prepare_environment() {
    echo -e "${YELLOW}Preparing test environment...${NC}"
    
    # Create directories
    mkdir -p "$COVERAGE_DIR"
    mkdir -p "$REPORT_DIR"
    
    # Clean previous results
    rm -f "$COVERAGE_DIR/*"
    rm -f "$REPORT_DIR/*"
}

# Function to run tests
run_tests() {
    echo -e "${YELLOW}Running tests...${NC}"
    
    # Run pytest with coverage
    "$PYTHON" -m pytest "$TEST_DIR" \
        -v \
        --asyncio-mode=auto \
        --cov=core_system.orchestrator \
        --cov-report=html:"$COVERAGE_DIR" \
        --cov-report=term \
        --html="$REPORT_DIR/report.html" \
        --self-contained-html \
        "$@"
    
    test_exit_code=$?
    
    # Check test results
    if [ $test_exit_code -eq 0 ]; then
        echo -e "${GREEN}All tests passed successfully!${NC}"
    else
        echo -e "${RED}Some tests failed. Check the reports for details.${NC}"
    fi
    
    return $test_exit_code
}

# Function to display coverage report
show_coverage() {
    echo -e "${YELLOW}Coverage Report:${NC}"
    echo -e "HTML coverage report: $COVERAGE_DIR/index.html"
    echo -e "Test report: $REPORT_DIR/report.html"
}

# Main function
main() {
    echo -e "${YELLOW}Starting test suite for Supreme AI Orchestrator...${NC}"
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Check virtual environment
    check_venv
    
    # Install dependencies
    install_dependencies
    
    # Prepare environment
    prepare_environment
    
    # Run tests with any additional arguments
    run_tests "$@"
    test_exit_code=$?
    
    # Show coverage
    show_coverage
    
    # Return test exit code
    return $test_exit_code
}

# Run main function with all script arguments
main "$@" 