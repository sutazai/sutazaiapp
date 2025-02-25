#!/bin/bash
set -euo pipefail

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging function
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")    echo -e "${GREEN}[INFO   $timestamp]${NC} $message" ;;
        "WARNING") echo -e "${YELLOW}[WARNING $timestamp]${NC} $message" ;;
        "ERROR")   echo -e "${RED}[ERROR  $timestamp]${NC} $message" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS $timestamp]${NC} $message" ;;
        *)         echo -e "[${timestamp}] $message" ;;
    esac
}

# Comprehensive testing workflow
run_tests() {
    log "INFO" "Starting comprehensive test suite"

    # Create virtual environment for testing
    python3 -m venv test_env
    source test_env/bin/activate

    # Install test dependencies
    pip install -r requirements-test.txt

    # Run different types of tests
    log "INFO" "Running unit tests"
    pytest tests/ --cov=. --cov-report=xml --cov-report=term-missing

    bandit -r . -f custom
    safety check

    log "INFO" "Running performance tests"
    locust -f tests/performance_test.py --headless -u 100 -r 10 -t 1m

    # Deactivate virtual environment
    deactivate

    log "SUCCESS" "Test suite completed successfully"
}

# Performance profiling
profile_application() {
    log "INFO" "Starting application performance profiling"

    # Use py-spy for continuous profiling
    py-spy record -o profile.svg --pid $(pgrep -f "python.*main.py") -d 60

    log "INFO" "Performance profile saved to profile.svg"
}

# Main testing workflow
main() {
    log "INFO" "Starting SutazAI Local Testing"

    # Ensure deployment is up
    ./local_deploy.sh || {
        log "ERROR" "Deployment failed before testing"
        exit 1
    }

    # Run comprehensive tests
    run_tests

    # Optional: Performance profiling
    profile_application

    log "SUCCESS" "Local testing completed successfully"
}

# Execute main testing workflow
main 