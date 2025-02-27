#!/bin/bash
set -euo pipefail

# Sutazaiapp Comprehensive CI/CD Pipeline

# Configuration
SUTAZAIAPP_HOME="/opt/sutazaiapp"
VENV_PATH="$SUTAZAIAPP_HOME/venv"
RESULTS_DIR="$SUTAZAIAPP_HOME/ci_results"

# Logging
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# Validate OTP for external dependencies
validate_otp() {
    python3 -c "
from scripts.otp_override import OTPManager

otp_manager = OTPManager()
is_valid = otp_manager.validate_otp('$1')
exit(0) if is_valid else exit(1)
"
}

# Static Analysis
run_static_analysis() {
    log "Running Static Analysis"
    mkdir -p "$RESULTS_DIR/static_analysis"
    
    # Semgrep Scan
    semgrep scan "$SUTAZAIAPP_HOME" \
        --config=r/all \
        --output "$RESULTS_DIR/static_analysis/semgrep_results.json"
    
    # Bandit Security Scan
    bandit -r "$SUTAZAIAPP_HOME" \
        -f json \
        -o "$RESULTS_DIR/static_analysis/bandit_results.json"
}

# Unit and Integration Tests
run_tests() {
    log "Running Tests"
    mkdir -p "$RESULTS_DIR/tests"
    
    source "$VENV_PATH/bin/activate"
    
    # Run pytest with coverage
    pytest \
        --cov="$SUTAZAIAPP_HOME/backend" \
        --cov-report=json:"$RESULTS_DIR/tests/coverage.json" \
        "$SUTAZAIAPP_HOME/tests"
}

# Performance Testing
run_performance_tests() {
    log "Running Performance Tests"
    mkdir -p "$RESULTS_DIR/performance"
    
    # Use locust for performance testing
    locust \
        -f "$SUTAZAIAPP_HOME/tests/performance_tests.py" \
        --headless \
        -u 100 \
        -r 10 \
        --run-time 1m \
        --csv "$RESULTS_DIR/performance/locust_results"
}

# Deployment
deploy() {
    log "Deploying Application"
    
    # Require OTP for deployment
    if [ $# -ne 1 ]; then
        log "OTP required for deployment"
        exit 1
    fi
    
    validate_otp "$1"
    
    "$SUTAZAIAPP_HOME/scripts/deploy.sh" "$1"
}

# Main Pipeline
main() {
    log "Starting Sutazaiapp CI/CD Pipeline"
    
    run_static_analysis
    run_tests
    run_performance_tests
    
    # Optional deployment stage
    if [ $# -eq 1 ]; then
        deploy "$1"
    fi
    
    log "CI/CD Pipeline Completed Successfully"
}

# Execute main function
main "$@" 