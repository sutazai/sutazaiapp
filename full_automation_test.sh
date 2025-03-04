#!/bin/bash

# Full Automation Test Script for SutazaiApp
# This script performs a complete test cycle with the goal of 100% success rate

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/opt/sutazaiapp"
SSH_KEY="/root/.ssh/sutazaiapp_sync_key"
REMOTE_SERVER="root@192.168.100.100"
REMOTE_PROJECT_PATH="/opt/sutazaiapp"
LOG_DIR="${PROJECT_ROOT}/logs"
RETRIES=3

# Print section header
section() {
    echo -e "\n${BLUE}===== $1 =====${NC}\n"
}

# Print success message
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Print error message
error() {
    echo -e "${RED}✗ $1${NC}"
}

# Print info message
info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check for required tools
check_requirements() {
    section "Checking Requirements"
    
    # Check for rsync
    if ! command -v rsync &> /dev/null; then
        error "rsync is not installed. Please install it first."
        exit 1
    else
        success "rsync is installed"
    fi
    
    # Check for SSH key
    if [ ! -f "${SSH_KEY}" ]; then
        error "SSH key not found at ${SSH_KEY}"
        exit 1
    else
        success "SSH key exists"
    fi
}

# Prepare the test environment locally
prepare_local_environment() {
    section "Preparing Local Environment"
    
    # Create necessary directories
    info "Creating necessary directories"
    mkdir -p ${PROJECT_ROOT}/logs
    mkdir -p ${PROJECT_ROOT}/coverage
    mkdir -p ${PROJECT_ROOT}/test_reports
    
    # Create log file if it doesn't exist
    touch ${PROJECT_ROOT}/logs/code_audit.log
    success "Local directories prepared"
}

# Prepare the test environment on remote server
prepare_remote_environment() {
    section "Preparing Remote Environment"
    
    # Create necessary directories on remote server
    info "Creating necessary directories on remote server"
    ssh -i ${SSH_KEY} ${REMOTE_SERVER} "mkdir -p ${REMOTE_PROJECT_PATH}/logs ${REMOTE_PROJECT_PATH}/coverage ${REMOTE_PROJECT_PATH}/test_reports"
    
    # Create log file if it doesn't exist
    ssh -i ${SSH_KEY} ${REMOTE_SERVER} "touch ${REMOTE_PROJECT_PATH}/logs/code_audit.log"
    
    # Install necessary dependencies
    info "Installing dependencies on remote server"
    ssh -i ${SSH_KEY} ${REMOTE_SERVER} "cd ${REMOTE_PROJECT_PATH} && source venv/bin/activate && pip install pytest pytest-asyncio pytest-cov pytest-xdist pytest-html psutil"
    
    success "Remote environment prepared"
}

# Deploy the latest code to the remote server
deploy_latest_code() {
    section "Deploying Latest Code"
    
    info "Syncing code to remote server..."
    rsync -av --exclude=venv --exclude=__pycache__ --exclude=.git --exclude=.pytest_cache -e "ssh -i ${SSH_KEY}" \
        ${PROJECT_ROOT}/core_system \
        ${PROJECT_ROOT}/ai_agents \
        ${PROJECT_ROOT}/scripts \
        ${PROJECT_ROOT}/backend \
        ${PROJECT_ROOT}/tests \
        ${REMOTE_SERVER}:${REMOTE_PROJECT_PATH}/
    
    if [ $? -eq 0 ]; then
        success "Code deployment successful"
    else
        error "Code deployment failed"
        exit 1
    fi
}

# Run tests with retry capability
run_tests_with_retry() {
    section "Running Tests With Retry Logic"
    
    local attempt=1
    local max_attempts=${RETRIES}
    local test_success=false
    
    while [ ${attempt} -le ${max_attempts} ]; do
        info "Test attempt ${attempt}/${max_attempts}"
        
        # Run the tests on the remote server
        ssh -i ${SSH_KEY} ${REMOTE_SERVER} "cd ${REMOTE_PROJECT_PATH} && source venv/bin/activate && ./scripts/run_tests.sh"
        test_result=$?
        
        if [ ${test_result} -eq 0 ]; then
            success "Tests passed successfully on attempt ${attempt}!"
            test_success=true
            break
        else
            if [ ${attempt} -lt ${max_attempts} ]; then
                info "Tests failed. Addressing common issues before retry..."
                
                # Fix common issues between retries
                fix_common_issues
                
                info "Waiting 3 seconds before retry..."
                sleep 3
            else
                error "All test attempts failed"
            fi
        fi
        
        ((attempt++))
    done
    
    if [ ${test_success} = true ]; then
        return 0
    else
        return 1
    fi
}

# Fix common test failures
fix_common_issues() {
    info "Ensuring all directories exist"
    ssh -i ${SSH_KEY} ${REMOTE_SERVER} "mkdir -p ${REMOTE_PROJECT_PATH}/logs ${REMOTE_PROJECT_PATH}/tmp"
    
    info "Checking for missing permissions"
    ssh -i ${SSH_KEY} ${REMOTE_SERVER} "chmod -R 755 ${REMOTE_PROJECT_PATH}/scripts"
    
    info "Ensuring log files exist"
    ssh -i ${SSH_KEY} ${REMOTE_SERVER} "touch ${REMOTE_PROJECT_PATH}/logs/code_audit.log"
    
    info "Checking for missing dependencies"
    ssh -i ${SSH_KEY} ${REMOTE_SERVER} "cd ${REMOTE_PROJECT_PATH} && source venv/bin/activate && pip install psutil pytest pytest-asyncio pytest-cov pytest-xdist pytest-html"
}

# Collect and display test results
collect_results() {
    section "Collecting Test Results"
    
    # Get the coverage report
    info "Retrieving coverage report"
    ssh -i ${SSH_KEY} ${REMOTE_SERVER} "cd ${REMOTE_PROJECT_PATH} && source venv/bin/activate && python -c 'import json; path=\"coverage/.coverage\"; data=json.load(open(path)) if os.path.exists(path) else {\"totals\": {\"percent_covered\": 0}}; print(data[\"totals\"][\"percent_covered\"])'"
    
    # Get test report
    info "Test report available at: ${REMOTE_PROJECT_PATH}/test_reports/report.html"
    
    # Option to copy reports back to local
    info "Copying reports back to local machine..."
    rsync -av -e "ssh -i ${SSH_KEY}" \
        ${REMOTE_SERVER}:${REMOTE_PROJECT_PATH}/coverage \
        ${REMOTE_SERVER}:${REMOTE_PROJECT_PATH}/test_reports \
        ${PROJECT_ROOT}/
    
    success "Test results collected"
}

# Main function
main() {
    section "Starting Full Automation Tests"
    
    # Check requirements
    check_requirements
    
    # Prepare environments
    prepare_local_environment
    prepare_remote_environment
    
    # Deploy code
    deploy_latest_code
    
    # Run tests with retry logic
    run_tests_with_retry
    test_result=$?
    
    # Collect results
    collect_results
    
    # Return final status
    if [ ${test_result} -eq 0 ]; then
        section "Testing Completed Successfully!"
        success "All tests passed with 100% success rate"
        return 0
    else
        section "Testing Failed"
        error "Tests did not achieve 100% success rate"
        return 1
    fi
}

# Run the main function
main "$@" 