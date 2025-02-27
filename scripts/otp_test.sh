#!/bin/bash
set -euo pipefail

# OTP Enforcement Testing Script

# Logging configuration
TEST_LOG_FILE="/var/log/sutazaiapp/otp_tests.log"

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Logging function
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$TEST_LOG_FILE"
}

# Generate a valid OTP
generate_valid_otp() {
    python3 -c "
from scripts.otp_override import OTPManager

otp_manager = OTPManager()
print(otp_manager.generate_otp())
"
}

# Test OTP validation
test_otp_validation() {
    local test_name="$1"
    local otp="$2"
    local expected_result="$3"
    
    log_message "Running test: $test_name"
    
    # Validate OTP
    validation_result=$(python3 -c "
from scripts.otp_override import OTPManager

otp_manager = OTPManager()
is_valid = otp_manager.validate_otp('$otp')
print('VALID' if is_valid else 'INVALID')
")
    
    if [ "$validation_result" == "$expected_result" ]; then
        echo -e "${GREEN}✓ $test_name PASSED${NC}"
        log_message "$test_name: PASSED"
    else
        echo -e "${RED}✗ $test_name FAILED${NC}"
        log_message "$test_name: FAILED"
        return 1
    fi
}

# Test external call blocking
test_external_call_blocking() {
    local stage="$1"
    local invalid_otp="000000"
    
    log_message "Testing external call blocking for stage: $stage"
    
    # Attempt deployment with invalid OTP
    if /opt/sutazaiapp/scripts/deploy.sh "$stage" "$invalid_otp"; then
        echo -e "${RED}✗ External Call Blocking Test FAILED${NC}"
        log_message "External call blocking test failed for $stage"
        return 1
    else
        echo -e "${GREEN}✓ External Call Blocking Test PASSED${NC}"
        log_message "External call blocking test passed for $stage"
    fi
}

# Comprehensive OTP test suite
run_otp_tests() {
    local valid_otp
    valid_otp=$(generate_valid_otp)
    
    # Test 1: Valid OTP Validation
    test_otp_validation "Valid OTP Test" "$valid_otp" "VALID"
    
    # Test 2: Invalid OTP Validation
    test_otp_validation "Invalid OTP Test" "000000" "INVALID"
    
    # Test 3: External Call Blocking - Git Pull
    test_external_call_blocking "git_pull"
    
    # Test 4: External Call Blocking - Pip Install
    test_external_call_blocking "pip_install"
    
    # Test 5: External Call Blocking - Database Migration
    test_external_call_blocking "database_migration"
}

# Main test execution
main() {
    log_message "Starting OTP Enforcement Test Suite"
    
    # Run tests
    if run_otp_tests; then
        echo -e "${GREEN}✓ All OTP Tests Passed Successfully${NC}"
        log_message "All OTP tests passed"
        exit 0
    else
        echo -e "${RED}✗ Some OTP Tests Failed${NC}"
        log_message "Some OTP tests failed"
        exit 1
    fi
}

# Execute main function
main 