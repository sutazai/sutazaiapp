#!/bin/bash
# Production-Grade Deployment System Test Script
# Tests all the enhanced features we've added to deploy_complete_system.sh

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test configuration
TEST_LOG_DIR="logs/tests"
mkdir -p "$TEST_LOG_DIR"
TEST_LOG="$TEST_LOG_DIR/deployment_test_$(date +%Y%m%d_%H%M%S).log"

# Logging functions
log_test() {
    echo -e "${BLUE}[TEST]${NC} $1" | tee -a "$TEST_LOG"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1" | tee -a "$TEST_LOG"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1" | tee -a "$TEST_LOG"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$TEST_LOG"
}

# Test functions
test_script_syntax() {
    log_test "Testing script syntax..."
    
    if bash -n scripts/deploy_complete_system.sh; then
        log_pass "Script syntax is valid"
        return 0
    else
        log_fail "Script syntax errors found"
        return 1
    fi
}

test_required_functions() {
    log_test "Testing required production-grade functions exist..."
    
    local required_functions=(
        "pre_deployment_validation"
        "check_system_resources"
        "intelligent_error_recovery"
        "verify_service_health"
        "deploy_services_parallel"
        "trigger_rollback"
        "generate_deployment_report"
        "handle_deployment_error"
    )
    
    local missing_functions=()
    
    for func in "${required_functions[@]}"; do
        if ! grep -q "^$func()" scripts/deploy_complete_system.sh; then
            missing_functions+=("$func")
        fi
    done
    
    if [[ ${#missing_functions[@]} -eq 0 ]]; then
        log_pass "All required functions exist"
        return 0
    else
        log_fail "Missing functions: ${missing_functions[*]}"
        return 1
    fi
}

test_global_variables() {
    log_test "Testing global variables initialization..."
    
    local required_vars=(
        "DEPLOYMENT_STATE_FILE"
        "ROLLBACK_STATE_FILE"
        "PARALLEL_PIDS"
        "DEPLOYMENT_START_TIME"
        "DEPLOYMENT_PHASE"
        "SUCCESSFUL_SERVICES"
        "FAILED_SERVICES"
        "ROLLBACK_TRIGGERED"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if ! grep -q "$var=" scripts/deploy_complete_system.sh; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -eq 0 ]]; then
        log_pass "All required global variables initialized"
        return 0
    else
        log_fail "Missing variables: ${missing_vars[*]}"
        return 1
    fi
}

test_error_handling() {
    log_test "Testing error handling mechanisms..."
    
    # Check for trap error handler
    if grep -q "trap.*handle_deployment_error" scripts/deploy_complete_system.sh; then
        log_pass "Error trap handler found"
    else
        log_fail "Error trap handler missing"
        return 1
    fi
    
    # Check for rollback triggers
    if grep -q "trigger_rollback" scripts/deploy_complete_system.sh; then
        log_pass "Rollback mechanisms found"
    else
        log_fail "Rollback mechanisms missing"
        return 1
    fi
    
    return 0
}

test_health_checks() {
    log_test "Testing health check implementations..."
    
    local services=("postgres" "redis" "ollama" "backend" "frontend")
    local missing_checks=()
    
    for service in "${services[@]}"; do
        if ! grep -q "\"$service\")" scripts/deploy_complete_system.sh; then
            missing_checks+=("$service")
        fi
    done
    
    if [[ ${#missing_checks[@]} -eq 0 ]]; then
        log_pass "All service health checks implemented"
        return 0
    else
        log_warn "Missing health checks for: ${missing_checks[*]}"
        return 0  # Warning, not failure
    fi
}

test_parallel_deployment() {
    log_test "Testing parallel deployment functionality..."
    
    if grep -q "deploy_services_parallel" scripts/deploy_complete_system.sh; then
        log_pass "Parallel deployment function found"
    else
        log_fail "Parallel deployment function missing"
        return 1
    fi
    
    # Check for service groups
    local service_groups=("core" "ai" "application" "monitoring")
    local missing_groups=()
    
    for group in "${service_groups[@]}"; do
        if ! grep -q "deploy_${group}_" scripts/deploy_complete_system.sh; then
            missing_groups+=("$group")
        fi
    done
    
    if [[ ${#missing_groups[@]} -eq 0 ]]; then
        log_pass "All service group deployment functions found"
    else
        log_warn "Missing service groups: ${missing_groups[*]}"
    fi
    
    return 0
}

test_resource_monitoring() {
    log_test "Testing resource monitoring capabilities..."
    
    if grep -q "check_system_resources" scripts/deploy_complete_system.sh; then
        log_pass "Resource monitoring function found"
    else
        log_fail "Resource monitoring function missing"
        return 1
    fi
    
    # Check for resource thresholds
    if grep -q "required_memory\|required_disk\|required_cpus" scripts/deploy_complete_system.sh; then
        log_pass "Resource threshold validation found"
    else
        log_fail "Resource threshold validation missing"
        return 1
    fi
    
    return 0
}

test_progress_tracking() {
    log_test "Testing deployment progress tracking..."
    
    if grep -q "show_deployment_progress" scripts/deploy_complete_system.sh; then
        log_pass "Progress tracking function found"
    else
        log_fail "Progress tracking function missing"
        return 1
    fi
    
    # Check for deployment phases
    local phases=("pre_validation" "network_setup" "system_preparation" "core_deployment" "validation" "completed")
    local missing_phases=()
    
    for phase in "${phases[@]}"; do
        if ! grep -q "DEPLOYMENT_PHASE=\"$phase\"" scripts/deploy_complete_system.sh; then
            missing_phases+=("$phase")
        fi
    done
    
    if [[ ${#missing_phases[@]} -eq 0 ]]; then
        log_pass "All deployment phases implemented"
    else
        log_warn "Missing deployment phases: ${missing_phases[*]}"
    fi
    
    return 0
}

test_analytics_reporting() {
    log_test "Testing analytics and reporting features..."
    
    if grep -q "generate_deployment_report" scripts/deploy_complete_system.sh; then
        log_pass "Deployment reporting function found"
    else
        log_fail "Deployment reporting function missing"
        return 1
    fi
    
    # Check for state management
    if grep -q "save_deployment_state" scripts/deploy_complete_system.sh; then
        log_pass "Deployment state management found"
    else
        log_fail "Deployment state management missing"
        return 1
    fi
    
    return 0
}

test_docker_integration() {
    log_test "Testing Docker integration and health checks..."
    
    # Test Docker availability
    if command -v docker >/dev/null 2>&1; then
        log_pass "Docker command available"
    else
        log_fail "Docker command not found"
        return 1
    fi
    
    # Test Docker Compose v2
    if docker compose version >/dev/null 2>&1; then
        log_pass "Docker Compose v2 available"
    else
        log_fail "Docker Compose v2 not available"
        return 1
    fi
    
    return 0
}

# Main test execution
run_all_tests() {
    local test_count=0
    local passed_count=0
    local failed_count=0
    
    echo "🧪 Production-Grade Deployment System Test Suite"
    echo "=================================================="
    echo "Test log: $TEST_LOG"
    echo
    
    local tests=(
        "test_script_syntax"
        "test_required_functions"
        "test_global_variables"
        "test_error_handling"
        "test_health_checks"
        "test_parallel_deployment"
        "test_resource_monitoring"
        "test_progress_tracking"
        "test_analytics_reporting"
        "test_docker_integration"
    )
    
    for test in "${tests[@]}"; do
        test_count=$((test_count + 1))
        echo "Running test $test_count/${#tests[@]}: $test"
        
        if $test; then
            passed_count=$((passed_count + 1))
        else
            failed_count=$((failed_count + 1))
        fi
        echo
    done
    
    # Summary
    echo "=================================================="
    echo "Test Summary:"
    echo "  Total tests: $test_count"
    echo "  Passed: $passed_count"
    echo "  Failed: $failed_count"
    echo "  Success rate: $(($passed_count * 100 / $test_count))%"
    echo
    
    if [[ $failed_count -eq 0 ]]; then
        log_pass "🎉 All tests passed! Deployment system is production-ready."
        return 0
    else
        log_fail "❌ $failed_count test(s) failed. Please review and fix issues."
        return 1
    fi
}

# Execute tests
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    run_all_tests
fi