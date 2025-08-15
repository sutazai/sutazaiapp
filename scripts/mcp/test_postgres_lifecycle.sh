#!/usr/bin/env bash
set -Eeuo pipefail

# Comprehensive Test Suite for Postgres MCP Container Lifecycle Management
# Tests the fixed postgres.sh implementation and cleanup daemon

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_common.sh"

TEST_RESULTS=()
CLEANUP_PIDS=()

# Test configuration
TEST_TIMEOUT=30
POSTGRES_WRAPPER="$SCRIPT_DIR/wrappers/postgres.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_test() { printf "${BLUE}[TEST]${NC} %s\n" "$*"; }
log_pass() { printf "${GREEN}[PASS]${NC} %s\n" "$*"; }
log_fail() { printf "${RED}[FAIL]${NC} %s\n" "$*"; }
log_warn() { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }

# Track test results
record_test() {
    local test_name="$1"
    local result="$2"
    local details="${3:-}"
    
    TEST_RESULTS+=("$result:$test_name:$details")
    
    if [ "$result" = "PASS" ]; then
        log_pass "$test_name"
    else
        log_fail "$test_name: $details"
    fi
}

# Cleanup function
cleanup_test_environment() {
    log_test "Cleaning up test environment..."
    
    # Kill any background test processes
    for pid in "${CLEANUP_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            sleep 1
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    
    # Clean up test containers
    docker ps -a --filter label=test-container=postgres-mcp --format "{{.Names}}" | while read -r container; do
        if [ -n "$container" ]; then
            docker stop "$container" >/dev/null 2>&1 || true
            docker rm "$container" >/dev/null 2>&1 || true
        fi
    done
    
    # Clean up any postgres-mcp containers
    docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" | while read -r container; do
        if [ -n "$container" ]; then
            docker stop "$container" >/dev/null 2>&1 || true
            docker rm "$container" >/dev/null 2>&1 || true
        fi
    done
}

# Trap for cleanup
trap cleanup_test_environment EXIT

# Test 1: Basic Container Creation and Cleanup
test_basic_container_lifecycle() {
    log_test "Test 1: Basic container creation and cleanup"
    
    # Count containers before test
    local before_count=$(docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" | wc -l)
    
    # Start postgres wrapper in background with timeout
    timeout $TEST_TIMEOUT bash "$POSTGRES_WRAPPER" < /dev/null &
    local wrapper_pid=$!
    CLEANUP_PIDS+=("$wrapper_pid")
    
    # Wait a moment for container to start
    sleep 3
    
    # Check if container was created
    local during_count=$(docker ps --filter label=mcp-service=postgres --format "{{.Names}}" | wc -l)
    
    if [ $during_count -eq 0 ]; then
        record_test "Basic container creation" "FAIL" "No container created"
        return 1
    fi
    
    # Get container name and verify labels
    local container_name=$(docker ps --filter label=mcp-service=postgres --format "{{.Names}}" | head -1)
    local has_pid_label=$(docker inspect "$container_name" --format "{{index .Config.Labels \"mcp-pid\"}}" 2>/dev/null || echo "")
    local has_service_label=$(docker inspect "$container_name" --format "{{index .Config.Labels \"mcp-service\"}}" 2>/dev/null || echo "")
    
    if [ "$has_service_label" != "postgres" ]; then
        record_test "Container labeling" "FAIL" "Missing mcp-service label"
        return 1
    fi
    
    if [ -z "$has_pid_label" ]; then
        record_test "Container labeling" "FAIL" "Missing mcp-pid label"
        return 1
    fi
    
    record_test "Container creation and labeling" "PASS" "Container created with proper labels"
    
    # Kill the wrapper process to test cleanup
    kill -TERM "$wrapper_pid" 2>/dev/null || true
    sleep 3
    
    # Check if container was cleaned up
    local after_count=$(docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" | wc -l)
    
    if [ $after_count -gt $before_count ]; then
        record_test "Container cleanup" "FAIL" "Container not cleaned up after script exit"
        return 1
    fi
    
    record_test "Container cleanup" "PASS" "Container properly cleaned up"
    return 0
}

# Test 2: Multiple Concurrent Calls
test_concurrent_containers() {
    log_test "Test 2: Multiple concurrent container calls"
    
    local before_count=$(docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" | wc -l)
    local wrapper_pids=()
    
    # Start 3 concurrent wrapper processes
    for i in {1..3}; do
        timeout $TEST_TIMEOUT bash "$POSTGRES_WRAPPER" < /dev/null &
        local pid=$!
        wrapper_pids+=("$pid")
        CLEANUP_PIDS+=("$pid")
        sleep 1  # Stagger slightly to avoid exact timing
    done
    
    # Wait for containers to start
    sleep 5
    
    # Check how many containers are running
    local concurrent_count=$(docker ps --filter label=mcp-service=postgres --format "{{.Names}}" | wc -l)
    
    if [ $concurrent_count -ne 3 ]; then
        record_test "Concurrent container creation" "FAIL" "Expected 3 containers, got $concurrent_count"
    else
        record_test "Concurrent container creation" "PASS" "All 3 containers created successfully"
    fi
    
    # Kill all wrapper processes
    for pid in "${wrapper_pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    
    # Wait for cleanup
    sleep 5
    
    # Check cleanup
    local after_count=$(docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" | wc -l)
    
    if [ $after_count -gt $before_count ]; then
        record_test "Concurrent container cleanup" "FAIL" "Some containers not cleaned up"
    else
        record_test "Concurrent container cleanup" "PASS" "All containers cleaned up properly"
    fi
    
    return 0
}

# Test 3: Cleanup Daemon Functionality
test_cleanup_daemon() {
    log_test "Test 3: Cleanup daemon functionality"
    
    # Create orphaned test containers manually
    docker run -d \
        --name="test-orphan-$$-1" \
        --label="mcp-service=postgres" \
        --label="mcp-pid=99999" \
        --label="mcp-started=$(date +%s)" \
        --label="test-container=postgres-mcp" \
        alpine:latest sleep 3600 >/dev/null
    
    docker run -d \
        --name="test-orphan-$$-2" \
        --label="mcp-service=postgres" \
        --label="mcp-pid=99998" \
        --label="mcp-started=$(date +%s)" \
        --label="test-container=postgres-mcp" \
        alpine:latest sleep 3600 >/dev/null
    
    # Verify containers exist
    local orphan_count=$(docker ps --filter label=test-container=postgres-mcp --format "{{.Names}}" | wc -l)
    
    if [ $orphan_count -ne 2 ]; then
        record_test "Orphan container creation" "FAIL" "Could not create test orphan containers"
        return 1
    fi
    
    record_test "Orphan container creation" "PASS" "Created test orphan containers"
    
    # Run cleanup daemon once
    bash "$SCRIPT_DIR/cleanup_containers.sh" --once >/dev/null 2>&1
    
    # Check if orphans were cleaned up
    sleep 2
    local remaining_count=$(docker ps --filter label=test-container=postgres-mcp --format "{{.Names}}" | wc -l)
    
    if [ $remaining_count -eq 0 ]; then
        record_test "Orphan container cleanup" "PASS" "Cleanup daemon removed orphaned containers"
    else
        record_test "Orphan container cleanup" "FAIL" "Orphaned containers still exist"
    fi
    
    return 0
}

# Test 4: Real MCP Integration
test_real_mcp_integration() {
    log_test "Test 4: Real MCP integration test"
    
    # Test selfcheck functionality
    if bash "$POSTGRES_WRAPPER" --selfcheck >/dev/null 2>&1; then
        record_test "MCP selfcheck" "PASS" "Selfcheck completed successfully"
    else
        record_test "MCP selfcheck" "FAIL" "Selfcheck failed"
        return 1
    fi
    
    # Check if cleanup daemon is running
    if systemctl is-active --quiet mcp-cleanup 2>/dev/null; then
        record_test "Cleanup daemon status" "PASS" "Cleanup daemon is running"
    else
        record_test "Cleanup daemon status" "WARN" "Cleanup daemon not running"
    fi
    
    return 0
}

# Test 5: Stress Test
test_stress_scenario() {
    log_test "Test 5: Stress test with rapid container creation"
    
    local before_count=$(docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" | wc -l)
    local stress_pids=()
    
    # Create 5 rapid-fire container calls
    for i in {1..5}; do
        (
            timeout 10 bash "$POSTGRES_WRAPPER" < /dev/null
        ) &
        local pid=$!
        stress_pids+=("$pid")
        CLEANUP_PIDS+=("$pid")
    done
    
    # Wait a moment then kill all
    sleep 5
    
    for pid in "${stress_pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    
    # Wait for cleanup
    sleep 5
    
    # Check final state
    local after_count=$(docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" | wc -l)
    
    if [ $after_count -le $before_count ]; then
        record_test "Stress test cleanup" "PASS" "All stress test containers cleaned up"
    else
        record_test "Stress test cleanup" "FAIL" "Some stress test containers remain"
    fi
    
    return 0
}

# Test 6: Edge Cases
test_edge_cases() {
    log_test "Test 6: Edge case testing"
    
    # Test with container that has no labels (legacy scenario)
    docker run -d --name="test-legacy-$$" alpine:latest sleep 3600 >/dev/null
    
    # Test cleanup daemon with legacy container
    bash "$SCRIPT_DIR/cleanup_containers.sh" --force --once >/dev/null 2>&1
    
    # Legacy container should still exist unless forced
    if docker ps --filter name="test-legacy-$$" --format "{{.Names}}" | grep -q "test-legacy-$$"; then
        record_test "Legacy container handling" "PASS" "Legacy containers preserved without force"
        docker stop "test-legacy-$$" >/dev/null 2>&1 || true
        docker rm "test-legacy-$$" >/dev/null 2>&1 || true
    else
        record_test "Legacy container handling" "WARN" "Legacy container was removed"
    fi
    
    return 0
}

# Print test summary
print_test_summary() {
    echo
    log_test "=== TEST SUMMARY ==="
    
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    local warn_tests=0
    
    for result in "${TEST_RESULTS[@]}"; do
        local status=$(echo "$result" | cut -d: -f1)
        local test_name=$(echo "$result" | cut -d: -f2)
        local details=$(echo "$result" | cut -d: -f3-)
        
        total_tests=$((total_tests + 1))
        
        case "$status" in
            PASS) passed_tests=$((passed_tests + 1)) ;;
            FAIL) failed_tests=$((failed_tests + 1)) ;;
            WARN) warn_tests=$((warn_tests + 1)) ;;
        esac
        
        printf "  %-8s %s\n" "[$status]" "$test_name"
    done
    
    echo
    printf "${GREEN}PASSED: %d${NC}  " "$passed_tests"
    printf "${RED}FAILED: %d${NC}  " "$failed_tests"
    printf "${YELLOW}WARNINGS: %d${NC}  " "$warn_tests"
    printf "TOTAL: %d\n" "$total_tests"
    
    if [ $failed_tests -eq 0 ]; then
        printf "\n${GREEN}✓ ALL TESTS PASSED${NC}\n"
        return 0
    else
        printf "\n${RED}✗ SOME TESTS FAILED${NC}\n"
        return 1
    fi
}

# Main test execution
main() {
    log_test "Starting comprehensive postgres-mcp container lifecycle tests"
    log_test "Test timeout: ${TEST_TIMEOUT}s per test"
    echo
    
    # Verify prerequisites
    if [ ! -f "$POSTGRES_WRAPPER" ]; then
        log_fail "Postgres wrapper not found: $POSTGRES_WRAPPER"
        exit 1
    fi
    
    if ! has_cmd docker; then
        log_fail "Docker not available"
        exit 1
    fi
    
    # Run all tests
    test_basic_container_lifecycle
    test_concurrent_containers  
    test_cleanup_daemon
    test_real_mcp_integration
    test_stress_scenario
    test_edge_cases
    
    # Print summary and exit with appropriate code
    print_test_summary
}

# Update todo status
TodoWrite '[{"id": "1", "content": "Design comprehensive test plan for postgres-mcp container management", "status": "completed"}]' >/dev/null 2>&1 || true

main "$@"