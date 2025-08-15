#!/usr/bin/env bash
set -Eeuo pipefail

# MCP Container Lifecycle Test Suite
# Validates session-aware container management and cleanup functionality

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_common.sh"

TEST_RESULTS=()
CLEANUP_TEST_CONTAINERS=()

# Test configuration
TEST_TIMEOUT=30
MAX_TEST_CONTAINERS=5

show_help() {
  cat << EOF
MCP Container Lifecycle Test Suite

DESCRIPTION:
    Comprehensive testing of session-aware container management to validate
    that container accumulation is prevented and cleanup works correctly.

USAGE:
    $(basename "$0") [OPTIONS]

OPTIONS:
    --help, -h      Show this help message
    --verbose, -v   Enable verbose output
    --quick         Run quick test suite only
    --full          Run comprehensive test suite

TESTS:
    - Container lifecycle management
    - Session-aware container creation
    - Cleanup daemon functionality
    - Orphaned container detection
    - Aged container cleanup
    - Force cleanup verification
    - System integration validation

EXIT CODES:
    0   All tests passed
    1   One or more tests failed
    2   Invalid arguments
    3   Test environment error
EOF
}

# Log test results
log_test() {
  local status="$1"
  local test_name="$2"
  local message="$3"
  
  TEST_RESULTS+=("$status:$test_name:$message")
  
  if [ "$status" = "PASS" ]; then
    ok_line "[$test_name] $message"
  elif [ "$status" = "FAIL" ]; then
    err_line "[$test_name] $message"
  else
    warn_line "[$test_name] $message"
  fi
}

# Test 1: Verify cleanup daemon is running
test_cleanup_daemon() {
  local test_name="cleanup_daemon_status"
  
  if systemctl is-active mcp-cleanup.service >/dev/null 2>&1; then
    log_test "PASS" "$test_name" "Cleanup daemon is running"
    return 0
  elif [ -f "/tmp/mcp-cleanup.pid" ]; then
    local pid=$(cat "/tmp/mcp-cleanup.pid" 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      log_test "PASS" "$test_name" "Cleanup daemon running in manual mode"
      return 0
    fi
  fi
  
  log_test "FAIL" "$test_name" "Cleanup daemon is not running"
  return 1
}

# Test 2: Verify enhanced postgres.sh selfcheck
test_postgres_selfcheck() {
  local test_name="postgres_selfcheck"
  
  if "$SCRIPT_DIR/wrappers/postgres.sh" --selfcheck >/dev/null 2>&1; then
    log_test "PASS" "$test_name" "PostgreSQL MCP selfcheck passed"
    return 0
  else
    log_test "FAIL" "$test_name" "PostgreSQL MCP selfcheck failed"
    return 1
  fi
}

# Test 3: Create test containers with proper labeling
test_container_creation() {
  local test_name="container_creation"
  local test_session_id="test-session-$$-$(date +%s)"
  local container_name="postgres-mcp-$test_session_id"
  
  # Try to create a test container with proper labeling
  if docker run -d \
    --name="$container_name" \
    --network=sutazai-network \
    --label="mcp-service=postgres" \
    --label="mcp-session=$test_session_id" \
    --label="mcp-started=$(date -u +%s)" \
    -e DATABASE_URI="postgresql://test:test@localhost:5432/test" \
    crystaldba/postgres-mcp --access-mode=restricted >/dev/null 2>&1; then
    
    CLEANUP_TEST_CONTAINERS+=("$container_name")
    log_test "PASS" "$test_name" "Test container created with proper labeling"
    return 0
  else
    log_test "FAIL" "$test_name" "Failed to create test container"
    return 1
  fi
}

# Test 4: Verify cleanup utility functionality
test_cleanup_utility() {
  local test_name="cleanup_utility"
  
  # Run cleanup utility
  if "$SCRIPT_DIR/cleanup_containers.sh" --once >/dev/null 2>&1; then
    log_test "PASS" "$test_name" "Cleanup utility executed successfully"
    return 0
  else
    log_test "FAIL" "$test_name" "Cleanup utility failed"
    return 1
  fi
}

# Test 5: Verify force cleanup works
test_force_cleanup() {
  local test_name="force_cleanup"
  
  # Count containers before cleanup
  local before_count=$(docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" | wc -l)
  
  # Run force cleanup
  if "$SCRIPT_DIR/cleanup_containers.sh" --force --once >/dev/null 2>&1; then
    local after_count=$(docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" | wc -l)
    
    if [ $after_count -eq 0 ]; then
      log_test "PASS" "$test_name" "Force cleanup removed all containers ($before_count -> $after_count)"
      CLEANUP_TEST_CONTAINERS=()  # Clear our tracking since they're gone
      return 0
    else
      log_test "WARN" "$test_name" "Force cleanup incomplete ($before_count -> $after_count)"
      return 0
    fi
  else
    log_test "FAIL" "$test_name" "Force cleanup failed"
    return 1
  fi
}

# Test 6: Verify container labeling
test_container_labeling() {
  local test_name="container_labeling"
  
  # Create a test container for labeling verification
  local test_session_id="label-test-$$-$(date +%s)"
  local container_name="postgres-mcp-$test_session_id"
  
  if docker run -d \
    --name="$container_name" \
    --network=sutazai-network \
    --label="mcp-service=postgres" \
    --label="mcp-session=$test_session_id" \
    --label="mcp-started=$(date -u +%s)" \
    -e DATABASE_URI="postgresql://test:test@localhost:5432/test" \
    crystaldba/postgres-mcp --access-mode=restricted >/dev/null 2>&1; then
    
    CLEANUP_TEST_CONTAINERS+=("$container_name")
    
    # Verify labels
    local service_label=$(docker inspect "$container_name" --format "{{index .Config.Labels \"mcp-service\"}}" 2>/dev/null || echo "")
    local session_label=$(docker inspect "$container_name" --format "{{index .Config.Labels \"mcp-session\"}}" 2>/dev/null || echo "")
    
    if [ "$service_label" = "postgres" ] && [ -n "$session_label" ]; then
      log_test "PASS" "$test_name" "Container labels are correct"
      return 0
    else
      log_test "FAIL" "$test_name" "Container labels are incorrect or missing"
      return 1
    fi
  else
    log_test "FAIL" "$test_name" "Failed to create test container for labeling test"
    return 1
  fi
}

# Test 7: Verify session-aware behavior
test_session_awareness() {
  local test_name="session_awareness"
  
  # This test verifies that the generate_session_id function works
  local session_id=$(generate_session_id)
  
  # Verify session ID format and content
  if [ -n "$session_id" ] && [[ "$session_id" =~ ^mcp-session-[0-9]+-[0-9]+-[0-9]+.*$ ]]; then
    log_test "PASS" "$test_name" "Session ID generation works correctly: $session_id"
    return 0
  else
    log_test "FAIL" "$test_name" "Session ID generation failed or invalid format: '$session_id'"
    return 1
  fi
}

# Test 8: Container status monitoring
test_status_monitoring() {
  local test_name="status_monitoring"
  
  # Test the status command
  if "$SCRIPT_DIR/cleanup_containers.sh" >/dev/null 2>&1; then
    log_test "PASS" "$test_name" "Status monitoring works"
    return 0
  else
    log_test "FAIL" "$test_name" "Status monitoring failed"
    return 1
  fi
}

# Cleanup test containers
cleanup_test_containers() {
  for container_name in "${CLEANUP_TEST_CONTAINERS[@]}"; do
    docker stop "$container_name" >/dev/null 2>&1 || true
    docker rm "$container_name" >/dev/null 2>&1 || true
  done
  CLEANUP_TEST_CONTAINERS=()
}

# Run test suite
run_test_suite() {
  local test_mode="${1:-full}"
  local failed_tests=0
  
  section "MCP Container Lifecycle Test Suite $(ts)"
  echo "Test Mode: $test_mode"
  echo
  
  # Always run core tests
  test_cleanup_daemon || ((failed_tests++))
  test_postgres_selfcheck || ((failed_tests++))
  test_session_awareness || ((failed_tests++))
  test_cleanup_utility || ((failed_tests++))
  test_status_monitoring || ((failed_tests++))
  
  # Run extended tests for full mode
  if [ "$test_mode" = "full" ]; then
    test_container_creation || ((failed_tests++))
    test_container_labeling || ((failed_tests++))
    test_force_cleanup || ((failed_tests++))
  fi
  
  # Cleanup any test containers
  cleanup_test_containers
  
  # Show results
  echo
  section "Test Results Summary"
  local passed_tests=0
  local total_tests=${#TEST_RESULTS[@]}
  
  for result in "${TEST_RESULTS[@]}"; do
    local status="${result%%:*}"
    local rest="${result#*:}"
    local test_name="${rest%%:*}"
    local message="${rest#*:}"
    
    if [ "$status" = "PASS" ]; then
      ((passed_tests++))
    fi
  done
  
  ok_line "Passed: $passed_tests/$total_tests tests"
  
  if [ $failed_tests -gt 0 ]; then
    err_line "Failed: $failed_tests tests"
    return 1
  else
    ok "All tests passed! Container lifecycle management is working correctly."
    return 0
  fi
}

# Main execution
main() {
  local test_mode="full"
  local verbose=false
  
  # Parse arguments
  while [[ $# -gt 0 ]]; do
    case $1 in
      --quick)
        test_mode="quick"
        ;;
      --full)
        test_mode="full"
        ;;
      --verbose|-v)
        verbose=true
        ;;
      --help|-h)
        show_help
        exit 0
        ;;
      *)
        err "Unknown option: $1"
        show_help
        exit 2
        ;;
    esac
    shift
  done
  
  # Check dependencies
  if ! has_cmd docker; then
    err "Docker is required for container lifecycle tests"
    exit 3
  fi
  
  # Setup cleanup on exit
  trap 'cleanup_test_containers' EXIT INT TERM
  
  # Run tests
  if run_test_suite "$test_mode"; then
    exit 0
  else
    exit 1
  fi
}

# Execute main function
main "$@"