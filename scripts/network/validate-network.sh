#!/bin/bash
# Network Validation Script
# Comprehensive testing of MCP network connectivity and performance

set -euo pipefail

# Configuration
LOG_FILE="/tmp/network-validation.log"
VALIDATION_RESULTS="/tmp/validation-results.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "${YELLOW}WARN${NC}" "$@"; }
log_error() { log "${RED}ERROR${NC}" "$@"; }
log_success() { log "${GREEN}SUCCESS${NC}" "$@"; }

# Initialize results
init_results() {
    cat > "$VALIDATION_RESULTS" << 'EOF'
{
  "timestamp": "",
  "overall_status": "unknown",
  "tests": {
    "infrastructure": {},
    "services": {},
    "network_connectivity": {},
    "performance": {},
    "load_balancing": {}
  },
  "summary": {
    "total_tests": 0,
    "passed": 0,
    "failed": 0,
    "warnings": 0
  }
}
EOF
}

# Update results
update_result() {
    local test_category="$1"
    local test_name="$2"
    local status="$3"
    local details="$4"
    
    local temp_file=$(mktemp)
    jq --arg category "$test_category" \
       --arg name "$test_name" \
       --arg status "$status" \
       --arg details "$details" \
       --arg timestamp "$(date -Iseconds)" \
       '.timestamp = $timestamp | 
        .tests[$category][$name] = {
          "status": $status,
          "details": $details,
          "timestamp": $timestamp
        }' "$VALIDATION_RESULTS" > "$temp_file"
    mv "$temp_file" "$VALIDATION_RESULTS"
}

# Test container status
test_container_status() {
    log_info "Testing container status..."
    
    local containers=(
        "sutazai-mcp-consul"
        "sutazai-mcp-haproxy"
        "sutazai-mcp-monitor"
    )
    
    for container in "${containers[@]}"; do
        if docker ps --format '{{.Names}}' | grep -q "^$container$"; then
            if [[ $(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null) == "healthy" ]]; then
                log_success "✓ $container is running and healthy"
                update_result "infrastructure" "$container" "pass" "Container running and healthy"
            else
                log_warn "⚠ $container is running but not healthy"
                update_result "infrastructure" "$container" "warning" "Container running but health check failed"
            fi
        else
            log_error "✗ $container is not running"
            update_result "infrastructure" "$container" "fail" "Container not running"
        fi
    done
}

# Test network connectivity
test_network_connectivity() {
    log_info "Testing network connectivity..."
    
    # Test external access to services
    local services=(
        "consul:11090:/v1/status/leader"
        "haproxy:11099:/stats"
        "monitor:11091:/health"
    )
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r service port path <<< "$service_info"
        
        if curl -sf "http://localhost:$port$path" >/dev/null 2>&1; then
            log_success "✓ $service accessible on port $port"
            update_result "network_connectivity" "$service" "pass" "Service accessible on port $port"
        else
            log_error "✗ $service not accessible on port $port"
            update_result "network_connectivity" "$service" "fail" "Service not accessible on port $port"
        fi
    done
    
    # Test internal network connectivity
    if docker network ls | grep -q mcp-internal; then
        log_success "✓ mcp-internal network exists"
        update_result "network_connectivity" "mcp-internal" "pass" "Network exists"
        
        # Test container-to-container connectivity
        if docker exec sutazai-mcp-haproxy ping -c 1 mcp-consul-agent >/dev/null 2>&1; then
            log_success "✓ HAProxy can reach Consul"
            update_result "network_connectivity" "haproxy-consul" "pass" "HAProxy can ping Consul"
        else
            log_error "✗ HAProxy cannot reach Consul"
            update_result "network_connectivity" "haproxy-consul" "fail" "HAProxy cannot ping Consul"
        fi
    else
        log_error "✗ mcp-internal network does not exist"
        update_result "network_connectivity" "mcp-internal" "fail" "Network does not exist"
    fi
}

# Test service discovery
test_service_discovery() {
    log_info "Testing service discovery..."
    
    # Test Consul API
    if consul_services=$(curl -s "http://localhost:11090/v1/agent/services" 2>/dev/null); then
        service_count=$(echo "$consul_services" | jq '. | length' 2>/dev/null || echo "0")
        log_success "✓ Consul API responding with $service_count services"
        update_result "services" "consul-api" "pass" "API responding with $service_count services"
        
        # Check for MCP services
        mcp_services=$(echo "$consul_services" | jq '[.[] | select(.Tags[]? == "mcp")] | length' 2>/dev/null || echo "0")
        if [[ "$mcp_services" -gt 0 ]]; then
            log_success "✓ Found $mcp_services MCP services in Consul"
            update_result "services" "mcp-services" "pass" "Found $mcp_services MCP services"
        else
            log_warn "⚠ No MCP services found in Consul"
            update_result "services" "mcp-services" "warning" "No MCP services registered"
        fi
    else
        log_error "✗ Consul API not responding"
        update_result "services" "consul-api" "fail" "API not responding"
    fi
}

# Test load balancer
test_load_balancer() {
    log_info "Testing load balancer..."
    
    # Test HAProxy stats
    if haproxy_stats=$(curl -s "http://localhost:11099/stats" 2>/dev/null); then
        log_success "✓ HAProxy stats accessible"
        update_result "load_balancing" "stats" "pass" "Stats page accessible"
        
        # Parse backend health
        backend_count=$(echo "$haproxy_stats" | grep -c "backend" || echo "0")
        log_info "Found $backend_count backends in HAProxy"
        update_result "load_balancing" "backends" "pass" "Found $backend_count backends"
    else
        log_error "✗ HAProxy stats not accessible"
        update_result "load_balancing" "stats" "fail" "Stats page not accessible"
    fi
    
    # Test load balancing for configured ports
    for port in 11100 11101 11102 11103 11104 11105; do
        if nc -z localhost "$port" 2>/dev/null; then
            log_success "✓ Port $port is accessible"
            update_result "load_balancing" "port-$port" "pass" "Port accessible"
        else
            log_warn "⚠ Port $port is not accessible"
            update_result "load_balancing" "port-$port" "warning" "Port not accessible (service may not be deployed)"
        fi
    done
}

# Test performance
test_performance() {
    log_info "Testing performance..."
    
    # Test response times
    local services=(
        "consul:11090:/v1/status/leader"
        "monitor:11091:/health"
    )
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r service port path <<< "$service_info"
        
        # Measure response time
        if response_time=$(curl -w '%{time_total}' -s -o /dev/null "http://localhost:$port$path" 2>/dev/null); then
            response_ms=$(echo "$response_time * 1000" | bc -l | cut -d. -f1)
            
            if [[ "$response_ms" -lt 1000 ]]; then
                log_success "✓ $service response time: ${response_ms}ms"
                update_result "performance" "$service-response" "pass" "Response time: ${response_ms}ms"
            elif [[ "$response_ms" -lt 5000 ]]; then
                log_warn "⚠ $service response time: ${response_ms}ms (slow)"
                update_result "performance" "$service-response" "warning" "Slow response time: ${response_ms}ms"
            else
                log_error "✗ $service response time: ${response_ms}ms (very slow)"
                update_result "performance" "$service-response" "fail" "Very slow response time: ${response_ms}ms"
            fi
        else
            log_error "✗ Failed to test $service response time"
            update_result "performance" "$service-response" "fail" "Failed to measure response time"
        fi
    done
}

# Test multi-client support
test_multi_client() {
    log_info "Testing multi-client support..."
    
    # Simulate concurrent requests
    local test_url="http://localhost:11091/health"
    local concurrent_requests=5
    local pids=()
    
    # Start concurrent requests
    for ((i=1; i<=concurrent_requests; i++)); do
        (
            start_time=$(date +%s.%N)
            if curl -s "$test_url" >/dev/null 2>&1; then
                end_time=$(date +%s.%N)
                duration=$(echo "$end_time - $start_time" | bc -l)
                echo "Request $i completed in ${duration}s"
            else
                echo "Request $i failed"
                exit 1
            fi
        ) &
        pids+=($!)
    done
    
    # Wait for all requests to complete
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            ((failed++))
        fi
    done
    
    if [[ "$failed" -eq 0 ]]; then
        log_success "✓ All $concurrent_requests concurrent requests succeeded"
        update_result "load_balancing" "concurrent-requests" "pass" "$concurrent_requests concurrent requests succeeded"
    else
        log_error "✗ $failed out of $concurrent_requests concurrent requests failed"
        update_result "load_balancing" "concurrent-requests" "fail" "$failed out of $concurrent_requests requests failed"
    fi
}

# Calculate final results
calculate_results() {
    log_info "Calculating final results..."
    
    local temp_file=$(mktemp)
    jq '
        .summary.total_tests = (.tests | to_entries | map(.value | to_entries | length) | add) |
        .summary.passed = (.tests | to_entries | map(.value | to_entries | map(select(.value.status == "pass")) | length) | add) |
        .summary.failed = (.tests | to_entries | map(.value | to_entries | map(select(.value.status == "fail")) | length) | add) |
        .summary.warnings = (.tests | to_entries | map(.value | to_entries | map(select(.value.status == "warning")) | length) | add) |
        .overall_status = (
            if .summary.failed > 0 then "failed"
            elif .summary.warnings > 0 then "warning" 
            else "passed"
            end
        )
    ' "$VALIDATION_RESULTS" > "$temp_file"
    mv "$temp_file" "$VALIDATION_RESULTS"
}

# Print summary
print_summary() {
    log_info "Validation Summary:"
    echo "=================================="
    
    local overall_status=$(jq -r '.overall_status' "$VALIDATION_RESULTS")
    local total_tests=$(jq -r '.summary.total_tests' "$VALIDATION_RESULTS")
    local passed=$(jq -r '.summary.passed' "$VALIDATION_RESULTS")
    local failed=$(jq -r '.summary.failed' "$VALIDATION_RESULTS")
    local warnings=$(jq -r '.summary.warnings' "$VALIDATION_RESULTS")
    
    echo "Overall Status: $overall_status"
    echo "Total Tests: $total_tests"
    echo "Passed: $passed"
    echo "Failed: $failed"
    echo "Warnings: $warnings"
    echo ""
    
    if [[ "$overall_status" == "passed" ]]; then
        log_success "🎉 All critical tests passed!"
        echo "✅ MCP network infrastructure is working correctly"
    elif [[ "$overall_status" == "warning" ]]; then
        log_warn "⚠️ Some non-critical issues detected"
        echo "🟡 MCP network infrastructure is mostly working"
    else
        log_error "❌ Critical issues detected"
        echo "🔴 MCP network infrastructure has problems"
    fi
    
    echo ""
    echo "Detailed Results: $VALIDATION_RESULTS"
    echo "Logs: $LOG_FILE"
    echo "=================================="
}

# Main validation function
main() {
    log_info "Starting MCP network validation..."
    
    # Initialize
    init_results
    
    # Check if required tools are available
    for tool in curl jq nc bc docker; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log_error "Required tool '$tool' not found"
            exit 1
        fi
    done
    
    # Run tests
    test_container_status
    test_network_connectivity
    test_service_discovery
    test_load_balancer
    test_performance
    test_multi_client
    
    # Calculate and display results
    calculate_results
    print_summary
    
    # Exit with appropriate code
    local overall_status=$(jq -r '.overall_status' "$VALIDATION_RESULTS")
    case "$overall_status" in
        "passed") exit 0 ;;
        "warning") exit 1 ;;
        "failed") exit 2 ;;
        *) exit 3 ;;
    esac
}

# Run main function
main "$@"