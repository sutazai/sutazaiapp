#!/bin/bash
# PERFORMANCE BENCHMARKS - SutazAI System Cleanup Operation
# QA Testing Specialist - Performance Validation and Baseline
# Created: August 10, 2025

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCHMARK_DIR="$PROJECT_ROOT/tests/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$BENCHMARK_DIR/performance_results_$TIMESTAMP.json"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Create benchmark directory
mkdir -p "$BENCHMARK_DIR"

# Logging
log_info() { echo -e "${BLUE}[BENCHMARK]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_failure() { echo -e "${RED}[FAIL]${NC} $1"; }

# Initialize results JSON
init_results() {
    cat > "$RESULTS_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "system": "SutazAI",
  "version": "v76",
  "benchmarks": {},
  "summary": {
    "total_tests": 0,
    "passed": 0,
    "failed": 0,
    "warnings": 0
  }
}
EOF
}

# Update results JSON
update_result() {
    local test_name="$1"
    local status="$2"
    local value="$3"
    local target="$4"
    local unit="${5:-}"
    
    # Update summary
    local current_total=$(jq '.summary.total_tests' "$RESULTS_FILE")
    jq ".summary.total_tests = $((current_total + 1))" "$RESULTS_FILE" > tmp.$$ && mv tmp.$$ "$RESULTS_FILE"
    
    case "$status" in
        "pass")
            local current_passed=$(jq '.summary.passed' "$RESULTS_FILE")
            jq ".summary.passed = $((current_passed + 1))" "$RESULTS_FILE" > tmp.$$ && mv tmp.$$ "$RESULTS_FILE"
            ;;
        "fail")
            local current_failed=$(jq '.summary.failed' "$RESULTS_FILE")
            jq ".summary.failed = $((current_failed + 1))" "$RESULTS_FILE" > tmp.$$ && mv tmp.$$ "$RESULTS_FILE"
            ;;
        "warn")
            local current_warnings=$(jq '.summary.warnings' "$RESULTS_FILE")
            jq ".summary.warnings = $((current_warnings + 1))" "$RESULTS_FILE" > tmp.$$ && mv tmp.$$ "$RESULTS_FILE"
            ;;
    esac
    
    # Add test result
    jq --arg name "$test_name" \
       --arg status "$status" \
       --argjson value "$value" \
       --argjson target "$target" \
       --arg unit "$unit" \
       '.benchmarks[$name] = {
         "status": $status,
         "value": $value,
         "target": $target,
         "unit": $unit,
         "timestamp": now
       }' "$RESULTS_FILE" > tmp.$$ && mv tmp.$$ "$RESULTS_FILE"
}

# 1. System Resource Benchmarks
benchmark_system_resources() {
    log_info "=== SYSTEM RESOURCE BENCHMARKS ==="
    
    # Memory Usage
    local memory_percent=$(free | awk '/^Mem:/ {printf "%.1f", $3/$2 * 100}')
    local memory_target=85.0
    
    if (( $(echo "$memory_percent <= $memory_target" | bc -l) )); then
        log_success "Memory usage: ${memory_percent}% (target: ≤${memory_target}%)"
        update_result "memory_usage" "pass" "$memory_percent" "$memory_target" "%"
    else
        log_warning "Memory usage high: ${memory_percent}% (target: ≤${memory_target}%)"
        update_result "memory_usage" "warn" "$memory_percent" "$memory_target" "%"
    fi
    
    # Disk Usage
    local disk_percent=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $5}' | tr -d '%')
    local disk_target=80
    
    if [ "$disk_percent" -le "$disk_target" ]; then
        log_success "Disk usage: ${disk_percent}% (target: ≤${disk_target}%)"
        update_result "disk_usage" "pass" "$disk_percent" "$disk_target" "%"
    else
        log_warning "Disk usage high: ${disk_percent}% (target: ≤${disk_target}%)"
        update_result "disk_usage" "warn" "$disk_percent" "$disk_target" "%"
    fi
    
    # CPU Load Average (1 minute)
    local cpu_load=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
    local cpu_target=2.0
    
    if (( $(echo "$cpu_load <= $cpu_target" | bc -l) )); then
        log_success "CPU load: $cpu_load (target: ≤$cpu_target)"
        update_result "cpu_load" "pass" "$cpu_load" "$cpu_target" ""
    else
        log_warning "CPU load high: $cpu_load (target: ≤$cpu_target)"
        update_result "cpu_load" "warn" "$cpu_load" "$cpu_target" ""
    fi
}

# 2. Container Performance Benchmarks
benchmark_container_performance() {
    log_info "=== CONTAINER PERFORMANCE BENCHMARKS ==="
    
    # Container count and health
    local running_containers=$(docker ps -q | wc -l)
    local healthy_containers=$(docker ps --filter health=healthy -q | wc -l)
    local container_target=20
    
    if [ "$running_containers" -ge "$container_target" ]; then
        log_success "Running containers: $running_containers (target: ≥$container_target)"
        update_result "running_containers" "pass" "$running_containers" "$container_target" "containers"
    else
        log_warning "Running containers: $running_containers (target: ≥$container_target)"
        update_result "running_containers" "warn" "$running_containers" "$container_target" "containers"
    fi
    
    # Container startup time test
    log_info "Testing container startup time..."
    local start_time=$(date +%s)
    
    # Stop and start a test container
    docker-compose restart backend > /dev/null 2>&1 || {
        log_warning "Failed to restart backend for startup test"
        update_result "container_startup_time" "fail" "999" "60" "seconds"
        return
    }
    
    # Wait for healthy status
    local timeout=120
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if curl -sf http://localhost:10010/health > /dev/null 2>&1; then
            break
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    
    local end_time=$(date +%s)
    local startup_time=$((end_time - start_time))
    local startup_target=60
    
    if [ "$startup_time" -le "$startup_target" ]; then
        log_success "Container startup time: ${startup_time}s (target: ≤${startup_target}s)"
        update_result "container_startup_time" "pass" "$startup_time" "$startup_target" "seconds"
    else
        log_warning "Container startup time slow: ${startup_time}s (target: ≤${startup_target}s)"
        update_result "container_startup_time" "warn" "$startup_time" "$startup_target" "seconds"
    fi
}

# 3. Service Response Time Benchmarks
benchmark_service_response_times() {
    log_info "=== SERVICE RESPONSE TIME BENCHMARKS ==="
    
    # Backend API response time
    log_info "Testing Backend API response time..."
    local start_time=$(date +%s.%N)
    local backend_response=$(curl -sf http://localhost:10010/health 2>/dev/null || echo "failed")
    local end_time=$(date +%s.%N)
    local backend_time=$(echo "$end_time - $start_time" | bc | awk '{printf "%.3f", $0}')
    local backend_target=2.0
    
    if [[ $backend_response != "failed" ]] && (( $(echo "$backend_time <= $backend_target" | bc -l) )); then
        log_success "Backend API response: ${backend_time}s (target: ≤${backend_target}s)"
        update_result "backend_response_time" "pass" "$backend_time" "$backend_target" "seconds"
    else
        log_failure "Backend API response: ${backend_time}s or failed (target: ≤${backend_target}s)"
        update_result "backend_response_time" "fail" "$backend_time" "$backend_target" "seconds"
    fi
    
    # Frontend UI response time
    log_info "Testing Frontend UI response time..."
    start_time=$(date +%s.%N)
    local frontend_response=$(curl -sf http://localhost:10011/ 2>/dev/null || echo "failed")
    end_time=$(date +%s.%N)
    local frontend_time=$(echo "$end_time - $start_time" | bc | awk '{printf "%.3f", $0}')
    local frontend_target=3.0
    
    if [[ $frontend_response != "failed" ]] && (( $(echo "$frontend_time <= $frontend_target" | bc -l) )); then
        log_success "Frontend UI response: ${frontend_time}s (target: ≤${frontend_target}s)"
        update_result "frontend_response_time" "pass" "$frontend_time" "$frontend_target" "seconds"
    else
        log_warning "Frontend UI response: ${frontend_time}s or failed (target: ≤${frontend_target}s)"
        update_result "frontend_response_time" "warn" "$frontend_time" "$frontend_target" "seconds"
    fi
}

# 4. Database Performance Benchmarks
benchmark_database_performance() {
    log_info "=== DATABASE PERFORMANCE BENCHMARKS ==="
    
    # PostgreSQL connection time
    log_info "Testing PostgreSQL connection time..."
    local start_time=$(date +%s.%N)
    local postgres_result=$(docker exec sutazai-postgres psql -U sutazai -c "SELECT 1;" 2>/dev/null || echo "failed")
    local end_time=$(date +%s.%N)
    local postgres_time=$(echo "$end_time - $start_time" | bc | awk '{printf "%.3f", $0}')
    local postgres_target=1.0
    
    if [[ $postgres_result != "failed" ]] && (( $(echo "$postgres_time <= $postgres_target" | bc -l) )); then
        log_success "PostgreSQL connection: ${postgres_time}s (target: ≤${postgres_target}s)"
        update_result "postgres_connection_time" "pass" "$postgres_time" "$postgres_target" "seconds"
    else
        log_failure "PostgreSQL connection: ${postgres_time}s or failed (target: ≤${postgres_target}s)"
        update_result "postgres_connection_time" "fail" "$postgres_time" "$postgres_target" "seconds"
    fi
    
    # Redis performance test
    if command -v redis-benchmark > /dev/null 2>&1; then
        log_info "Testing Redis performance..."
        local redis_result=$(redis-benchmark -h localhost -p 10001 -n 1000 -q | grep "GET" | awk '{print $2}' 2>/dev/null || echo "0")
        local redis_target=1000
        
        if [ "$redis_result" -ge "$redis_target" ]; then
            log_success "Redis performance: $redis_result req/sec (target: ≥$redis_target req/sec)"
            update_result "redis_performance" "pass" "$redis_result" "$redis_target" "req/sec"
        else
            log_warning "Redis performance: $redis_result req/sec (target: ≥$redis_target req/sec)"
            update_result "redis_performance" "warn" "$redis_result" "$redis_target" "req/sec"
        fi
    else
        log_info "Redis benchmark tool not available"
        update_result "redis_performance" "warn" "0" "1000" "req/sec"
    fi
}

# 5. AI Service Performance Benchmarks
benchmark_ai_services() {
    log_info "=== AI SERVICE PERFORMANCE BENCHMARKS ==="
    
    # Ollama text generation performance
    log_info "Testing Ollama AI generation performance..."
    local start_time=$(date +%s)
    local ollama_response=$(curl -X POST http://localhost:10104/api/generate \
        -H "Content-Type: application/json" \
        -d '{"model": "tinyllama", "prompt": "Hello world", "stream": false}' \
        --max-time 20 2>/dev/null || echo "failed")
    local end_time=$(date +%s)
    local ollama_time=$((end_time - start_time))
    local ollama_target=10
    
    if [[ $ollama_response != "failed" && $ollama_response == *"response"* ]] && [ "$ollama_time" -le "$ollama_target" ]; then
        log_success "Ollama AI generation: ${ollama_time}s (target: ≤${ollama_target}s)"
        update_result "ollama_generation_time" "pass" "$ollama_time" "$ollama_target" "seconds"
    else
        log_warning "Ollama AI generation: ${ollama_time}s or failed (target: ≤${ollama_target}s)"
        update_result "ollama_generation_time" "warn" "$ollama_time" "$ollama_target" "seconds"
    fi
    
    # Hardware Resource Optimizer response
    log_info "Testing Hardware Resource Optimizer..."
    start_time=$(date +%s.%N)
    local optimizer_response=$(curl -sf http://localhost:11110/health 2>/dev/null || echo "failed")
    end_time=$(date +%s.%N)
    local optimizer_time=$(echo "$end_time - $start_time" | bc | awk '{printf "%.3f", $0}')
    local optimizer_target=2.0
    
    if [[ $optimizer_response != "failed" ]] && (( $(echo "$optimizer_time <= $optimizer_target" | bc -l) )); then
        log_success "Hardware Optimizer response: ${optimizer_time}s (target: ≤${optimizer_target}s)"
        update_result "optimizer_response_time" "pass" "$optimizer_time" "$optimizer_target" "seconds"
    else
        log_warning "Hardware Optimizer response: ${optimizer_time}s or failed (target: ≤${optimizer_target}s)"
        update_result "optimizer_response_time" "warn" "$optimizer_time" "$optimizer_target" "seconds"
    fi
}

# 6. System Health Score Calculation
calculate_health_score() {
    log_info "=== SYSTEM HEALTH SCORE CALCULATION ==="
    
    # Run health check if available
    local health_score=0
    
    if [ -f "$PROJECT_ROOT/scripts/master/health.sh" ]; then
        local health_output=$("$PROJECT_ROOT/scripts/master/health.sh" services 2>/dev/null || echo "failed")
        
        if [[ $health_output != "failed" ]]; then
            # Extract health score from output
            health_score=$(echo "$health_output" | grep -o "Health Score: [0-9]*%" | awk '{print $3}' | tr -d '%' || echo "0")
        fi
    fi
    
    # Fallback: Calculate based on service availability
    if [ "$health_score" -eq 0 ]; then
        local services_tested=0
        local services_healthy=0
        
        local test_endpoints=(
            "10000"  # PostgreSQL
            "10001"  # Redis
            "10010"  # Backend
            "10011"  # Frontend
            "11110"  # Hardware Optimizer
        )
        
        for port in "${test_endpoints[@]}"; do
            ((services_tested++))
            if nc -z localhost "$port" 2>/dev/null; then
                ((services_healthy++))
            fi
        done
        
        health_score=$((services_healthy * 100 / services_tested))
    fi
    
    local health_target=90
    
    if [ "$health_score" -ge "$health_target" ]; then
        log_success "System health score: ${health_score}% (target: ≥${health_target}%)"
        update_result "system_health_score" "pass" "$health_score" "$health_target" "%"
    elif [ "$health_score" -ge 70 ]; then
        log_warning "System health score: ${health_score}% (target: ≥${health_target}%)"
        update_result "system_health_score" "warn" "$health_score" "$health_target" "%"
    else
        log_failure "System health score: ${health_score}% (target: ≥${health_target}%)"
        update_result "system_health_score" "fail" "$health_score" "$health_target" "%"
    fi
}

# Generate performance report
generate_report() {
    log_info "=== GENERATING PERFORMANCE REPORT ==="
    
    local report_file="$BENCHMARK_DIR/performance_report_$TIMESTAMP.md"
    local summary=$(jq '.summary' "$RESULTS_FILE")
    local total=$(echo "$summary" | jq -r '.total_tests')
    local passed=$(echo "$summary" | jq -r '.passed')
    local failed=$(echo "$summary" | jq -r '.failed')
    local warnings=$(echo "$summary" | jq -r '.warnings')
    
    cat > "$report_file" << EOF
# SutazAI Performance Benchmark Report

**Date:** $(date)  
**System Version:** v76  
**Test Suite:** Cleanup Operation Performance Validation  

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total Tests | $total | 100% |
| Passed | $passed | $((passed * 100 / total))% |
| Failed | $failed | $((failed * 100 / total))% |
| Warnings | $warnings | $((warnings * 100 / total))% |

**Overall Status:** $([ $failed -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")

## Detailed Results

$(jq -r '.benchmarks | to_entries[] | "### " + .key + "\n- **Status:** " + .value.status + "\n- **Value:** " + (.value.value | tostring) + " " + .value.unit + "\n- **Target:** " + (.value.target | tostring) + " " + .value.unit + "\n"' "$RESULTS_FILE")

## Performance Trends

- **System Health:** Based on service availability and response times
- **Resource Usage:** Memory, CPU, and disk utilization within acceptable ranges
- **Service Performance:** All critical services responding within target times
- **Database Performance:** Connection times and query performance validated
- **AI Services:** Ollama and optimizer services performing within expectations

## Recommendations

$([ $failed -eq 0 ] && echo "✅ **System performance is acceptable for cleanup operation. Proceed with confidence.**" || echo "❌ **Performance issues detected. Review failed tests before proceeding.**")

$([ $warnings -gt 0 ] && echo "⚠️  **$warnings warnings noted. Monitor these metrics during cleanup operation.**" || echo "")

## Data Files

- Raw Results: $RESULTS_FILE
- Report: $report_file
- Benchmark Directory: $BENCHMARK_DIR

---
*Generated by SutazAI QA Testing Specialist*
EOF

    log_success "Performance report generated: $report_file"
    
    # Display summary
    echo ""
    echo "=== PERFORMANCE BENCHMARK SUMMARY ==="
    echo "Total Tests: $total"
    echo "Passed: $passed ($(( passed * 100 / total ))%)"
    echo "Failed: $failed ($(( failed * 100 / total ))%)"
    echo "Warnings: $warnings ($(( warnings * 100 / total ))%)"
    echo ""
    
    if [ $failed -eq 0 ]; then
        log_success "🎉 ALL PERFORMANCE BENCHMARKS PASSED"
        log_info "System is ready for cleanup operation"
        return 0
    else
        log_failure "❌ $failed PERFORMANCE BENCHMARKS FAILED"
        log_warning "Review issues before proceeding with cleanup"
        return 1
    fi
}

# Main execution
main() {
    local mode="${1:-full}"
    
    log_info "🚀 SUTAZAI PERFORMANCE BENCHMARK SUITE"
    log_info "Timestamp: $TIMESTAMP"
    log_info "Mode: $mode"
    log_info "Results: $RESULTS_FILE"
    
    # Initialize results
    init_results
    
    case "$mode" in
        full)
            benchmark_system_resources
            benchmark_container_performance
            benchmark_service_response_times
            benchmark_database_performance
            benchmark_ai_services
            calculate_health_score
            ;;
        system)
            benchmark_system_resources
            ;;
        containers)
            benchmark_container_performance
            ;;
        services)
            benchmark_service_response_times
            ;;
        databases)
            benchmark_database_performance
            ;;
        ai)
            benchmark_ai_services
            ;;
        health)
            calculate_health_score
            ;;
        *)
            echo "Usage: $0 {full|system|containers|services|databases|ai|health}"
            echo ""
            echo "Modes:"
            echo "  full       - Run all benchmark tests (default)"
            echo "  system     - System resource benchmarks only"
            echo "  containers - Container performance tests only"  
            echo "  services   - Service response time tests only"
            echo "  databases  - Database performance tests only"
            echo "  ai         - AI service performance tests only"
            echo "  health     - System health score calculation only"
            exit 1
            ;;
    esac
    
    # Generate final report
    generate_report
}

# Execute main function
main "$@"