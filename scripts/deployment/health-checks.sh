#!/bin/bash

# ============================================================================
# Comprehensive Health Check Suite for Perfect Jarvis Blue/Green Deployment
# ============================================================================
#
# This script performs comprehensive health checks for blue/green environments:
# - Service availability checks
# - API endpoint validation
# - Database connectivity
# - Model loading verification
# - Response time validation
#
# Following CLAUDE.md rules:
# - Rule 1: No fantasy elements - only production-ready implementations
# - Rule 2: Don't break existing functionality
# - Rule 16: Use local LLMs via Ollama with TinyLlama
#
# Usage:
#   ./health-checks.sh --environment blue|green
#   ./health-checks.sh --quick
#   ./health-checks.sh --all
#   ./health-checks.sh --verbose
#
# Exit codes:
#   0 - All checks passed
#   1 - Critical check failed
#   2 - Warning issues found
# ============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/health-checks-$(date +%Y%m%d_%H%M%S).log"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Health check configuration
TIMEOUT=30
MAX_RETRIES=3
RESPONSE_TIME_THRESHOLD=5000  # milliseconds
CRITICAL_SERVICES=("backend" "postgres" "redis" "ollama")
WARNING_THRESHOLD=2000  # milliseconds

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=""
QUICK_CHECK=false
ALL_ENVIRONMENTS=false
VERBOSE=false
PARALLEL_CHECKS=true

# Results tracking
declare -A CHECK_RESULTS
declare -A CHECK_TIMES
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Color based on log level
    case "$level" in
        "INFO")  echo -e "${GREEN}[${timestamp}] [INFO]${NC} ${message}" ;;
        "WARN")  echo -e "${YELLOW}[${timestamp}] [WARN]${NC} ${message}" ;;
        "ERROR") echo -e "${RED}[${timestamp}] [ERROR]${NC} ${message}" ;;
        "DEBUG") [[ "$VERBOSE" == "true" ]] && echo -e "${BLUE}[${timestamp}] [DEBUG]${NC} ${message}" ;;
        "PASS")  echo -e "${GREEN}[${timestamp}] [PASS]${NC} ${message}" ;;
        "FAIL")  echo -e "${RED}[${timestamp}] [FAIL]${NC} ${message}" ;;
        *)       echo -e "[${timestamp}] [${level}] ${message}" ;;
    esac
    
    # Also log to file
    echo "[${timestamp}] [${level}] ${message}" >> "${LOG_FILE}"
}

record_check_result() {
    local check_name="$1"
    local status="$2"  # PASS, FAIL, WARN
    local duration="$3"
    local message="${4:-}"
    
    CHECK_RESULTS["$check_name"]="$status"
    CHECK_TIMES["$check_name"]="$duration"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    case "$status" in
        "PASS") PASSED_CHECKS=$((PASSED_CHECKS + 1)) ;;
        "FAIL") FAILED_CHECKS=$((FAILED_CHECKS + 1)) ;;
        "WARN") WARNING_CHECKS=$((WARNING_CHECKS + 1)) ;;
    esac
    
    log "$status" "${check_name}: ${message} (${duration}ms)"
}

measure_time() {
    local start_time=$(date +%s%N)
    "$@"
    local end_time=$(date +%s%N)
    local duration=$(( (end_time - start_time) / 1000000 ))  # Convert to milliseconds
    echo "$duration"
}

check_command_exists() {
    local cmd="$1"
    if ! command -v "$cmd" &> /dev/null; then
        log "ERROR" "Required command not found: $cmd"
        return 1
    fi
    return 0
}

# ============================================================================
# HEALTH CHECK FUNCTIONS
# ============================================================================

check_http_endpoint() {
    local name="$1"
    local url="$2"
    local expected_status="${3:-200}"
    local expected_content="${4:-}"
    
    log "DEBUG" "Checking HTTP endpoint: $url"
    
    local start_time=$(date +%s%N)
    local status_code
    local response_body
    local curl_exit_code=0
    
    # Perform HTTP request with timeout and retries
    for attempt in $(seq 1 $MAX_RETRIES); do
        if response=$(curl -s -w "%{http_code}" -m $TIMEOUT "$url" 2>/dev/null); then
            status_code="${response: -3}"
            response_body="${response%???}"
            break
        else
            curl_exit_code=$?
            if [[ $attempt -eq $MAX_RETRIES ]]; then
                local end_time=$(date +%s%N)
                local duration=$(( (end_time - start_time) / 1000000 ))
                record_check_result "$name" "FAIL" "$duration" "Connection failed (curl exit code: $curl_exit_code)"
                return 1
            fi
            log "DEBUG" "Attempt $attempt failed, retrying..."
            sleep 2
        fi
    done
    
    local end_time=$(date +%s%N)
    local duration=$(( (end_time - start_time) / 1000000 ))
    
    # Check status code
    if [[ "$status_code" != "$expected_status" ]]; then
        record_check_result "$name" "FAIL" "$duration" "HTTP $status_code (expected $expected_status)"
        return 1
    fi
    
    # Check response content if specified
    if [[ -n "$expected_content" ]] && [[ ! "$response_body" =~ $expected_content ]]; then
        record_check_result "$name" "FAIL" "$duration" "Response content mismatch"
        return 1
    fi
    
    # Check response time threshold
    local status="PASS"
    local message="HTTP $status_code"
    
    if [[ $duration -gt $RESPONSE_TIME_THRESHOLD ]]; then
        status="FAIL"
        message="$message (too slow: ${duration}ms > ${RESPONSE_TIME_THRESHOLD}ms)"
    elif [[ $duration -gt $WARNING_THRESHOLD ]]; then
        status="WARN"
        message="$message (slow: ${duration}ms > ${WARNING_THRESHOLD}ms)"
    fi
    
    record_check_result "$name" "$status" "$duration" "$message"
    return 0
}

check_port_connectivity() {
    local name="$1"
    local host="$2"
    local port="$3"
    
    log "DEBUG" "Checking port connectivity: $host:$port"
    
    local start_time=$(date +%s%N)
    
    if timeout $TIMEOUT bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
        local end_time=$(date +%s%N)
        local duration=$(( (end_time - start_time) / 1000000 ))
        
        local status="PASS"
        local message="Port accessible"
        
        if [[ $duration -gt $WARNING_THRESHOLD ]]; then
            status="WARN"
            message="$message (slow connection: ${duration}ms)"
        fi
        
        record_check_result "$name" "$status" "$duration" "$message"
        return 0
    else
        local end_time=$(date +%s%N)
        local duration=$(( (end_time - start_time) / 1000000 ))
        record_check_result "$name" "FAIL" "$duration" "Port not accessible"
        return 1
    fi
}

check_docker_service() {
    local name="$1"
    local container_name="$2"
    
    log "DEBUG" "Checking Docker service: $container_name"
    
    local start_time=$(date +%s%N)
    
    # Check if container is running
    if ! docker ps --format "table {{.Names}}" | grep -q "^${container_name}$"; then
        local end_time=$(date +%s%N)
        local duration=$(( (end_time - start_time) / 1000000 ))
        record_check_result "$name" "FAIL" "$duration" "Container not running"
        return 1
    fi
    
    # Check container health if health check is defined
    local health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "no-health-check")
    
    local end_time=$(date +%s%N)
    local duration=$(( (end_time - start_time) / 1000000 ))
    
    if [[ "$health_status" == "healthy" ]] || [[ "$health_status" == "no-health-check" ]]; then
        local message="Container running"
        [[ "$health_status" != "no-health-check" ]] && message="$message (health: $health_status)"
        record_check_result "$name" "PASS" "$duration" "$message"
        return 0
    else
        record_check_result "$name" "FAIL" "$duration" "Container unhealthy (health: $health_status)"
        return 1
    fi
}

check_database_connectivity() {
    local name="$1"
    local color="${2:-shared}"
    
    log "DEBUG" "Checking database connectivity"
    
    local start_time=$(date +%s%N)
    
    # Test PostgreSQL connectivity
    if docker exec sutazai-postgres pg_isready -U sutazai &>/dev/null; then
        local end_time=$(date +%s%N)
        local duration=$(( (end_time - start_time) / 1000000 ))
        record_check_result "$name" "PASS" "$duration" "PostgreSQL ready"
        return 0
    else
        local end_time=$(date +%s%N)
        local duration=$(( (end_time - start_time) / 1000000 ))
        record_check_result "$name" "FAIL" "$duration" "PostgreSQL not ready"
        return 1
    fi
}

check_redis_connectivity() {
    local name="$1"
    local color="${2:-shared}"
    
    log "DEBUG" "Checking Redis connectivity"
    
    local start_time=$(date +%s%N)
    
    # Test Redis connectivity
    if docker exec sutazai-redis redis-cli ping | grep -q "PONG"; then
        local end_time=$(date +%s%N)
        local duration=$(( (end_time - start_time) / 1000000 ))
        record_check_result "$name" "PASS" "$duration" "Redis responding"
        return 0
    else
        local end_time=$(date +%s%N)
        local duration=$(( (end_time - start_time) / 1000000 ))
        record_check_result "$name" "FAIL" "$duration" "Redis not responding"
        return 1
    fi
}

check_ollama_model() {
    local name="$1"
    local expected_model="${2:-tinyllama}"
    
    log "DEBUG" "Checking Ollama model: $expected_model"
    
    local start_time=$(date +%s%N)
    
    # Check if Ollama has the expected model loaded
    local models_response
    if models_response=$(curl -s -m $TIMEOUT "http://localhost:10104/api/tags" 2>/dev/null); then
        if echo "$models_response" | grep -q "$expected_model"; then
            local end_time=$(date +%s%N)
            local duration=$(( (end_time - start_time) / 1000000 ))
            record_check_result "$name" "PASS" "$duration" "Model $expected_model available"
            return 0
        else
            local end_time=$(date +%s%N)
            local duration=$(( (end_time - start_time) / 1000000 ))
            record_check_result "$name" "WARN" "$duration" "Model $expected_model not found"
            return 0  # Don't fail deployment for missing model
        fi
    else
        local end_time=$(date +%s%N)
        local duration=$(( (end_time - start_time) / 1000000 ))
        record_check_result "$name" "FAIL" "$duration" "Ollama API not responding"
        return 1
    fi
}

check_api_endpoints() {
    local color="$1"
    
    log "INFO" "Checking API endpoints for $color environment"
    
    # Determine ports based on environment
    local api_port
    local frontend_port
    
    if [[ "$color" == "blue" ]]; then
        api_port="21010"
        frontend_port="21010"  # Same port for direct access
    elif [[ "$color" == "green" ]]; then
        api_port="21011"
        frontend_port="21011"  # Same port for direct access
    else
        log "ERROR" "Invalid environment color: $color"
        return 1
    fi
    
    # Core API endpoints
    check_http_endpoint "${color}-api-health" "http://localhost:${api_port}/health" "200" "healthy|ok"
    
    # Test API endpoints (if backend supports them)
    check_http_endpoint "${color}-api-agents" "http://localhost:${api_port}/api/v1/agents" "200" ""
    
    # Test model integration
    check_http_endpoint "${color}-api-models" "http://localhost:${api_port}/api/v1/models" "200" ""
}

check_shared_services() {
    log "INFO" "Checking shared services"
    
    # Database services
    check_docker_service "postgres-container" "sutazai-postgres"
    check_database_connectivity "postgres-connection"
    
    check_docker_service "redis-container" "sutazai-redis"
    check_redis_connectivity "redis-connection"
    
    check_docker_service "neo4j-container" "sutazai-neo4j"
    check_port_connectivity "neo4j-browser" "localhost" "10002"
    check_port_connectivity "neo4j-bolt" "localhost" "10003"
    
    # Ollama service
    check_docker_service "ollama-container" "sutazai-ollama"
    check_port_connectivity "ollama-api" "localhost" "10104"
    check_ollama_model "ollama-model" "tinyllama"
    
    # Vector databases
    check_docker_service "chromadb-container" "sutazai-chromadb"
    check_port_connectivity "chromadb-api" "localhost" "10100"
    
    check_docker_service "qdrant-container" "sutazai-qdrant"
    check_port_connectivity "qdrant-api" "localhost" "10101"
    
    check_docker_service "faiss-container" "sutazai-faiss"
    check_port_connectivity "faiss-api" "localhost" "10103"
    
    # Monitoring services
    check_docker_service "prometheus-container" "sutazai-prometheus"
    check_http_endpoint "prometheus-api" "http://localhost:10200/-/healthy" "200"
    
    check_docker_service "grafana-container" "sutazai-grafana"
    check_http_endpoint "grafana-api" "http://localhost:10201/api/health" "200"
}

check_environment_services() {
    local color="$1"
    
    log "INFO" "Checking $color environment services"
    
    # Core application services
    check_docker_service "${color}-backend-container" "sutazai-${color}-backend"
    check_docker_service "${color}-frontend-container" "sutazai-${color}-frontend"
    
    # Check specific agent services if they exist
    if docker ps --format "table {{.Names}}" | grep -q "sutazai-${color}-jarvis-voice-interface"; then
        check_docker_service "${color}-jarvis-voice-container" "sutazai-${color}-jarvis-voice-interface"
    fi
    
    # API endpoint checks
    check_api_endpoints "$color"
}

# ============================================================================
# PERFORMANCE AND LOAD TESTING
# ============================================================================

performance_test() {
    local color="$1"
    local api_port
    
    if [[ "$color" == "blue" ]]; then
        api_port="21010"
    else
        api_port="21011"
    fi
    
    log "INFO" "Running performance test for $color environment"
    
    # Simple performance test - multiple concurrent requests
    local concurrent_requests=5
    local requests_per_client=10
    local total_requests=$((concurrent_requests * requests_per_client))
    
    log "DEBUG" "Sending $total_requests requests with $concurrent_requests concurrent clients"
    
    local start_time=$(date +%s%N)
    local temp_dir=$(mktemp -d)
    
    # Launch concurrent requests
    for i in $(seq 1 $concurrent_requests); do
        {
            for j in $(seq 1 $requests_per_client); do
                curl -s -w "%{time_total},%{http_code}\n" -o /dev/null \
                    "http://localhost:${api_port}/health" || echo "0,0"
            done > "${temp_dir}/client_${i}.txt"
        } &
    done
    
    # Wait for all clients to complete
    wait
    
    local end_time=$(date +%s%N)
    local total_duration=$(( (end_time - start_time) / 1000000 ))
    
    # Analyze results
    local successful_requests=0
    local total_response_time=0
    local max_response_time=0
    
    for file in "${temp_dir}"/client_*.txt; do
        while IFS=',' read -r time_total http_code; do
            if [[ "$http_code" == "200" ]]; then
                successful_requests=$((successful_requests + 1))
                local time_ms=$(echo "$time_total * 1000" | bc -l 2>/dev/null | cut -d. -f1)
                total_response_time=$((total_response_time + time_ms))
                if [[ $time_ms -gt $max_response_time ]]; then
                    max_response_time=$time_ms
                fi
            fi
        done < "$file"
    done
    
    # Clean up
    rm -rf "$temp_dir"
    
    # Calculate metrics
    local success_rate=$(( (successful_requests * 100) / total_requests ))
    local avg_response_time=0
    if [[ $successful_requests -gt 0 ]]; then
        avg_response_time=$((total_response_time / successful_requests))
    fi
    local requests_per_second=$(( (successful_requests * 1000) / total_duration ))
    
    # Determine status
    local status="PASS"
    local message="$success_rate% success, avg ${avg_response_time}ms, ${requests_per_second} req/s"
    
    if [[ $success_rate -lt 95 ]]; then
        status="FAIL"
        message="$message (low success rate)"
    elif [[ $avg_response_time -gt $RESPONSE_TIME_THRESHOLD ]]; then
        status="FAIL"
        message="$message (high average response time)"
    elif [[ $avg_response_time -gt $WARNING_THRESHOLD ]]; then
        status="WARN"
        message="$message (elevated response time)"
    fi
    
    record_check_result "${color}-performance" "$status" "$total_duration" "$message"
}

# ============================================================================
# MAIN CHECK ORCHESTRATION
# ============================================================================

run_health_checks() {
    local environment="$1"
    
    log "INFO" "Starting health checks for $environment environment"
    log "INFO" "Log file: $LOG_FILE"
    
    # Prerequisites check
    log "INFO" "Checking prerequisites"
    check_command_exists "docker" || exit 1
    check_command_exists "curl" || exit 1
    check_command_exists "timeout" || exit 1
    
    if [[ "$environment" == "shared" ]] || [[ "$environment" == "all" ]]; then
        # Check shared services
        check_shared_services
    fi
    
    if [[ "$environment" == "blue" ]] || [[ "$environment" == "all" ]]; then
        # Check blue environment
        check_environment_services "blue"
        
        if [[ "$QUICK_CHECK" == "false" ]]; then
            performance_test "blue"
        fi
    fi
    
    if [[ "$environment" == "green" ]] || [[ "$environment" == "all" ]]; then
        # Check green environment
        check_environment_services "green"
        
        if [[ "$QUICK_CHECK" == "false" ]]; then
            performance_test "green"
        fi
    fi
}

print_summary() {
    log "INFO" "=== Health Check Summary ==="
    log "INFO" "Total checks: $TOTAL_CHECKS"
    log "INFO" "Passed: $PASSED_CHECKS"
    log "INFO" "Failed: $FAILED_CHECKS"
    log "INFO" "Warnings: $WARNING_CHECKS"
    
    if [[ $FAILED_CHECKS -gt 0 ]]; then
        log "ERROR" "Health checks FAILED"
        log "INFO" "Failed checks:"
        for check_name in "${!CHECK_RESULTS[@]}"; do
            if [[ "${CHECK_RESULTS[$check_name]}" == "FAIL" ]]; then
                log "ERROR" "  - $check_name (${CHECK_TIMES[$check_name]}ms)"
            fi
        done
        return 1
    elif [[ $WARNING_CHECKS -gt 0 ]]; then
        log "WARN" "Health checks completed with WARNINGS"
        log "INFO" "Warning checks:"
        for check_name in "${!CHECK_RESULTS[@]}"; do
            if [[ "${CHECK_RESULTS[$check_name]}" == "WARN" ]]; then
                log "WARN" "  - $check_name (${CHECK_TIMES[$check_name]}ms)"
            fi
        done
        return 2
    else
        log "PASS" "All health checks PASSED"
        return 0
    fi
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

show_help() {
    cat << EOF
Comprehensive Health Check Suite for Perfect Jarvis Blue/Green Deployment

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --environment ENV      Check specific environment (blue|green|shared|all)
    --quick               Skip performance tests
    --verbose             Enable verbose logging
    --timeout SECONDS     Set timeout for individual checks [default: 30]
    --help                Show this help message

EXAMPLES:
    # Check blue environment
    $0 --environment blue
    
    # Quick check of all environments
    $0 --environment all --quick
    
    # Verbose check of shared services
    $0 --environment shared --verbose
    
    # Check with custom timeout
    $0 --environment green --timeout 60

EXIT CODES:
    0 - All checks passed
    1 - Critical check failed
    2 - Warning issues found

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --quick)
            QUICK_CHECK=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Validate environment parameter
if [[ -z "$ENVIRONMENT" ]]; then
    log "ERROR" "Environment parameter is required. Use --environment (blue|green|shared|all)"
    exit 1
fi

if [[ "$ENVIRONMENT" != "blue" && "$ENVIRONMENT" != "green" && "$ENVIRONMENT" != "shared" && "$ENVIRONMENT" != "all" ]]; then
    log "ERROR" "Invalid environment: $ENVIRONMENT. Must be 'blue', 'green', 'shared', or 'all'"
    exit 1
fi

# Run health checks
log "INFO" "Perfect Jarvis Health Check Suite starting..."
log "INFO" "Environment: $ENVIRONMENT"
log "INFO" "Quick check: $QUICK_CHECK"
log "INFO" "Timeout: ${TIMEOUT}s"

run_health_checks "$ENVIRONMENT"

# Print summary and exit with appropriate code
print_summary
exit_code=$?

log "INFO" "Health check suite completed with exit code: $exit_code"
log "INFO" "Detailed log saved to: $LOG_FILE"

exit $exit_code