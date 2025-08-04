#!/bin/bash

# SutazAI Comprehensive Load Testing Execution Script
# This script orchestrates all load testing scenarios for SutazAI

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="${SCRIPT_DIR}/tests"
REPORTS_DIR="${SCRIPT_DIR}/reports"
LOGS_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create directories
mkdir -p "${REPORTS_DIR}" "${LOGS_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "${LOGS_DIR}/load_test_${TIMESTAMP}.log"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "${LOGS_DIR}/load_test_${TIMESTAMP}.log"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "${LOGS_DIR}/load_test_${TIMESTAMP}.log"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "${LOGS_DIR}/load_test_${TIMESTAMP}.log"
}

# Default configuration
DEFAULT_BASE_URL="http://localhost"
DEFAULT_DURATION="300s"
DEFAULT_VUS="50"
DEFAULT_TEST_SUITE="all"

# Parse command line arguments
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -u, --base-url URL          Base URL for testing (default: ${DEFAULT_BASE_URL})
    -d, --duration DURATION     Test duration (default: ${DEFAULT_DURATION})
    -v, --vus VUS               Virtual users (default: ${DEFAULT_VUS})
    -s, --suite SUITE           Test suite to run (default: ${DEFAULT_TEST_SUITE})
                                Options: all, agents, database, jarvis, mesh, gateway, integration, stress
    -r, --report-only           Only generate reports from existing results
    -c, --cleanup               Cleanup test data after tests
    -h, --help                  Show this help message

EXAMPLES:
    $0                                      # Run all tests with defaults
    $0 -s agents -v 100 -d 600s           # Run agent tests with 100 VUs for 10 minutes
    $0 -s stress -v 500                    # Run stress tests with 500 VUs
    $0 --base-url http://staging.sutazai.com -s integration  # Test staging environment
EOF
}

# Parse arguments
BASE_URL="${DEFAULT_BASE_URL}"
DURATION="${DEFAULT_DURATION}"
VUS="${DEFAULT_VUS}"
TEST_SUITE="${DEFAULT_TEST_SUITE}"
REPORT_ONLY=false
CLEANUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--base-url)
            BASE_URL="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -v|--vus)
            VUS="$2"
            shift 2
            ;;
        -s|--suite)
            TEST_SUITE="$2"
            shift 2
            ;;
        -r|--report-only)
            REPORT_ONLY=true
            shift
            ;;
        -c|--cleanup)
            CLEANUP=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    # Check if k6 is installed
    if ! command -v k6 &> /dev/null; then
        error "k6 is not installed. Please install k6 first."
        echo "Install with: curl https://github.com/grafana/k6/releases/download/v0.47.0/k6-v0.47.0-linux-amd64.tar.gz -L | tar xvz --strip-components 1"
        exit 1
    fi
    
    # Check if jq is installed for JSON processing
    if ! command -v jq &> /dev/null; then
        warning "jq is not installed. Report generation may be limited."
    fi
    
    # Check if curl is available for health checks
    if ! command -v curl &> /dev/null; then
        error "curl is required for health checks"
        exit 1
    fi
    
    success "Dependencies check passed"
}

# Pre-flight health check
health_check() {
    log "Performing system health check..."
    
    local health_endpoints=(
        "${BASE_URL}:8000/health"
        "${BASE_URL}:8501/"
        "${BASE_URL}:10104/api/tags"
        "${BASE_URL}:10000"
        "${BASE_URL}:10001"
    )
    
    local failed_checks=0
    
    for endpoint in "${health_endpoints[@]}"; do
        if curl -s --max-time 10 "${endpoint}" > /dev/null 2>&1; then
            success "✓ ${endpoint}"
        else
            warning "✗ ${endpoint} - Not accessible"
            ((failed_checks++))
        fi
    done
    
    if [[ ${failed_checks} -gt 2 ]]; then
        error "Too many services are unavailable. Please check system status."
        exit 1
    fi
    
    log "Health check completed with ${failed_checks} failed endpoints"
}

# Generate dynamic agent configuration
generate_agent_config() {
    log "Generating agent configuration..."
    
    local agent_config_file="${SCRIPT_DIR}/agent-endpoints.json"
    
    # Read from agent registry if available
    if [[ -f "../agents/agent_registry.json" ]]; then
        log "Using agent registry for configuration"
        # Extract agent names and generate port mappings
        jq -r '.agents | keys[]' ../agents/agent_registry.json | head -69 | awk 'BEGIN{print "{"} {printf "  \"%s\": %d%s\n", $1, 8080+NR-1, (NR==69 ? "" : ",")} END{print "}"}' > "${agent_config_file}"
    else
        warning "Agent registry not found, using default configuration"
        cat > "${agent_config_file}" << 'EOF'
{
  "ai-system-architect": 8080,
  "ai-senior-engineer": 8081,
  "ai-qa-team-lead": 8082,
  "testing-qa-validator": 8083,
  "deployment-automation-master": 8084
}
EOF
    fi
    
    success "Agent configuration generated"
}

# Run individual test suite
run_test_suite() {
    local suite_name=$1
    local test_file=$2
    local custom_options=$3
    
    log "Running ${suite_name} test suite..."
    
    local output_file="${REPORTS_DIR}/${suite_name}_${TIMESTAMP}.json"
    local summary_file="${REPORTS_DIR}/${suite_name}_${TIMESTAMP}_summary.txt"
    
    # Set environment variables for k6
    export BASE_URL="${BASE_URL}"
    export DURATION="${DURATION}"
    export VUS="${VUS}"
    
    # Run k6 test with custom options
    if k6 run \
        --out json="${output_file}" \
        --summary-export="${summary_file}" \
        --vus "${VUS}" \
        --duration "${DURATION}" \
        ${custom_options} \
        "${test_file}" 2>&1 | tee "${LOGS_DIR}/${suite_name}_${TIMESTAMP}.log"; then
        success "${suite_name} test suite completed successfully"
        return 0
    else
        error "${suite_name} test suite failed"
        return 1
    fi
}

# Main test execution
run_tests() {
    if [[ "${REPORT_ONLY}" == "true" ]]; then
        log "Skipping test execution, generating reports only"
        return 0
    fi
    
    log "Starting SutazAI load testing with configuration:"
    log "  Base URL: ${BASE_URL}"
    log "  Duration: ${DURATION}"
    log "  Virtual Users: ${VUS}"
    log "  Test Suite: ${TEST_SUITE}"
    
    local failed_suites=0
    local total_suites=0
    
    case "${TEST_SUITE}" in
        "all")
            log "Running comprehensive test suite..."
            
            # Individual component tests
            ((total_suites++))
            if run_test_suite "agent-performance" "${TESTS_DIR}/agent-performance.js"; then
                success "Agent performance tests passed"
            else
                ((failed_suites++))
            fi
            
            ((total_suites++))
            if run_test_suite "database-load" "${TESTS_DIR}/database-load.js"; then
                success "Database load tests passed"
            else
                ((failed_suites++))
            fi
            
            ((total_suites++))
            if run_test_suite "jarvis-concurrent" "${TESTS_DIR}/jarvis-concurrent.js"; then
                success "Jarvis concurrent tests passed"
            else
                ((failed_suites++))
            fi
            
            ((total_suites++))
            if run_test_suite "service-mesh-resilience" "${TESTS_DIR}/service-mesh-resilience.js"; then
                success "Service mesh resilience tests passed"
            else
                ((failed_suites++))
            fi
            
            ((total_suites++))
            if run_test_suite "api-gateway-throughput" "${TESTS_DIR}/api-gateway-throughput.js"; then
                success "API gateway throughput tests passed"
            else
                ((failed_suites++))
            fi
            
            # System integration tests
            ((total_suites++))
            if run_test_suite "system-integration" "${TESTS_DIR}/system-integration.js" "--duration ${DURATION}"; then
                success "System integration tests passed"
            else
                ((failed_suites++))
            fi
            ;;
        
        "agents")
            ((total_suites++))
            run_test_suite "agent-performance" "${TESTS_DIR}/agent-performance.js"
            [[ $? -ne 0 ]] && ((failed_suites++))
            ;;
        
        "database")
            ((total_suites++))
            run_test_suite "database-load" "${TESTS_DIR}/database-load.js"
            [[ $? -ne 0 ]] && ((failed_suites++))
            ;;
        
        "jarvis")
            ((total_suites++))
            run_test_suite "jarvis-concurrent" "${TESTS_DIR}/jarvis-concurrent.js"
            [[ $? -ne 0 ]] && ((failed_suites++))
            ;;
        
        "mesh")
            ((total_suites++))
            run_test_suite "service-mesh-resilience" "${TESTS_DIR}/service-mesh-resilience.js"
            [[ $? -ne 0 ]] && ((failed_suites++))
            ;;
        
        "gateway")
            ((total_suites++))
            run_test_suite "api-gateway-throughput" "${TESTS_DIR}/api-gateway-throughput.js"
            [[ $? -ne 0 ]] && ((failed_suites++))
            ;;
        
        "integration")
            ((total_suites++))
            run_test_suite "system-integration" "${TESTS_DIR}/system-integration.js"
            [[ $? -ne 0 ]] && ((failed_suites++))
            ;;
        
        "stress")
            log "Running stress test with increased load..."
            local stress_vus=$((VUS * 3))
            export VUS="${stress_vus}"
            
            ((total_suites++))
            run_test_suite "stress-test" "${TESTS_DIR}/system-integration.js" "--stages 2m:${VUS},5m:${stress_vus},2m:0"
            [[ $? -ne 0 ]] && ((failed_suites++))
            ;;
        
        *)
            error "Unknown test suite: ${TEST_SUITE}"
            exit 1
            ;;
    esac
    
    log "Test execution summary:"
    log "  Total suites: ${total_suites}"
    log "  Failed suites: ${failed_suites}"
    log "  Success rate: $(( (total_suites - failed_suites) * 100 / total_suites ))%"
    
    return ${failed_suites}
}

# Generate comprehensive reports
generate_reports() {
    log "Generating comprehensive test reports..."
    
    local report_file="${REPORTS_DIR}/comprehensive_report_${TIMESTAMP}.html"
    local json_report="${REPORTS_DIR}/comprehensive_report_${TIMESTAMP}.json"
    
    # Create HTML report
    cat > "${report_file}" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>SutazAI Load Testing Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .success { background: #d4edda; border-color: #c3e6cb; }
        .warning { background: #fff3cd; border-color: #ffeaa7; }
        .error { background: #f8d7da; border-color: #f5c6cb; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 3px; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>SutazAI Load Testing Report</h1>
        <p>Generated: $(date)</p>
        <p>Test Configuration: ${VUS} VUs, ${DURATION} duration, ${BASE_URL}</p>
    </div>
EOF
    
    # Process each test result if jq is available
    if command -v jq &> /dev/null; then
        for result_file in "${REPORTS_DIR}"/*_${TIMESTAMP}.json; do
            if [[ -f "${result_file}" ]]; then
                local test_name=$(basename "${result_file}" | sed "s/_${TIMESTAMP}.json//")
                
                echo "    <div class='section'>" >> "${report_file}"
                echo "        <h2>${test_name} Results</h2>" >> "${report_file}"
                
                # Extract key metrics using jq
                local avg_duration=$(jq -r '.metrics.http_req_duration.avg // "N/A"' "${result_file}")
                local p95_duration=$(jq -r '.metrics.http_req_duration.p95 // "N/A"' "${result_file}")
                local error_rate=$(jq -r '.metrics.http_req_failed.rate // "N/A"' "${result_file}")
                local total_requests=$(jq -r '.metrics.http_reqs.count // "N/A"' "${result_file}")
                
                cat >> "${report_file}" << EOF
        <div class='metric'>Avg Response Time: ${avg_duration}ms</div>
        <div class='metric'>P95 Response Time: ${p95_duration}ms</div>
        <div class='metric'>Error Rate: ${error_rate}%</div>
        <div class='metric'>Total Requests: ${total_requests}</div>
EOF
                echo "    </div>" >> "${report_file}"
            fi
        done
    else
        echo "    <div class='section warning'>" >> "${report_file}"
        echo "        <p>jq not available - detailed metrics analysis skipped</p>" >> "${report_file}"
        echo "    </div>" >> "${report_file}"
    fi
    
    echo "</body></html>" >> "${report_file}"
    
    success "HTML report generated: ${report_file}"
    
    # Generate JSON summary
    cat > "${json_report}" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "configuration": {
        "base_url": "${BASE_URL}",
        "virtual_users": ${VUS},
        "duration": "${DURATION}",
        "test_suite": "${TEST_SUITE}"
    },
    "reports_location": "${REPORTS_DIR}",
    "logs_location": "${LOGS_DIR}"
}
EOF
    
    success "JSON report generated: ${json_report}"
}

# Cleanup function
cleanup_test_data() {
    if [[ "${CLEANUP}" == "true" ]]; then
        log "Cleaning up test data..."
        
        # Clean up any test data created during load testing
        # This would typically involve database cleanup, cache clearing, etc.
        
        warning "Test data cleanup is not yet implemented"
        # TODO: Implement actual cleanup logic
    fi
}

# Signal handlers
cleanup_on_exit() {
    log "Cleaning up on exit..."
    cleanup_test_data
}

trap cleanup_on_exit EXIT

# Main execution
main() {
    log "Starting SutazAI comprehensive load testing"
    
    check_dependencies
    health_check
    generate_agent_config
    
    if run_tests; then
        success "All test suites completed successfully"
        exit_code=0
    else
        error "Some test suites failed"
        exit_code=1
    fi
    
    generate_reports
    cleanup_test_data
    
    log "Load testing completed. Check reports in: ${REPORTS_DIR}"
    
    exit ${exit_code}
}

# Run main function
main "$@"