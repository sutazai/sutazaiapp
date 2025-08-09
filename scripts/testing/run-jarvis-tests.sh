#!/bin/bash

# Comprehensive Jarvis Testing Suite Runner
# File: scripts/run-jarvis-tests.sh

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$PROJECT_DIR/docs"
TESTING_DIR="$DOCS_DIR/testing"
RESULTS_DIR="$PROJECT_DIR/test-results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
RUN_API_TESTS=true
RUN_E2E_TESTS=false  # Requires GUI/X11
RUN_LOAD_TESTS=false # Only on demand
SKIP_HEALTH_CHECK=false
CLEANUP_AFTER=true
WAIT_TIME=60
VERBOSE=false

# Service URLs
BASE_URL="http://localhost:10010"
FRONTEND_URL="http://localhost:10011"
JARVIS_VOICE_URL="http://localhost:11150"
JARVIS_KNOWLEDGE_URL="http://localhost:11101"
JARVIS_AUTOMATION_URL="http://localhost:11102"
JARVIS_MULTIMODAL_URL="http://localhost:11103"
JARVIS_HARDWARE_URL="http://localhost:11104"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1"
    fi
}

# Help function
show_help() {
    cat << EOF
Jarvis Testing Suite Runner

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -a, --api-only          Run only API tests (default)
    -e, --e2e-only          Run only E2E tests (requires GUI)
    -l, --load-only         Run only load tests
    -f, --full              Run all test suites
    --skip-health-check     Skip initial health checks
    --no-cleanup           Don't clean up after tests
    --wait-time <seconds>   Service startup wait time (default: 60)
    --verbose              Enable verbose logging
    
    # Service URLs (override defaults)
    --backend-url <url>     Backend API URL (default: $BASE_URL)
    --frontend-url <url>    Frontend URL (default: $FRONTEND_URL)
    --voice-url <url>       Voice service URL (default: $JARVIS_VOICE_URL)
    
Environment Variables:
    BASE_URL               Backend API URL
    FRONTEND_URL           Frontend URL  
    JARVIS_VOICE_URL       Voice service URL
    JARVIS_KNOWLEDGE_URL   Knowledge service URL
    JARVIS_AUTOMATION_URL  Automation service URL
    JARVIS_MULTIMODAL_URL  Multimodal service URL
    JARVIS_HARDWARE_URL    Hardware optimizer URL
    SKIP_HEALTH_CHECK      Skip health checks (true/false)
    TEST_WAIT_TIME         Service wait time in seconds
    CLEANUP_AFTER          Cleanup after tests (true/false)

Examples:
    $0                     # Run API tests only
    $0 --full              # Run all test suites
    $0 --api-only --verbose # Run API tests with verbose output
    $0 --load-only --backend-url http://staging.example.com
    $0 --e2e-only --no-cleanup
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -a|--api-only)
                RUN_API_TESTS=true
                RUN_E2E_TESTS=false
                RUN_LOAD_TESTS=false
                shift
                ;;
            -e|--e2e-only)
                RUN_API_TESTS=false
                RUN_E2E_TESTS=true
                RUN_LOAD_TESTS=false
                shift
                ;;
            -l|--load-only)
                RUN_API_TESTS=false
                RUN_E2E_TESTS=false
                RUN_LOAD_TESTS=true
                shift
                ;;
            -f|--full)
                RUN_API_TESTS=true
                RUN_E2E_TESTS=true
                RUN_LOAD_TESTS=true
                shift
                ;;
            --skip-health-check)
                SKIP_HEALTH_CHECK=true
                shift
                ;;
            --no-cleanup)
                CLEANUP_AFTER=false
                shift
                ;;
            --wait-time)
                WAIT_TIME="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --backend-url)
                BASE_URL="$2"
                shift 2
                ;;
            --frontend-url)
                FRONTEND_URL="$2"
                shift 2
                ;;
            --voice-url)
                JARVIS_VOICE_URL="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if docker-compose is available
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if required test files exist
    local required_files=(
        "$TESTING_DIR/postman_collection_jarvis_endpoints.json"
        "$TESTING_DIR/newman_ci_integration.js"
    )
    
    if [ "$RUN_E2E_TESTS" = true ]; then
        required_files+=("$TESTING_DIR/cypress_e2e_tests.js")
    fi
    
    if [ "$RUN_LOAD_TESTS" = true ]; then
        required_files+=("$TESTING_DIR/k6_load_tests.js")
    fi
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Required test file not found: $file"
            exit 1
        fi
    done
    
    # Check Newman installation for API tests
    if [ "$RUN_API_TESTS" = true ]; then
        if ! command -v newman &> /dev/null; then
            log_warning "Newman not found, attempting to install..."
            if command -v npm &> /dev/null; then
                npm install -g newman newman-reporter-html
            else
                log_error "npm not found, please install Newman manually: npm install -g newman"
                exit 1
            fi
        fi
    fi
    
    # Check Cypress for E2E tests
    if [ "$RUN_E2E_TESTS" = true ]; then
        if [ ! -f "$PROJECT_DIR/node_modules/.bin/cypress" ] && ! command -v cypress &> /dev/null; then
            log_warning "Cypress not found, attempting to install..."
            if command -v npm &> /dev/null; then
                cd "$PROJECT_DIR" && npm install cypress --save-dev
            else
                log_error "npm not found, please install Cypress manually"
                exit 1
            fi
        fi
    fi
    
    # Check K6 for load tests
    if [ "$RUN_LOAD_TESTS" = true ]; then
        if ! command -v k6 &> /dev/null; then
            log_error "K6 not found. Please install K6: https://k6.io/docs/get-started/installation/"
            exit 1
        fi
    fi
    
    log_success "Prerequisites check completed"
}

# Start services
start_services() {
    log_info "Starting SutazAI services..."
    
    # Ensure Docker network exists
    docker network create sutazai-network 2>/dev/null || true
    
    # Start services
    cd "$PROJECT_DIR"
    docker-compose up -d
    
    log_info "Waiting $WAIT_TIME seconds for services to start..."
    sleep "$WAIT_TIME"
    
    log_success "Services started"
}

# Health check function
perform_health_checks() {
    if [ "$SKIP_HEALTH_CHECK" = true ]; then
        log_info "Skipping health checks"
        return 0
    fi
    
    log_info "Performing health checks..."
    
    local services=(
        "$BASE_URL/health:Backend API"
        "$JARVIS_VOICE_URL/health:Jarvis Voice Interface"
        "$JARVIS_KNOWLEDGE_URL/health:Jarvis Knowledge Management"
        "$JARVIS_AUTOMATION_URL/health:Jarvis Automation Agent"
        "$JARVIS_MULTIMODAL_URL/health:Jarvis Multimodal AI"
        "$JARVIS_HARDWARE_URL/health:Jarvis Hardware Optimizer"
    )
    
    local failed_checks=0
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r url name <<< "$service_info"
        log_verbose "Checking $name at $url"
        
        if curl -sf "$url" >/dev/null 2>&1; then
            log_success "✓ $name is healthy"
        else
            log_warning "✗ $name health check failed"
            ((failed_checks++))
        fi
    done
    
    if [ $failed_checks -gt 0 ]; then
        log_warning "$failed_checks services failed health checks"
        log_warning "Continuing with tests, but some may fail..."
    else
        log_success "All services passed health checks"
    fi
}

# Run API tests
run_api_tests() {
    log_info "Running API tests with Newman..."
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Set environment variables
    export BASE_URL
    export JARVIS_VOICE_URL
    export JARVIS_KNOWLEDGE_URL
    export JARVIS_AUTOMATION_URL
    export JARVIS_MULTIMODAL_URL
    export JARVIS_HARDWARE_URL
    
    # Run Newman tests
    cd "$PROJECT_DIR"
    if node "$TESTING_DIR/newman_ci_integration.js"; then
        log_success "API tests completed successfully"
        return 0
    else
        log_error "API tests failed"
        return 1
    fi
}

# Run E2E tests
run_e2e_tests() {
    log_info "Running E2E tests with Cypress..."
    
    # Check if GUI is available
    if [ -z "$DISPLAY" ] && [ "$CI" != "true" ]; then
        log_warning "No DISPLAY variable set, running in headless mode"
    fi
    
    cd "$PROJECT_DIR"
    
    # Copy test file to Cypress directory structure
    mkdir -p cypress/e2e
    cp "$TESTING_DIR/cypress_e2e_tests.js" cypress/e2e/jarvis_interface.cy.js
    
    # Run Cypress tests
    if npx cypress run --spec "cypress/e2e/jarvis_interface.cy.js" --config baseUrl="$FRONTEND_URL"; then
        log_success "E2E tests completed successfully"
        return 0
    else
        log_error "E2E tests failed"
        return 1
    fi
}

# Run load tests
run_load_tests() {
    log_info "Running load tests with K6..."
    
    # Set environment variables for K6
    export BASE_URL
    export JARVIS_VOICE_URL
    export JARVIS_KNOWLEDGE_URL
    export JARVIS_AUTOMATION_URL
    export JARVIS_MULTIMODAL_URL
    export JARVIS_HARDWARE_URL
    
    cd "$PROJECT_DIR"
    
    # Run K6 load tests with baseline scenario
    if k6 run --scenario baseline_load "$TESTING_DIR/k6_load_tests.js"; then
        log_success "Load tests completed successfully"
        return 0
    else
        log_error "Load tests failed"
        return 1
    fi
}

# Generate test report
generate_report() {
    log_info "Generating test report..."
    
    local report_file="$RESULTS_DIR/test-summary.md"
    local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    
    cat > "$report_file" << EOF
# Jarvis Testing Suite Report

**Generated**: $timestamp

## Test Configuration
- API Tests: $RUN_API_TESTS
- E2E Tests: $RUN_E2E_TESTS  
- Load Tests: $RUN_LOAD_TESTS
- Backend URL: $BASE_URL
- Frontend URL: $FRONTEND_URL

## Service Endpoints Tested
- Jarvis Voice Interface: $JARVIS_VOICE_URL
- Jarvis Knowledge Management: $JARVIS_KNOWLEDGE_URL
- Jarvis Automation Agent: $JARVIS_AUTOMATION_URL
- Jarvis Multimodal AI: $JARVIS_MULTIMODAL_URL
- Jarvis Hardware Optimizer: $JARVIS_HARDWARE_URL

## Results
EOF
    
    if [ "$RUN_API_TESTS" = true ]; then
        if [ -f "$RESULTS_DIR/newman-results.json" ]; then
            echo "- ✅ API Tests: PASSED" >> "$report_file"
        else
            echo "- ❌ API Tests: FAILED" >> "$report_file"
        fi
    fi
    
    if [ "$RUN_E2E_TESTS" = true ]; then
        if [ -d "cypress/videos" ]; then
            echo "- ✅ E2E Tests: PASSED" >> "$report_file"
        else
            echo "- ❌ E2E Tests: FAILED" >> "$report_file"
        fi
    fi
    
    if [ "$RUN_LOAD_TESTS" = true ]; then
        if [ -f "load-test-results.json" ]; then
            echo "- ✅ Load Tests: PASSED" >> "$report_file"
        else
            echo "- ❌ Load Tests: FAILED" >> "$report_file"
        fi
    fi
    
    echo "" >> "$report_file"
    echo "## Generated Artifacts" >> "$report_file"
    
    # List generated files
    if [ -d "$RESULTS_DIR" ]; then
        echo "### API Test Results" >> "$report_file"
        find "$RESULTS_DIR" -name "*.html" -o -name "*.json" -o -name "*.xml" | sort | sed 's/^/- /' >> "$report_file"
    fi
    
    if [ -d "cypress" ]; then
        echo "### E2E Test Results" >> "$report_file"
        find cypress -name "*.mp4" -o -name "*.png" | sort | sed 's/^/- /' >> "$report_file"
    fi
    
    if [ -f "load-test-report.html" ]; then
        echo "### Load Test Results" >> "$report_file"
        echo "- load-test-report.html" >> "$report_file"
        echo "- load-test-results.json" >> "$report_file"
    fi
    
    log_success "Test report generated: $report_file"
}

# Cleanup function
cleanup() {
    if [ "$CLEANUP_AFTER" = true ]; then
        log_info "Cleaning up..."
        
        # Stop services if they were started by this script
        cd "$PROJECT_DIR"
        docker-compose down
        
        # Clean up temporary files
        rm -f cypress/e2e/jarvis_interface.cy.js 2>/dev/null || true
        
        log_success "Cleanup completed"
    else
        log_info "Skipping cleanup (--no-cleanup specified)"
    fi
}

# Signal handlers
trap cleanup EXIT
trap 'log_error "Script interrupted"; exit 130' INT TERM

# Main execution
main() {
    local start_time=$(date +%s)
    local exit_code=0
    
    log_info "Starting Jarvis Testing Suite"
    log_info "Project directory: $PROJECT_DIR"
    log_info "Test configuration: API=$RUN_API_TESTS, E2E=$RUN_E2E_TESTS, Load=$RUN_LOAD_TESTS"
    
    # Parse arguments
    parse_args "$@"
    
    # Check prerequisites
    check_prerequisites
    
    # Start services
    start_services
    
    # Perform health checks
    perform_health_checks
    
    # Run selected test suites
    if [ "$RUN_API_TESTS" = true ]; then
        if ! run_api_tests; then
            exit_code=1
        fi
    fi
    
    if [ "$RUN_E2E_TESTS" = true ]; then
        if ! run_e2e_tests; then
            exit_code=1
        fi
    fi
    
    if [ "$RUN_LOAD_TESTS" = true ]; then
        if ! run_load_tests; then
            exit_code=1
        fi
    fi
    
    # Generate report
    generate_report
    
    # Calculate execution time
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        log_success "All tests completed successfully in ${duration}s"
    else
        log_error "Some tests failed (exit code: $exit_code) after ${duration}s"
    fi
    
    exit $exit_code
}

# Run main function with all arguments
main "$@"